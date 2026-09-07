/*
 * npu_server.c - persistent QNN HTP inference server (1 or 2 context binaries).
 *
 * Usage:
 *   npu_server <ctx1> <resp_fifo> <in_dims1> <out_specs1>
 *             [<ctx2> <in_dims2> <out_specs2>]
 *   in_dims  : e.g. "1,3,640,640"        (input tensor "images")
 *   out_specs: e.g. "output_0:1,38,8400"
 *              or  "output_0:1,116,2100;output_1:1,32,80,80"
 *
 * Protocol (stdin -> request, FIFO <- response):
 *   request : for each model: u32 input_len + input bytes
 *   response: u32 n_out_total, then for each output: u32 len + bytes
 */
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

#include "QnnInterface.h"

typedef Qnn_ErrorHandle_t (*GetProvidersFn_t)(const QnnInterface_t ***providerList,
                                              uint32_t *numProviders);

#define MAX_MODELS 2
#define MAX_OUT 4
#define MAX_DIMS 4

typedef struct {
    char name[64];
    uint32_t dims[MAX_DIMS];
    uint32_t rank;
    size_t nbytes;
    void *buf;
    Qnn_Tensor_t t;
} TensorSpec;

typedef struct {
    Qnn_ContextHandle_t ctx;
    Qnn_GraphHandle_t graph;
    Qnn_Tensor_t inT;
    uint32_t inDims[MAX_DIMS];
    uint32_t inRank;
    size_t inBytes;
    void *inBuf;
    TensorSpec outs[MAX_OUT];
    int n_out;
} Model;

static const QnnInterface_t *g_iface;
static char g_logbuf[1 << 20];
static size_t g_loglen;

static void fail(const char *m) {
    fprintf(stderr, "NPU_SERVER_ERROR: %s\n", m);
    exit(1);
}

#define STEP(msg) fprintf(stderr, "STEP: %s\n", msg)

static void log_cb(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    (void)level; (void)timestamp;
    if (g_loglen < sizeof(g_logbuf) - 1) {
        int n = vsnprintf(g_logbuf + g_loglen, sizeof(g_logbuf) - g_loglen, fmt, args);
        if (n > 0) {
            g_loglen += (size_t)n;
            if (g_loglen > sizeof(g_logbuf) - 1) g_loglen = sizeof(g_logbuf) - 1;
        }
    }
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

/* 后端报错里会给出图张量的真实 ID（每次加载可能不同），按名字解析回来。 */
static void discover_tensor_ids(Model *m) {
    char *p = g_logbuf;
    while ((p = strstr(p, "Tensor ID: ")) != NULL) {
        unsigned id = 0;
        char name[64] = {0};
        if (sscanf(p, "Tensor ID: %u, Name: %63[^,]", &id, name) == 2) {
            if (strcmp(name, "images") == 0) {
                m->inT.v2.id = id;
            } else {
                for (int i = 0; i < m->n_out; i++) {
                    if (strcmp(name, m->outs[i].name) == 0) {
                        m->outs[i].t.v2.id = id;
                        break;
                    }
                }
            }
            fprintf(stderr, "discovered %s -> id %u\n", name, id);
        }
        p += 10;
    }
}

static int count_known_ids(Model *m) {
    int n = (m->inT.v2.id != 0);
    for (int i = 0; i < m->n_out; i++) n += (m->outs[i].t.v2.id != 0);
    return n;
}

static int parse_dims(const char *s, uint32_t *dims, uint32_t *rank, size_t *nbytes) {
    int r = 0;
    *nbytes = 4;
    const char *p = s;
    while (*p && r < MAX_DIMS) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        dims[r++] = (uint32_t)v;
        *nbytes *= (size_t)v;
        p = end;
        if (*p == ',') p++;
        else break;
    }
    *rank = (uint32_t)r;
    return r;
}

static int parse_out_specs(const char *s, TensorSpec *outs, int maxn) {
    int n = 0;
    const char *p = s;
    while (*p && n < maxn) {
        const char *colon = strchr(p, ':');
        if (!colon) break;
        size_t nl = (size_t)(colon - p);
        if (nl >= sizeof(outs[n].name)) nl = sizeof(outs[n].name) - 1;
        memcpy(outs[n].name, p, nl);
        outs[n].name[nl] = 0;
        size_t nb = 0;
        if (parse_dims(colon + 1, outs[n].dims, &outs[n].rank, &nb) < 1) break;
        outs[n].nbytes = nb;
        n++;
        p = colon + 1;
        while (*p && *p != ';') p++;
        if (*p == ';') p++;
    }
    return n;
}

/* 从上下文二进制里扫描 graph_ 开头的名字，逐个尝试 graphRetrieve。 */
static int retrieve_graph(Qnn_ContextHandle_t ctx, const char *bin, size_t sz,
                          Qnn_GraphHandle_t *graph) {
    const char *end = bin + sz;
    const char *p = bin;
    while ((p = memchr(p, 'g', (size_t)(end - p))) != NULL) {
        if ((size_t)(end - p) > 6 && strncmp(p, "graph_", 6) == 0) {
            const char *q = p + 6;
            size_t n = 0;
            while (q + n < end && (isalnum((unsigned char)q[n]) || q[n] == '_')) n++;
            if (n >= 4) {
                char cand[96];
                size_t cl = n + 6;
                if (cl >= sizeof(cand)) cl = sizeof(cand) - 1;
                memcpy(cand, p, cl);
                cand[cl] = 0;
                if (g_iface->QNN_INTERFACE_VER_NAME.graphRetrieve(ctx, cand, graph)
                        == QNN_SUCCESS) {
                    fprintf(stderr, "graph: %s\n", cand);
                    return 1;
                }
            }
        }
        p++;
    }
    return 0;
}

static void tensor_setup(Qnn_Tensor_t *t, uint32_t id, const char *name,
                         uint32_t *dims, uint32_t rank, Qnn_TensorType_t type,
                         void *buf, uint32_t dataSize) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_2;
    t->v2.id = id;
    t->v2.name = name;
    t->v2.type = type;
    t->v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->v2.dataType = QNN_DATATYPE_FLOAT_32;
    t->v2.rank = rank;
    t->v2.dimensions = dims;
    t->v2.memType = QNN_TENSORMEMTYPE_RAW;
    t->v2.clientBuf.data = buf;
    t->v2.clientBuf.dataSize = dataSize;
}

static int read_exact(void *dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(0, (char *)dst + got, n - got);
        if (r <= 0) return r == 0 ? 0 : -1;
        got += (size_t)r;
    }
    return 1;
}

static void send_resp(int ofd, const void *buf, size_t len) {
    uint32_t l = (uint32_t)len;
    if (write(ofd, &l, 4) != 4) fail("resp len");
    if (write(ofd, buf, len) != (ssize_t)len) fail("resp data");
}

static void setup_model(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                        Qnn_LogHandle_t logger, const char *ctx_path,
                        const char *in_dims_str, const char *out_specs_str,
                        Model *m) {
    (void)logger;
    memset(m, 0, sizeof(*m));

    FILE *f = fopen(ctx_path, "rb");
    if (!f) fail("open ctx");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *bin = malloc((size_t)sz);
    if (fread(bin, 1, (size_t)sz, f) != (size_t)sz) fail("read ctx");
    fclose(f);

    Qnn_ErrorHandle_t rc = g_iface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            backend, device, NULL, bin, (Qnn_ContextBinarySize_t)sz, &m->ctx, NULL);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "contextCreateFromBinary(%s) rc=0x%x\n", ctx_path, (unsigned)rc);
        fail("contextCreateFromBinary");
    }
    STEP("contextCreateFromBinary ok");

    if (!retrieve_graph(m->ctx, bin, (size_t)sz, &m->graph)) fail("graphRetrieve");
    STEP("graphRetrieve ok");
    free(bin);

    size_t inBytes = 0;
    if (parse_dims(in_dims_str, m->inDims, &m->inRank, &inBytes) < 1) fail("bad in_dims");
    m->n_out = parse_out_specs(out_specs_str, m->outs, MAX_OUT);
    if (m->n_out < 1) fail("bad out_specs");

    if (posix_memalign(&m->inBuf, 4096, inBytes) != 0) fail("alloc in");
    m->inBytes = inBytes;
    tensor_setup(&m->inT, 0, "images", m->inDims, m->inRank, QNN_TENSOR_TYPE_APP_WRITE,
                 m->inBuf, (uint32_t)inBytes);
    for (int i = 0; i < m->n_out; i++) {
        if (posix_memalign(&m->outs[i].buf, 4096, m->outs[i].nbytes) != 0) fail("alloc out");
        tensor_setup(&m->outs[i].t, 0, m->outs[i].name, m->outs[i].dims,
                     m->outs[i].rank, QNN_TENSOR_TYPE_APP_READ,
                     m->outs[i].buf, (uint32_t)m->outs[i].nbytes);
    }

    Qnn_Tensor_t *outTensors = calloc((size_t)m->n_out, sizeof(Qnn_Tensor_t));
    for (int i = 0; i < m->n_out; i++) outTensors[i] = m->outs[i].t;

    /* 迭代探测：后端一次最多报两个缺失张量，把已发现的 ID 填回去再探。 */
    int prev_known = -1;
    rc = (Qnn_ErrorHandle_t)-1;
    for (int attempt = 0; attempt < 10; attempt++) {
        g_loglen = 0;
        rc = g_iface->QNN_INTERFACE_VER_NAME.graphExecute(
                m->graph, &m->inT, 1, outTensors, (uint32_t)m->n_out, NULL, NULL);
        if (rc == QNN_SUCCESS) {
            STEP("id discovery ok");
            break;
        }
        discover_tensor_ids(m);
        for (int i = 0; i < m->n_out; i++) outTensors[i] = m->outs[i].t;
        int known = count_known_ids(m);
        if (known == prev_known) {
            fprintf(stderr, "no new ids at attempt %d (known=%d)\n", attempt, known);
            break;
        }
        prev_known = known;
    }
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "graphExecute probe rc=0x%x\n", (unsigned)rc);
        fail("graphExecute probe");
    }
    free(outTensors);
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 8) {
        fail("usage: npu_server <ctx1> <fifo> <in_dims1> <out_specs1> "
             "[<ctx2> <in_dims2> <out_specs2>]");
    }

    void *h = dlopen("/usr/lib/libQnnHtp.so", RTLD_NOW);
    if (!h) fail(dlerror());
    GetProvidersFn_t gp = (GetProvidersFn_t)dlsym(h, "QnnInterface_getProviders");
    if (!gp) fail("dlsym getProviders");

    const QnnInterface_t **provs = NULL;
    uint32_t nprov = 0;
    if (gp(&provs, &nprov) != QNN_SUCCESS || nprov < 1) fail("getProviders");
    g_iface = provs[0];

    Qnn_BackendHandle_t backend = NULL;
    Qnn_LogHandle_t logger = NULL;
    if (g_iface->QNN_INTERFACE_VER_NAME.logCreate(log_cb, QNN_LOG_LEVEL_INFO, &logger)
            != QNN_SUCCESS) fail("logCreate");
    if (g_iface->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend) != QNN_SUCCESS)
        fail("backendCreate");
    STEP("backendCreate ok");

    Qnn_DeviceHandle_t device = NULL;
    if (g_iface->QNN_INTERFACE_VER_NAME.deviceCreate(logger, NULL, &device) != QNN_SUCCESS)
        fail("deviceCreate");
    STEP("deviceCreate ok");

    Model models[MAX_MODELS];
    int n_models = 1;
    setup_model(backend, device, logger, argv[1], argv[3], argv[4], &models[0]);
    if (argc == 8) {
        n_models = 2;
        setup_model(backend, device, logger, argv[5], argv[6], argv[7], &models[1]);
    }

    int ofd = open(argv[2], O_WRONLY);
    if (ofd < 0) {
        fprintf(stderr, "open fifo %s: %s\n", argv[2], strerror(errno));
        fail("open resp fifo");
    }
    fprintf(stderr, "NPU_SERVER_READY models=%d\n", n_models);
    fflush(stderr);

    while (1) {
        for (int mm = 0; mm < n_models; mm++) {
            uint32_t len = 0;
            int r = read_exact(&len, 4);
            if (r == 0) return 0;
            if (r < 0) fail("stdin");
            if (len != models[mm].inBytes) fail("input size");
            if (read_exact(models[mm].inBuf, models[mm].inBytes) != 1) fail("stdin data");
        }
        fprintf(stderr, "DBG inputs read ok\n");
        fflush(stderr);

        for (int mm = 0; mm < n_models; mm++) {
            fprintf(stderr, "DBG exec model%d\n", mm);
            fflush(stderr);
            Qnn_Tensor_t *outTensors = calloc((size_t)models[mm].n_out, sizeof(Qnn_Tensor_t));
            for (int i = 0; i < models[mm].n_out; i++) outTensors[i] = models[mm].outs[i].t;
            Qnn_ErrorHandle_t rc = g_iface->QNN_INTERFACE_VER_NAME.graphExecute(
                    models[mm].graph, &models[mm].inT, 1,
                    outTensors, (uint32_t)models[mm].n_out, NULL, NULL);
            if (rc != QNN_SUCCESS) {
                fprintf(stderr, "graphExecute model%d rc=0x%x\n", mm, (unsigned)rc);
                fail("graphExecute");
            }
            free(outTensors);
        }
        fprintf(stderr, "DBG execs done\n");
        fflush(stderr);

        uint32_t total = 0;
        for (int mm = 0; mm < n_models; mm++) total += (uint32_t)models[mm].n_out;
        if (write(ofd, &total, 4) != 4) fail("resp count");
        for (int mm = 0; mm < n_models; mm++) {
            for (int i = 0; i < models[mm].n_out; i++) {
                send_resp(ofd, models[mm].outs[i].buf, models[mm].outs[i].nbytes);
            }
        }
    }
}
