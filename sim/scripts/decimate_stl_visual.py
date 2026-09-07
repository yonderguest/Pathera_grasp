"""将超过 MuJoCo 面数上限的二进制 STL 转为仅视觉用途的降面网格。

不依赖 Blender、MeshLab 或额外 pip 包。使用确定性的顶点体素聚类，而不是
随意抽样删除三角面：相邻三角形的顶点会一起合并，再剔除退化/重复面，从而尽量
保留连续外壳。原始 STL 不会被修改。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


_STL_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def read_binary_stl(path: Path) -> np.ndarray:
    """读取经过长度校验的二进制 STL，返回 N×3×3 顶点数组。"""
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"STL 文件过小：{path}")
    face_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + face_count * _STL_RECORD.itemsize
    if len(raw) != expected:
        raise ValueError(
            f"仅支持二进制 STL，{path} 长度 {len(raw)} 与头部三角数 {face_count} 不匹配"
        )
    return np.frombuffer(raw, dtype=_STL_RECORD, offset=84).copy()["vertices"]


def clustered_faces(vertices: np.ndarray, voxel_m: float) -> np.ndarray:
    """以体素聚类重建三角面，保留首次出现的三角形绕序。"""
    flat = vertices.reshape(-1, 3)
    origin = flat.min(axis=0)
    keys = np.rint((flat - origin) / voxel_m).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    # 每个体素以其中原始顶点的平均值代表，视觉表面比直接吸附网格中心更平滑。
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, flat)
    counts = np.bincount(inverse, minlength=len(unique_keys))
    merged_vertices = (sums / counts[:, None]).astype(np.float32)

    faces = inverse.reshape(-1, 3)
    non_degenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[non_degenerate]

    # 同一面可能因体素合并重复；以无方向索引去重，但保留首个面的绕序。
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    return merged_vertices[faces]


def write_binary_stl(path: Path, faces: np.ndarray) -> None:
    """写入标准二进制 STL，并按降面后的顶点重新计算法向。"""
    edge_a = faces[:, 1] - faces[:, 0]
    edge_b = faces[:, 2] - faces[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    faces = faces[valid]
    normals = normals[valid] / lengths[valid, None]

    records = np.zeros(len(faces), dtype=_STL_RECORD)
    records["normal"] = normals
    records["vertices"] = faces
    header = b"pathera_grasp visual mesh, voxel clustered".ljust(80, b" ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(records)))
        handle.write(records.tobytes())


def decimate(source: Path, destination: Path, max_faces: int) -> tuple[int, int, float]:
    """自动寻找保留细节最多、且不超过 max_faces 的体素尺寸。"""
    original = read_binary_stl(source)
    source_faces = len(original)
    if source_faces <= max_faces:
        write_binary_stl(destination, original)
        return source_faces, source_faces, 0.0

    extent_m = float(np.ptp(original.reshape(-1, 3), axis=0).max())
    voxel_m = extent_m / 6000.0
    reduced = original
    while len(reduced) > max_faces:
        reduced = clustered_faces(original, voxel_m)
        voxel_m *= 1.35
        if voxel_m > extent_m:
            raise RuntimeError("体素尺寸超过模型范围，无法生成有效视觉网格")

    write_binary_stl(destination, reduced)
    return source_faces, len(reduced), voxel_m / 1.35


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-faces", type=int, default=180_000)
    args = parser.parse_args()
    if args.max_faces < 4:
        raise ValueError("--max-faces 必须不小于 4")
    before, after, voxel_m = decimate(args.source, args.destination, args.max_faces)
    print(f"STL_DECIMATION source_faces={before} output_faces={after} voxel_m={voxel_m:.8f}")


if __name__ == "__main__":
    main()
