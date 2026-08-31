# -*- coding: utf-8 -*-
"""识别组件异常定义"""


class ASRError(Exception):
    """语音识别基础异常"""


class ModelNotFoundError(ASRError):
    """模型文件不存在"""


class AudioReadError(ASRError):
    """音频读取/录音失败"""


class RecognitionError(ASRError):
    """识别推理失败"""
