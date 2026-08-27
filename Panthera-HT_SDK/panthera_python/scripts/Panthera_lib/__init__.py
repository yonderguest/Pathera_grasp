"""Panthera-HT arm library and visual grasp helpers."""

__version__ = "1.0.0"
__author__ = "HighTorque Robotics"

__all__ = [
    "Panthera",
    "GraspConfig",
    "GraspPlanner",
    "VisionStreamer",
    "GraspNetCandidateProvider",
    "NpuYoloDetector",
]


def __getattr__(name):
    if name == "Panthera":
        from .Panthera import Panthera

        return Panthera
    if name == "GraspConfig":
        from .grasp_config import GraspConfig

        return GraspConfig
    if name == "GraspPlanner":
        from .grasp_planner import GraspPlanner

        return GraspPlanner
    if name == "VisionStreamer":
        from .vision_streamer import VisionStreamer

        return VisionStreamer
    if name == "GraspNetCandidateProvider":
        from .graspnet_pipeline import GraspNetCandidateProvider

        return GraspNetCandidateProvider
    if name == "NpuYoloDetector":
        from .npu_inference import NpuYoloDetector

        return NpuYoloDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
