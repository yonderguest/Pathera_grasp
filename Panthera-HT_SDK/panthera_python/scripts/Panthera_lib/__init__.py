"""Panthera-HT arm library and visual grasp helpers."""

__version__ = "1.0.0"
__author__ = "HighTorque Robotics"

__all__ = ["Panthera", "GraspConfig", "GraspPlanner", "VisionStreamer"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
