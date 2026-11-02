"""
Module 1: Video Ingestion Pipeline.
"""

from .sampling import (
    SamplingGate,
    FixedFPSSampling,
    KeyframeSampling,
    SceneChangeSampling,
    AdaptiveSampling,
    create_sampling_gate,
)
from .queue import BoundedFrameQueue
from .extractor import FrameExtractor

__all__ = [
    "SamplingGate",
    "FixedFPSSampling",
    "KeyframeSampling",
    "SceneChangeSampling",
    "AdaptiveSampling",
    "create_sampling_gate",
    "BoundedFrameQueue",
    "FrameExtractor",
]

