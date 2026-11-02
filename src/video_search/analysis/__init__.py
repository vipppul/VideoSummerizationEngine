"""
Module 2: Dense Scene Understanding & OCR.
"""

from .detector import ObjectDetector
from .vlm import VisionLanguageModel
from .aggregator import FrameMetadataAggregator
from .summarizer import GlobalSummaryBuilder

__all__ = [
    "ObjectDetector",
    "VisionLanguageModel",
    "FrameMetadataAggregator",
    "GlobalSummaryBuilder",
]

