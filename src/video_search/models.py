"""
Data models and schemas for Video Search Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class IngestConfig:
    sample_strategy: str = "fixed_fps"
    sample_interval_sec: float = 1.0
    scene_threshold: float = 0.3
    min_interval_sec: 0.5 = 0.5
    queue_max_size: int = 64
    queue_overflow_policy: str = "block"
    ffmpeg_hwaccel: Optional[str] = None


@dataclass
class RawFrame:
    """Represents a decoded video frame with temporal metadata."""
    video_id: str
    frame_index: int
    timestamp_sec: float
    timestamp_iso: str
    image: np.ndarray  # BGR numpy array
    width: int
    height: int
    sampling_strategy: str
    source_fps: float
    source_uri: str

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        d = {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "timestamp_iso": self.timestamp_iso,
            "width": self.width,
            "height": self.height,
            "sampling_strategy": self.sampling_strategy,
            "source_fps": self.source_fps,
            "source_uri": self.source_uri,
        }
        if include_image:
            d["image"] = self.image
        return d


@dataclass
class Detection:
    """Bounding box detection from object detector."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]        # [x1, y1, x2, y2] in pixels
    bbox_normalized: List[float]  # [x1, y1, x2, y2] normalized to [0, 1]


@dataclass
class RegionCaption:
    """Dense captioning output for a specific image region."""
    caption: str
    bbox_xyxy: List[float]
    bbox_normalized: List[float]


@dataclass
class OCRResult:
    """Recognized text with location and confidence."""
    text: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_normalized: List[float]


@dataclass
class FrameRecord:
    """Aggregated scene understanding metadata for a single frame."""
    video_id: str
    frame_index: int
    timestamp_sec: float
    timestamp_iso: str
    source_uri: str
    width: int
    height: int

    # Object Detection
    detections: List[Detection] = field(default_factory=list)

    # Dense Captioning
    frame_caption: str = ""
    region_captions: List[RegionCaption] = field(default_factory=list)

    # OCR
    ocr_results: List[OCRResult] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)
    ocr_full: str = ""

    # Synthesized Fields
    detection_summary: str = ""
    combined_text: str = ""
    segment_id: int = 0
    frame_image_path: Optional[str] = None
    processing_timestamp: str = ""

    @property
    def all_ocr_text(self) -> str:
        return self.ocr_full

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoSegment:
    """Contiguous semantic interval in a video."""
    start_sec: float
    end_sec: float
    summary_text: str
    representative_frame_index: int
    key_entities: List[str] = field(default_factory=list)


@dataclass
class VideoSummary:
    """Chronological narrative and segment summary for an entire video."""
    video_id: str
    source_uri: str
    total_duration_sec: float
    total_frames_processed: int
    segments: List[VideoSegment] = field(default_factory=list)
    narrative: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """Result item returned by vector / hybrid search."""
    point_id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[Dict[str, List[float]]] = None


@dataclass
class TimestampReference:
    """Exact timestamp reference cited by RAG engine."""
    time_sec: float
    time_formatted: str
    description: str
    frame_index: int


@dataclass
class QueryResponse:
    """Structured response to a natural language video query."""
    query: str
    answer: str
    timestamps: List[TimestampReference] = field(default_factory=list)
    confidence: float = 0.0
    evidence_frames: List[Dict[str, Any]] = field(default_factory=list)
    video_summary_used: bool = False
    follow_up_suggestions: List[str] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

