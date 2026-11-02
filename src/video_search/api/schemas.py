"""
FastAPI Pydantic request and response schemas.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_uri: str = Field(..., description="Path to video file or RTSP stream URI")
    video_id: Optional[str] = Field(None, description="Optional custom video ID")
    sample_strategy: Optional[str] = Field("fixed_fps", description="fixed_fps, keyframe, scene_change, adaptive")
    sample_interval_sec: Optional[float] = Field(1.0, description="Sampling interval in seconds")


class IngestResponse(BaseModel):
    video_id: str
    status: str
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    video_id: Optional[str] = Field(None, description="Optional target video ID filter")
    top_k: Optional[int] = Field(20, description="Number of results to retrieve")


class SearchItem(BaseModel):
    point_id: str
    score: float
    video_id: str
    frame_index: int
    timestamp_sec: float
    timestamp_formatted: str
    frame_caption: str
    detection_summary: str
    ocr_full: str
    frame_image_path: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchItem]


class AskRequest(BaseModel):
    query: str = Field(..., description="Natural language query to be answered by RAG")
    video_id: Optional[str] = Field(None, description="Optional target video ID")
    top_k: Optional[int] = Field(20, description="Number of candidate frames to consider")


class TimestampItem(BaseModel):
    time_sec: float
    time_formatted: str
    description: str
    frame_index: int


class AskResponse(BaseModel):
    query: str
    answer: str
    timestamps: List[TimestampItem]
    confidence: float
    follow_up_suggestions: List[str]
    evidence_frames_count: int
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    frames_indexed: int
    summary_available: bool


class VideoSummaryResponse(BaseModel):
    video_id: str
    source_uri: str
    duration_sec: float
    total_frames: int
    narrative: str
    segments: List[Dict[str, Any]]
    short_summary: Optional[str] = None
    key_entities: List[str] = Field(default_factory=list)

