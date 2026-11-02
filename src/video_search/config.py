"""
Configuration loader with Pydantic schema validation.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "video-search-engine"
    log_level: str = "INFO"
    frame_storage_path: str = "./data/frames"
    summary_storage_path: str = "./data/summaries"


class IngestionSettings(BaseModel):
    sample_strategy: str = "fixed_fps"
    sample_interval_sec: float = 1.0
    scene_threshold: float = 0.3
    min_interval_sec: float = 0.5
    queue_max_size: int = 64
    queue_overflow_policy: str = "block"
    ffmpeg_hwaccel: Optional[str] = None


class DetectionSettings(BaseModel):
    model: str = "yolov8n.pt"
    device: str = "cuda:0"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    input_size: int = 640
    target_classes: Optional[List[int]] = None
    batch_size: int = 16


class VLMSettings(BaseModel):
    model: str = "microsoft/Florence-2-base"
    device: str = "cuda:0"
    max_new_tokens: int = 512
    batch_size: int = 4
    tasks: List[str] = Field(
        default_factory=lambda: [
            "MORE_DETAILED_CAPTION",
            "OCR_WITH_REGION",
        ]
    )
    parallel_tasks: bool = False
    fallback_to_heuristic: bool = True


class EmbeddingSettings(BaseModel):
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    dim: int = 512
    device: str = "cuda:0"
    batch_size: int = 32
    fusion_strategy: str = "weighted_average"
    fusion_alpha: float = 0.6


class VectorDBSettings(BaseModel):
    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection_name: str = "video_frames"
    use_grpc: bool = False
    use_memory: bool = True
    storage_path: str = "./data/qdrant_db"
    quantization: str = "none"


class SummaryStoreSettings(BaseModel):
    backend: str = "sqlite"
    sqlite_path: str = "./data/summaries/summaries.db"


class QuerySettings(BaseModel):
    default_top_k: int = 20
    reranker_enabled: bool = True
    reranker_type: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 8
    score_threshold: float = 0.15
    temporal_cluster_window_sec: float = 5.0
    rrf_k: int = 60


class LLMSettings(BaseModel):
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GOOGLE_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.1


class SystemConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    summary_store: SummaryStoreSettings = Field(default_factory=SummaryStoreSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Loads configuration from YAML file, merges environment overrides,
    and returns a validated SystemConfig object.
    """
    if config_path is None:
        env_config = os.getenv("CONFIG_PATH")
        if env_config and os.path.exists(env_config):
            config_path = env_config
        else:
            default_path = Path("config/default.yaml")
            if default_path.exists():
                config_path = str(default_path)

    data: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # Environment variable overrides
    if "GOOGLE_API_KEY" in os.environ:
        pass  # Loaded dynamically by LLM engine
    if "QDRANT_HOST" in os.environ:
        data.setdefault("vector_db", {})["host"] = os.environ["QDRANT_HOST"]
    if "QDRANT_PORT" in os.environ:
        data.setdefault("vector_db", {})["port"] = int(os.environ["QDRANT_PORT"])
    if "QDRANT_USE_MEMORY" in os.environ:
        data.setdefault("vector_db", {})["use_memory"] = os.environ["QDRANT_USE_MEMORY"].lower() in ("true", "1")
    if "LOG_LEVEL" in os.environ:
        data.setdefault("app", {})["log_level"] = os.environ["LOG_LEVEL"]

    return SystemConfig(**data)

