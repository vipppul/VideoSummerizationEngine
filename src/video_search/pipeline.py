"""
End-to-End Multimodal Video Processing Pipeline Orchestrator.
Coordinates Ingestion, Analysis, Embedding, and Storage.
"""

from __future__ import annotations
import asyncio
import os
import uuid
from typing import Dict, Any, Optional, List
import numpy as np

from .models import RawFrame, FrameRecord, VideoSummary, IngestConfig
from .config import SystemConfig
from .utils.logging import setup_logger
from .utils.ffmpeg import get_video_metadata

from .ingestion.extractor import FrameExtractor
from .ingestion.queue import BoundedFrameQueue
from .analysis.detector import ObjectDetector
from .analysis.vlm import VisionLanguageModel
from .analysis.aggregator import FrameMetadataAggregator
from .analysis.summarizer import GlobalSummaryBuilder
from .embedding.encoder import MultimodalEncoder
from .embedding.fuser import EmbeddingFuser
from .storage.vector_store import QdrantVectorStore
from .storage.summary_store import SQLiteSummaryStore
from .storage.frame_store import DiskFrameStore

logger = setup_logger("pipeline.orchestrator")


class PipelineOrchestrator:
    """
    Coordinates the execution of all 5 modules from video ingestion to storage.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

        # Storage components
        self.frame_store = DiskFrameStore(base_dir=self.config.app.frame_storage_path)
        self.summary_store = SQLiteSummaryStore(db_path=self.config.summary_store.sqlite_path)
        self.vector_store = QdrantVectorStore(
            settings=self.config.vector_db,
            vector_dim=self.config.embedding.dim,
        )

        # ML Inference components
        logger.info("Initializing Object Detector...")
        self.detector = ObjectDetector(self.config.detection)

        logger.info("Initializing Vision-Language Model...")
        self.vlm = VisionLanguageModel(self.config.vlm)

        logger.info("Initializing Multimodal Encoder & Fuser...")
        self.encoder = MultimodalEncoder(self.config.embedding)
        self.fuser = EmbeddingFuser(
            strategy=self.config.embedding.fusion_strategy,
            alpha=self.config.embedding.fusion_alpha,
        )

        self.aggregator = FrameMetadataAggregator()
        self.summarizer = GlobalSummaryBuilder()

        # Job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

    def get_job_status(self, video_id: str) -> Dict[str, Any]:
        if video_id in self.active_jobs:
            return self.active_jobs[video_id]
        count = self.vector_store.count_frames(video_id)
        has_summary = self.summary_store.get_summary(video_id) is not None
        if count > 0 or has_summary:
            return {
                "video_id": video_id,
                "status": "completed",
                "frames_indexed": count,
                "summary_available": has_summary,
            }
        return {
            "video_id": video_id,
            "status": "not_found",
            "frames_indexed": 0,
            "summary_available": False,
        }

    async def process_video_async(
        self,
        source_uri: str,
        video_id: Optional[str] = None,
        ingest_config: Optional[IngestConfig] = None,
    ) -> str:
        """
        Launches video processing in the background and returns video_id.
        """
        vid = video_id or f"vid_{uuid.uuid4().hex[:8]}"
        self.active_jobs[vid] = {
            "video_id": vid,
            "status": "processing",
            "frames_indexed": 0,
            "summary_available": False,
            "error": None,
        }
        asyncio.create_task(self._run_pipeline(source_uri, vid, ingest_config))
        return vid

    async def _run_pipeline(
        self,
        source_uri: str,
        video_id: str,
        ingest_config: Optional[IngestConfig] = None,
    ) -> None:
        cfg = ingest_config or IngestConfig(
            sample_strategy=self.config.ingestion.sample_strategy,
            sample_interval_sec=self.config.ingestion.sample_interval_sec,
            scene_threshold=self.config.ingestion.scene_threshold,
            min_interval_sec=self.config.ingestion.min_interval_sec,
            queue_max_size=self.config.ingestion.queue_max_size,
            queue_overflow_policy=self.config.ingestion.queue_overflow_policy,
        )

        metadata = get_video_metadata(source_uri)
        duration_sec = metadata.get("duration_sec", 0.0)

        extractor = FrameExtractor(cfg)
        queue = BoundedFrameQueue(
            maxsize=cfg.queue_max_size,
            overflow_policy=cfg.queue_overflow_policy,
        )

        # Launch ingestion producer task
        producer_task = asyncio.create_task(
            extractor.stream_frames(source_uri, video_id, queue)
        )

        batch_size = self.config.detection.batch_size
        current_batch: List[RawFrame] = []
        all_records: List[FrameRecord] = []

        try:
            while True:
                frame = await queue.get()
                if frame is not None:
                    current_batch.append(frame)
                    queue.task_done()

                # Process batch when full or at EOF
                if (len(current_batch) >= batch_size) or (frame is None and len(current_batch) > 0):
                    batch_records = await asyncio.to_thread(
                        self._process_batch_sync, current_batch
                    )
                    all_records.extend(batch_records)
                    self.active_jobs[video_id]["frames_indexed"] = len(all_records)
                    current_batch = []

                if frame is None:
                    break

            await producer_task

            # Build and persist global chronological summary
            logger.info(f"Building video narrative summary for '{video_id}'...")
            summary = self.summarizer.build_summary(
                records=all_records,
                video_id=video_id,
                source_uri=source_uri,
                total_duration_sec=duration_sec,
            )
            self.summary_store.save_summary(summary)

            self.active_jobs[video_id]["status"] = "completed"
            self.active_jobs[video_id]["summary_available"] = True
            logger.info(f"Video '{video_id}' successfully processed and indexed! Total frames: {len(all_records)}")

        except Exception as e:
            logger.error(f"Pipeline failure for video '{video_id}': {e}", exc_info=True)
            self.active_jobs[video_id]["status"] = "failed"
            self.active_jobs[video_id]["error"] = str(e)

    def _process_batch_sync(self, frames: List[RawFrame]) -> List[FrameRecord]:
        """
        Synchronous batch inference and storage execution.
        """
        if not frames:
            return []

        # 1. Save frame images to disk
        image_paths = []
        for f in frames:
            path = self.frame_store.save_frame(
                video_id=f.video_id,
                frame_index=f.frame_index,
                timestamp_sec=f.timestamp_sec,
                image=f.image,
            )
            image_paths.append(path)

        # 2. Object Detection (YOLO)
        detections = self.detector.detect_batch(frames)

        # 3. Dense Captioning & OCR (VLM)
        vlm_outputs = self.vlm.process_batch(frames)

        # 4. Aggregate into FrameRecords
        records = self.aggregator.aggregate_batch(
            frames=frames,
            batch_detections=detections,
            batch_vlm=vlm_outputs,
            image_paths=image_paths,
        )

        # 5. Multimodal Vectorization
        # Visual embedding
        image_vecs = self.encoder.embed_images([f.image for f in frames])
        # Text embedding
        texts = [r.combined_text for r in records]
        text_vecs = self.encoder.embed_texts(texts)
        # Fusion
        fused_vecs = self.fuser.fuse(image_vecs, text_vecs)

        # 6. Upsert to Qdrant
        self.vector_store.upsert_frames(
            records=records,
            fused_vecs=fused_vecs,
            image_vecs=image_vecs,
            text_vecs=text_vecs,
        )

        return records

    def delete_video(self, video_id: str) -> None:
        """Deletes all frames, vectors, and summaries for a given video."""
        self.vector_store.delete_video_frames(video_id)
        self.summary_store.delete_summary(video_id)
        self.frame_store.delete_video_frames(video_id)
        if video_id in self.active_jobs:
            del self.active_jobs[video_id]
        logger.info(f"Completely removed video '{video_id}' from the system")

