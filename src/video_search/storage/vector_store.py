"""
Qdrant Vector Database integration and schema management.
Supports multi-vector named collections, payload indexes, and hybrid filtering.
"""

from __future__ import annotations
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..models import FrameRecord, SearchResult
from ..config import VectorDBSettings
from ..utils.logging import setup_logger

logger = setup_logger("storage.vector_store")


class QdrantVectorStore:
    """
    Manages vector storage and search with Qdrant.
    Configured with triple named vectors ("fused", "image", "text")
    and indexed payload fields.
    """

    def __init__(self, settings: Optional[VectorDBSettings] = None, vector_dim: int = 512):
        self.settings = settings or VectorDBSettings()
        self.vector_dim = vector_dim
        self.collection_name = self.settings.collection_name
        self.client = self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        """
        Initializes Qdrant client. If server connection fails or use_memory is set,
        falls back gracefully to embedded local disk/memory Qdrant.
        """
        if self.settings.use_memory:
            if self.settings.storage_path and self.settings.storage_path != ":memory:":
                logger.info(f"Initializing Qdrant in local disk mode at '{self.settings.storage_path}'...")
                import os
                os.makedirs(self.settings.storage_path, exist_ok=True)
                try:
                    return QdrantClient(path=self.settings.storage_path)
                except RuntimeError as e:
                    if "already accessed by another instance" in str(e):
                        logger.warning(
                            f"Folder '{self.settings.storage_path}' is locked by another running process (such as the FastAPI server). "
                            f"Falling back to in-memory Qdrant client."
                        )
                        return QdrantClient(":memory:")
                    raise
            else:
                logger.info("Initializing Qdrant in in-memory mode...")
                return QdrantClient(":memory:")

        try:
            logger.info(f"Connecting to Qdrant server at {self.settings.host}:{self.settings.port}...")
            client = QdrantClient(
                host=self.settings.host,
                port=self.settings.port,
                grpc_port=self.settings.grpc_port if self.settings.use_grpc else None,
                prefer_grpc=self.settings.use_grpc,
                timeout=5.0,
            )
            # Ping
            client.get_collections()
            logger.info("Successfully connected to Qdrant server!")
            return client
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant server ({e}). "
                f"Falling back to embedded disk storage at '{self.settings.storage_path}'."
            )
            import os
            os.makedirs(self.settings.storage_path, exist_ok=True)
            return QdrantClient(path=self.settings.storage_path)

    def _ensure_collection(self) -> None:
        """Creates collection with named vectors and payload indices if not present."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
        except Exception:
            exists = False

        if not exists:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with named vectors...")
            vectors_config = {
                "fused": qmodels.VectorParams(size=self.vector_dim, distance=qmodels.Distance.COSINE),
                "image": qmodels.VectorParams(size=self.vector_dim, distance=qmodels.Distance.COSINE),
                "text":  qmodels.VectorParams(size=self.vector_dim, distance=qmodels.Distance.COSINE),
            }

            quant_config = None
            if self.settings.quantization == "int8":
                quant_config = qmodels.ScalarQuantization(
                    scalar=qmodels.ScalarQuantizationConfig(
                        type=qmodels.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                quantization_config=quant_config,
            )
            self._create_payload_indexes()
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def _create_payload_indexes(self) -> None:
        """Sets up payload indexes for high-speed filtering."""
        fields_to_index = [
            ("video_id", qmodels.PayloadSchemaType.KEYWORD),
            ("timestamp_sec", qmodels.PayloadSchemaType.FLOAT),
            ("detections[].class_name", qmodels.PayloadSchemaType.KEYWORD),
            ("ocr_texts", qmodels.PayloadSchemaType.KEYWORD),
            ("ocr_full", qmodels.PayloadSchemaType.TEXT),
            ("frame_caption", qmodels.PayloadSchemaType.TEXT),
            ("segment_id", qmodels.PayloadSchemaType.INTEGER),
        ]
        for field_name, field_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Payload index for '{field_name}' returned: {e}")

    def upsert_frames(
        self,
        records: List[FrameRecord],
        fused_vecs: np.ndarray,
        image_vecs: np.ndarray,
        text_vecs: np.ndarray,
    ) -> None:
        """
        Upserts a batch of FrameRecords with their multi-vector representations.
        """
        if not records:
            return

        points: List[qmodels.PointStruct] = []
        for i, rec in enumerate(records):
            point_id = str(uuid.uuid4())
            vector_dict = {
                "fused": fused_vecs[i].tolist(),
                "image": image_vecs[i].tolist(),
                "text":  text_vecs[i].tolist(),
            }
            payload = rec.to_dict()

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} frame points into collection '{self.collection_name}'")

    def search(
        self,
        query_vector: np.ndarray,
        vector_name: str = "fused",
        filter_obj: Optional[qmodels.Filter] = None,
        limit: int = 20,
        score_threshold: float = 0.15,
    ) -> List[SearchResult]:
        """
        Executes semantic search against the specified named vector with optional filters.
        Supports both modern query_points and legacy search APIs.
        """
        vec_list = query_vector.tolist()

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vec_list,
                using=vector_name,
                query_filter=filter_obj,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            raw_points = response.points
        else:
            raw_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=qmodels.NamedVector(
                    name=vector_name,
                    vector=vec_list,
                ),
                query_filter=filter_obj,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

        return [
            SearchResult(
                point_id=str(r.id),
                score=float(r.score),
                payload=dict(r.payload or {}),
            )
            for r in raw_points
        ]

    def delete_video_frames(self, video_id: str) -> None:
        """Deletes all frame points associated with a video_id."""
        filter_obj = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="video_id",
                    match=qmodels.MatchValue(value=video_id),
                )
            ]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(filter=filter_obj),
        )
        logger.info(f"Deleted points for video_id '{video_id}' from '{self.collection_name}'")

    def count_frames(self, video_id: Optional[str] = None) -> int:
        """Counts frames in the collection, optionally filtered by video_id."""
        filter_obj = None
        if video_id:
            filter_obj = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="video_id",
                        match=qmodels.MatchValue(value=video_id),
                    )
                ]
            )
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_obj,
        )
        return res.count
