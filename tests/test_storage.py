"""
Unit tests for Module 4: Vector and Summary Storage.
"""

import numpy as np
from video_search.config import VectorDBSettings
from video_search.storage.vector_store import QdrantVectorStore
from video_search.storage.summary_store import SQLiteSummaryStore
from video_search.models import FrameRecord, VideoSummary, VideoSegment, Detection


def test_qdrant_in_memory_upsert_and_search():
    settings = VectorDBSettings(
        use_memory=True,
        storage_path=":memory:",
        collection_name="test_video_frames",
    )
    store = QdrantVectorStore(settings=settings, vector_dim=4)

    rec = FrameRecord(
        video_id="test_vid_01",
        frame_index=1,
        timestamp_sec=2.5,
        timestamp_iso="2026-11-02T12:00:00Z",
        source_uri="test.mp4",
        width=640,
        height=480,
        detections=[Detection("car", 2, 0.95, [0, 0, 50, 50], [0, 0, 0.1, 0.1])],
        frame_caption="A blue car driving down a suburban street",
        ocr_texts=["ABC-1234"],
        ocr_full="ABC-1234",
    )

    fused_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    img_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    txt_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    store.upsert_frames([rec], fused_vec, img_vec, txt_vec)

    # Search with identical vector
    results = store.search(query_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert len(results) == 1
    assert results[0].payload["video_id"] == "test_vid_01"
    assert results[0].payload["frame_caption"] == "A blue car driving down a suburban street"
    assert results[0].score > 0.99


def test_sqlite_summary_store(tmp_path):
    db_file = tmp_path / "test_summaries.db"
    store = SQLiteSummaryStore(db_path=str(db_file))

    summary = VideoSummary(
        video_id="vid_test_summary",
        source_uri="sample.mp4",
        total_duration_sec=30.0,
        total_frames_processed=15,
        segments=[
            VideoSegment(0.0, 15.0, "Car driving", 0, ["car"]),
            VideoSegment(15.0, 30.0, "Pedestrian crossing", 8, ["person"]),
        ],
        narrative="[00:00 - 00:15] Car driving\n[00:15 - 00:30] Pedestrian crossing",
    )

    store.save_summary(summary)
    retrieved = store.get_summary("vid_test_summary")

    assert retrieved is not None
    assert retrieved.video_id == "vid_test_summary"
    assert len(retrieved.segments) == 2
    assert retrieved.segments[1].summary_text == "Pedestrian crossing"
