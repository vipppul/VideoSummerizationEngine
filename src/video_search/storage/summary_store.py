"""
SQLite companion store for video summaries.
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..models import VideoSummary, VideoSegment
from ..utils.logging import setup_logger

logger = setup_logger("storage.summary_store")


class SQLiteSummaryStore:
    """
    Persistent SQLite storage for video narrative summaries and segments.
    """

    def __init__(self, db_path: str = "./data/summaries/summaries.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_summaries (
                    video_id       TEXT PRIMARY KEY,
                    source_uri     TEXT NOT NULL,
                    duration_sec   REAL,
                    total_frames   INTEGER,
                    narrative      TEXT,
                    segments_json  TEXT,
                    created_at     TEXT,
                    metadata_json  TEXT
                );
                """
            )
            conn.commit()

    def save_summary(self, summary: VideoSummary) -> None:
        segments_data = [
            {
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "summary_text": s.summary_text,
                "representative_frame_index": s.representative_frame_index,
                "key_entities": s.key_entities,
            }
            for s in summary.segments
        ]

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO video_summaries 
                (video_id, source_uri, duration_sec, total_frames, narrative, segments_json, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    summary.video_id,
                    summary.source_uri,
                    summary.total_duration_sec,
                    summary.total_frames_processed,
                    summary.narrative,
                    json.dumps(segments_data),
                    summary.created_at,
                    json.dumps({}),
                ),
            )
            conn.commit()
        logger.info(f"Saved summary for video '{summary.video_id}'")

    def get_summary(self, video_id: str) -> Optional[VideoSummary]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM video_summaries WHERE video_id = ?;", (video_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            raw_segments = json.loads(row["segments_json"] or "[]")
            segments = [
                VideoSegment(
                    start_sec=s["start_sec"],
                    end_sec=s["end_sec"],
                    summary_text=s["summary_text"],
                    representative_frame_index=s["representative_frame_index"],
                    key_entities=s.get("key_entities", []),
                )
                for s in raw_segments
            ]

            return VideoSummary(
                video_id=row["video_id"],
                source_uri=row["source_uri"],
                total_duration_sec=row["duration_sec"],
                total_frames_processed=row["total_frames"],
                segments=segments,
                narrative=row["narrative"],
                created_at=row["created_at"],
            )

    def list_all(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT video_id, source_uri, duration_sec, total_frames, created_at FROM video_summaries ORDER BY created_at DESC;"
            )
            return [dict(r) for r in cursor.fetchall()]

    def delete_summary(self, video_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM video_summaries WHERE video_id = ?;", (video_id,))
            conn.commit()
            logger.info(f"Deleted summary for video '{video_id}'")

