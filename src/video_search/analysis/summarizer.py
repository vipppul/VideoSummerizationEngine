"""
Global Summary Builder for generating chronological video narratives.
"""

from __future__ import annotations
from typing import List
from datetime import datetime, timezone

from ..models import FrameRecord, VideoSegment, VideoSummary
from ..utils.ffmpeg import format_timestamp


class GlobalSummaryBuilder:
    """
    Builds chronological video summaries and semantic segments from FrameRecords.
    """

    def __init__(self, segment_time_window_sec: float = 15.0):
        self.segment_time_window_sec = segment_time_window_sec

    def build_summary(
        self,
        records: List[FrameRecord],
        video_id: str,
        source_uri: str = "",
        total_duration_sec: float = 0.0,
    ) -> VideoSummary:
        if not records:
            return VideoSummary(
                video_id=video_id,
                source_uri=source_uri,
                total_duration_sec=total_duration_sec,
                total_frames_processed=0,
                segments=[],
                narrative="No frames processed for this video.",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        # Sort chronologically
        sorted_records = sorted(records, key=lambda r: r.timestamp_sec)
        if total_duration_sec <= 0:
            total_duration_sec = sorted_records[-1].timestamp_sec

        # Cluster records into segments based on time window
        segments: List[VideoSegment] = []
        current_cluster: List[FrameRecord] = [sorted_records[0]]

        for rec in sorted_records[1:]:
            time_diff = rec.timestamp_sec - current_cluster[0].timestamp_sec
            if time_diff < self.segment_time_window_sec:
                current_cluster.append(rec)
            else:
                segments.append(self._summarize_segment(current_cluster))
                current_cluster = [rec]

        if current_cluster:
            segments.append(self._summarize_segment(current_cluster))

        # Compose narrative
        narrative_lines: List[str] = []
        for seg in segments:
            start_str = format_timestamp(seg.start_sec)[:8]  # HH:MM:SS
            end_str = format_timestamp(seg.end_sec)[:8]
            narrative_lines.append(f"[{start_str} - {end_str}] {seg.summary_text}")

        full_narrative = "\n".join(narrative_lines)

        return VideoSummary(
            video_id=video_id,
            source_uri=source_uri,
            total_duration_sec=total_duration_sec,
            total_frames_processed=len(sorted_records),
            segments=segments,
            narrative=full_narrative,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _summarize_segment(self, cluster: List[FrameRecord]) -> VideoSegment:
        start_sec = cluster[0].timestamp_sec
        end_sec = cluster[-1].timestamp_sec

        # Choose the representative frame with the most information
        best_rec = max(
            cluster,
            key=lambda r: len(r.detections) + len(r.ocr_results) + len(r.frame_caption),
        )

        # Collect unique entities across the cluster
        entities = set()
        for r in cluster:
            for d in r.detections:
                entities.add(d.class_name)
            for ocr in r.ocr_texts:
                if len(ocr) > 2:
                    entities.add(f"Text:'{ocr}'")

        summary_parts = []
        if best_rec.frame_caption:
            summary_parts.append(best_rec.frame_caption)
        if best_rec.detection_summary and best_rec.detection_summary != "No prominent objects detected":
            summary_parts.append(f"Featuring {best_rec.detection_summary}.")
        ocr_text = getattr(best_rec, "ocr_full", "") or getattr(best_rec, "all_ocr_text", "")
        if ocr_text:
            summary_parts.append(f"Detected text: {ocr_text}.")

        summary_text = " ".join(summary_parts) if summary_parts else f"Activity recorded between {start_sec:.1f}s and {end_sec:.1f}s."

        return VideoSegment(
            start_sec=start_sec,
            end_sec=end_sec,
            summary_text=summary_text,
            representative_frame_index=best_rec.frame_index,
            key_entities=sorted(list(entities)),
        )

