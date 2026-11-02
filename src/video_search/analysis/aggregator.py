"""
Frame Metadata Aggregator combining Object Detection and VLM/OCR results.
"""

from __future__ import annotations
from typing import List, Tuple
from collections import Counter
from datetime import datetime, timezone

from ..models import RawFrame, Detection, RegionCaption, OCRResult, FrameRecord


class FrameMetadataAggregator:
    """
    Combines object detections, dense captions, and OCR outputs into
    a unified FrameRecord with combined textual description.
    """

    @staticmethod
    def format_detection_summary(detections: List[Detection]) -> str:
        if not detections:
            return "No prominent objects detected"
        counts = Counter([d.class_name for d in detections])
        parts = [f"{count} {cls_name}{'s' if count > 1 and not cls_name.endswith('s') else ''}" for cls_name, count in counts.items()]
        return ", ".join(parts)

    def aggregate(
        self,
        frame: RawFrame,
        detections: List[Detection],
        vlm_output: Tuple[str, List[RegionCaption], List[OCRResult]],
        frame_image_path: str = None,
        segment_id: int = 0,
    ) -> FrameRecord:
        frame_caption, region_captions, ocr_results = vlm_output

        # OCR extraction
        ocr_texts = [res.text for res in ocr_results if res.text.strip()]
        all_ocr_text = " ".join(ocr_texts).strip()

        # Detection summary
        det_summary = self.format_detection_summary(detections)

        # Combined text representation
        combined_text = (
            f"Frame at {frame.timestamp_sec:.2f}s: {frame_caption.strip()} "
            f"Objects detected: {det_summary}. "
            f"Text visible: {all_ocr_text if all_ocr_text else 'None'}."
        ).strip()

        now_iso = datetime.now(timezone.utc).isoformat()

        return FrameRecord(
            video_id=frame.video_id,
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            timestamp_iso=frame.timestamp_iso,
            source_uri=frame.source_uri,
            width=frame.width,
            height=frame.height,
            detections=detections,
            frame_caption=frame_caption,
            region_captions=region_captions,
            ocr_results=ocr_results,
            ocr_texts=ocr_texts,
            ocr_full=all_ocr_text,
            detection_summary=det_summary,
            combined_text=combined_text,
            segment_id=segment_id,
            frame_image_path=frame_image_path,
            processing_timestamp=now_iso,
        )

    def aggregate_batch(
        self,
        frames: List[RawFrame],
        batch_detections: List[List[Detection]],
        batch_vlm: List[Tuple[str, List[RegionCaption], List[OCRResult]]],
        image_paths: List[str] = None,
        segment_ids: List[int] = None,
    ) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        for i, frame in enumerate(frames):
            img_path = image_paths[i] if image_paths else None
            seg_id = segment_ids[i] if segment_ids else 0
            rec = self.aggregate(
                frame=frame,
                detections=batch_detections[i],
                vlm_output=batch_vlm[i],
                frame_image_path=img_path,
                segment_id=seg_id,
            )
            records.append(rec)
        return records

