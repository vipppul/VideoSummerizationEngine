"""
Query analyzer and filter extraction for natural language queries.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from qdrant_client.http import models as qmodels


@dataclass
class AnalyzedQuery:
    original_query: str
    search_text: str
    vector_name: str  # "fused" | "image" | "text"
    filters: Optional[qmodels.Filter]
    time_range: Optional[Tuple[float, float]]
    target_video_id: Optional[str]
    target_ocr: Optional[str]
    top_k: int = 20


class QueryAnalyzer:
    """
    Parses natural language queries to extract intent, structured filters,
    and routing targets.
    """

    def analyze(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: int = 20,
    ) -> AnalyzedQuery:
        search_text = query.strip()
        extracted_video_id = video_id
        time_range: Optional[Tuple[float, float]] = None
        target_ocr: Optional[str] = None
        target_class: Optional[str] = None
        vector_name = "fused"

        # 1. Video ID extraction from query (e.g. "in video cam01")
        vid_match = re.search(r'\bin video\s+([A-Za-z0-9_-]+)', search_text, re.IGNORECASE)
        if vid_match:
            extracted_video_id = vid_match.group(1)
            search_text = search_text.replace(vid_match.group(0), "").strip()

        # 2. Time range extraction (e.g. "between 10s and 30s" or "between 10 and 30 seconds")
        time_between = re.search(r'\bbetween\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s+and\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b', search_text, re.IGNORECASE)
        if time_between:
            t1, t2 = float(time_between.group(1)), float(time_between.group(2))
            time_range = (min(t1, t2), max(t1, t2))
            search_text = search_text[:time_between.start()] + search_text[time_between.end():]
            search_text = re.sub(r'\s+', ' ', search_text).strip()

        # Check timestamp pattern e.g. "at 01:23" or "at 45s"
        at_time = re.search(r'\bat\s+(?:(\d+):)?(\d+)(?:\.(\d+))?\s*(?:s|sec|seconds)?\b', search_text, re.IGNORECASE)
        if at_time and not time_range:
            mins = int(at_time.group(1)) if at_time.group(1) else 0
            secs = float(at_time.group(2))
            center = mins * 60 + secs
            time_range = (max(0.0, center - 5.0), center + 5.0)

        # 3. License plate / specific OCR text extraction
        # Matches quoted text or patterns like ABC-1234, TRUCK-99, or "plate ABC-1234"
        plate_match = re.search(r'\b(?:license plate|plate)\s+["\']?([A-Za-z0-9-]+)["\']?', search_text, re.IGNORECASE)
        quoted_text = re.search(r'["\']([A-Za-z0-9 -]{2,})["\']', search_text)

        if plate_match:
            candidate = plate_match.group(1).strip()
            if candidate.lower() not in ("the", "a", "an", "in", "of", "on", "is"):
                target_ocr = candidate
        elif quoted_text:
            target_ocr = quoted_text.group(1).strip()

        # 4. Routing selection
        if any(w in query.lower() for w in ["color", "red", "blue", "green", "dark", "bright", "looks like"]):
            if not target_ocr:
                vector_name = "fused"

        # 5. Build Qdrant Filter object
        must_conditions: List[qmodels.Condition] = []

        if extracted_video_id:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="video_id",
                    match=qmodels.MatchValue(value=extracted_video_id),
                )
            )

        if time_range:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="timestamp_sec",
                    range=qmodels.Range(
                        gte=time_range[0],
                        lte=time_range[1],
                    ),
                )
            )

        if target_ocr:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="ocr_texts",
                    match=qmodels.MatchValue(value=target_ocr),
                )
            )

        filter_obj = qmodels.Filter(must=must_conditions) if must_conditions else None

        return AnalyzedQuery(
            original_query=query,
            search_text=search_text if search_text else query,
            vector_name=vector_name,
            filters=filter_obj,
            time_range=time_range,
            target_video_id=extracted_video_id,
            target_ocr=target_ocr,
            top_k=top_k,
        )

