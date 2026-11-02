"""
Context Assembler for building LLM synthesis prompts.
"""

from __future__ import annotations
from typing import List, Optional
from ..models import SearchResult, VideoSummary
from ..utils.ffmpeg import format_timestamp


class ContextAssembler:
    """
    Assembles video summary and retrieved frame metadata into a structured prompt.
    """

    def assemble_prompt(
        self,
        query: str,
        results: List[SearchResult],
        summary: Optional[VideoSummary] = None,
    ) -> str:
        summary_text = summary.narrative if summary else "No global video summary available."

        frames_context = []
        for i, res in enumerate(results, 1):
            p = res.payload
            ts = float(p.get("timestamp_sec", 0.0))
            ts_str = format_timestamp(ts)
            cap = p.get("frame_caption", "No description")
            det = p.get("detection_summary", "None")
            ocr = p.get("ocr_full", "None")
            score = res.score
            vid_id = p.get("video_id", "unknown")
            frame_idx = p.get("frame_index", 0)

            frame_entry = (
                f"[Evidence Frame {i}]\n"
                f"- Video ID: {vid_id} | Frame Index: {frame_idx}\n"
                f"- Timestamp: {ts:.2f}s ({ts_str}) | Relevance Score: {score:.3f}\n"
                f"- Scene Caption: {cap}\n"
                f"- Objects Detected: {det}\n"
                f"- Text Visible (OCR): {ocr}\n"
            )
            frames_context.append(frame_entry)

        frames_str = "\n".join(frames_context) if frames_context else "No matching frames found."

        prompt = f"""You are an Expert Multimodal Video Analysis & Retrieval Assistant.
You answer user questions about video content using the provided frame-level evidence and video summary.

=== GLOBAL VIDEO SUMMARY ===
{summary_text}

=== RETRIEVED FRAME EVIDENCE (Ordered by Relevance) ===
{frames_str}

=== USER QUERY ===
{query}

=== INSTRUCTIONS ===
1. Answer the query thoroughly and accurately based ONLY on the provided frame evidence and summary.
2. For EVERY visual claim, event, or object mentioned, cite the exact timestamp (e.g. [00:00:47.430] or 47.43s).
3. If multiple frames capture an event, cite the start and end timestamps.
4. If the evidence does not contain sufficient details to answer, state clearly what is known and what cannot be confirmed.
5. Return your answer strictly in the following JSON format:

{{
  "answer": "Detailed natural language answer citing timestamps inline...",
  "timestamps": [
    {{
      "time_sec": 47.43,
      "time_formatted": "00:00:47.430",
      "description": "Short description of what happens at this timestamp",
      "frame_index": 1423
    }}
  ],
  "confidence": 0.95,
  "follow_up_suggestions": [
    "Suggested follow up question 1",
    "Suggested follow up question 2"
  ]
}}
"""
        return prompt

