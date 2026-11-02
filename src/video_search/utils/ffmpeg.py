"""
FFmpeg and video stream utilities.
"""

from __future__ import annotations
import json
import subprocess
from typing import Dict, Any, Optional
import cv2


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts video metadata (dimensions, fps, duration, total frames)
    using ffprobe, with fallback to OpenCV VideoCapture.
    """
    metadata: Dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration_sec": 0.0,
        "total_frames": 0,
    }

    # Try ffprobe first
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if video_stream:
            metadata["width"] = int(video_stream.get("width", 1920))
            metadata["height"] = int(video_stream.get("height", 1080))
            # Calculate FPS
            r_fps = video_stream.get("r_frame_rate", "30/1")
            if "/" in r_fps:
                num, den = r_fps.split("/")
                fps_val = float(num) / max(float(den), 1.0)
            else:
                fps_val = float(r_fps)
            metadata["fps"] = fps_val if fps_val > 0 else 30.0

            if "duration" in video_stream:
                metadata["duration_sec"] = float(video_stream["duration"])
            elif "duration" in info.get("format", {}):
                metadata["duration_sec"] = float(info["format"]["duration"])

            if "nb_frames" in video_stream:
                metadata["total_frames"] = int(video_stream["nb_frames"])
            elif metadata["duration_sec"] > 0:
                metadata["total_frames"] = int(metadata["duration_sec"] * metadata["fps"])
            return metadata
    except Exception:
        pass

    # Fallback to OpenCV
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        metadata["fps"] = float(fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        metadata["total_frames"] = total_frames
        if fps > 0 and total_frames > 0:
            metadata["duration_sec"] = total_frames / fps
        cap.release()

    return metadata


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

