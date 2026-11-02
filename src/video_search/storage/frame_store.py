"""
Filesystem storage for extracted video frame images.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from ..utils.logging import setup_logger

logger = setup_logger("storage.frame_store")


class DiskFrameStore:
    """
    Saves and serves frame images from local disk.
    """

    def __init__(self, base_dir: str = "./data/frames"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(
        self,
        video_id: str,
        frame_index: int,
        timestamp_sec: float,
        image: np.ndarray,
    ) -> str:
        """
        Saves a frame to disk and returns its relative or absolute path.
        """
        video_dir = self.base_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{frame_index:06d}_{timestamp_sec:08.2f}s.jpg"
        file_path = video_dir / filename
        cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(file_path)

    def get_frame_path(self, video_id: str, frame_index: int) -> Optional[str]:
        video_dir = self.base_dir / video_id
        if not video_dir.exists():
            return None
        matches = list(video_dir.glob(f"frame_{frame_index:06d}_*.jpg"))
        return str(matches[0]) if matches else None

    def delete_video_frames(self, video_id: str) -> None:
        video_dir = self.base_dir / video_id
        if video_dir.exists():
            shutil.rmtree(video_dir, ignore_errors=True)
            logger.info(f"Deleted frames directory for video '{video_id}'")

