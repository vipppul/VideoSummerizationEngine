"""
Frame extraction module decoding video files or RTSP streams.
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
import cv2

from ..models import RawFrame, IngestConfig
from ..utils.ffmpeg import get_video_metadata
from ..utils.logging import setup_logger
from .sampling import create_sampling_gate
from .queue import BoundedFrameQueue

logger = setup_logger("ingestion.extractor")


class FrameExtractor:
    """
    Decodes video files or live RTSP streams and yields sampled RawFrames with
    exact PTS/timestamps and frame indices.
    """

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()

    async def stream_frames(
        self,
        source_uri: str,
        video_id: str,
        queue: BoundedFrameQueue,
    ) -> None:
        """
        Background worker that reads from source_uri, filters via sampling gate,
        and pushes RawFrame objects into the queue. Sends None when completed.
        """
        gate = create_sampling_gate(
            strategy=self.config.sample_strategy,
            interval_sec=self.config.sample_interval_sec,
            scene_threshold=self.config.scene_threshold,
            min_interval_sec=self.config.min_interval_sec,
            max_interval_sec=getattr(self.config, "max_interval_sec", 3.0),
        )

        metadata = get_video_metadata(source_uri)
        fps = metadata.get("fps", 30.0)
        source_width = metadata.get("width", 1920)
        source_height = metadata.get("height", 1080)

        logger.info(
            f"Starting ingestion for video '{video_id}' from '{source_uri}' | "
            f"FPS: {fps:.2f}, Res: {source_width}x{source_height}, Strategy: {self.config.sample_strategy}"
        )

        cap = cv2.VideoCapture(source_uri)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source_uri}")
            await queue.put(None)
            return

        frame_idx = 0
        now_epoch = datetime.now(timezone.utc).timestamp()

        try:
            while True:
                # Run blocking OpenCV read in thread pool to keep async loop non-blocking
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret or frame is None:
                    break

                # Get exact PTS timestamp in seconds
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec and pos_msec > 0:
                    timestamp_sec = float(pos_msec) / 1000.0
                else:
                    timestamp_sec = frame_idx / max(fps, 1.0)

                # Generate ISO timestamp
                iso_time = datetime.fromtimestamp(
                    now_epoch + timestamp_sec, tz=timezone.utc
                ).isoformat()

                if gate.should_sample(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    frame_image=frame,
                    is_keyframe=(frame_idx % int(max(fps, 1.0) * 2) == 0),
                ):
                    raw_frame = RawFrame(
                        video_id=video_id,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        timestamp_iso=iso_time,
                        image=frame,
                        width=frame.shape[1],
                        height=frame.shape[0],
                        sampling_strategy=self.config.sample_strategy,
                        source_fps=fps,
                        source_uri=source_uri,
                    )
                    await queue.put(raw_frame)

                frame_idx += 1
                # Yield control briefly for high frame rates
                if frame_idx % 30 == 0:
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error during video ingestion for {video_id}: {e}", exc_info=True)
        finally:
            cap.release()
            await queue.put(None)  # EOF Sentinel
            logger.info(
                f"Completed ingestion for '{video_id}'. Total frames read: {frame_idx}, "
                f"Emitted: {queue.frames_emitted}, Dropped: {queue.frames_dropped}"
            )

    async def extract_frames(
        self,
        source_uri: str,
        video_id: str,
    ) -> AsyncIterator[RawFrame]:
        """
        Async generator yielding sampled RawFrames directly.
        """
        queue = BoundedFrameQueue(
            maxsize=self.config.queue_max_size,
            overflow_policy=self.config.queue_overflow_policy,
        )
        task = asyncio.create_task(self.stream_frames(source_uri, video_id, queue))

        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield frame
            queue.task_done()

        await task

