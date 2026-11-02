"""
Bounded async frame queue with backpressure and drop policies.
"""

from __future__ import annotations
import asyncio
from typing import Optional
from ..models import RawFrame


class BoundedFrameQueue:
    """
    Asynchronous queue holding RawFrame items with backpressure support.
    """

    def __init__(self, maxsize: int = 64, overflow_policy: str = "block"):
        self.maxsize = maxsize
        self.overflow_policy = overflow_policy.lower()  # "block" or "drop_oldest"
        self._queue: asyncio.Queue[Optional[RawFrame]] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self.frames_emitted = 0
        self.frames_dropped = 0

    async def put(self, frame: Optional[RawFrame]) -> None:
        """
        Pushes a frame to the queue. When frame is None, indicates EOF sentinel.
        """
        if frame is None:
            await self._queue.put(None)
            self._closed = True
            return

        if self._queue.full() and self.overflow_policy == "drop_oldest":
            try:
                # Remove oldest frame to make room
                _ = self._queue.get_nowait()
                self._queue.task_done()
                self.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass

        await self._queue.put(frame)
        self.frames_emitted += 1

    async def get(self) -> Optional[RawFrame]:
        """
        Retrieves next frame from queue. Returns None on EOF sentinel.
        """
        item = await self._queue.get()
        return item

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def is_closed(self) -> bool:
        return self._closed

