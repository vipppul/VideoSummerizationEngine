"""
Unit tests for Module 1: Ingestion and Sampling.
"""

import pytest
import numpy as np
from video_search.ingestion.sampling import (
    FixedFPSSampling,
    KeyframeSampling,
    SceneChangeSampling,
    AdaptiveSampling,
)
from video_search.ingestion.queue import BoundedFrameQueue
from video_search.models import RawFrame


def test_fixed_fps_sampling():
    gate = FixedFPSSampling(sample_interval_sec=1.0)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame at 0.0s -> should sample
    assert gate.should_sample(0, 0.0, dummy_img) is True
    # Frame at 0.5s -> should not sample (< 1.0s interval)
    assert gate.should_sample(1, 0.5, dummy_img) is False
    # Frame at 1.0s -> should sample
    assert gate.should_sample(2, 1.0, dummy_img) is True
    # Frame at 1.2s -> should not sample
    assert gate.should_sample(3, 1.2, dummy_img) is False


def test_keyframe_sampling():
    gate = KeyframeSampling()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert gate.should_sample(0, 0.0, dummy_img, is_keyframe=True) is True
    assert gate.should_sample(1, 0.1, dummy_img, is_keyframe=False) is False


@pytest.mark.asyncio
async def test_bounded_queue_overflow_drop_oldest():
    q = BoundedFrameQueue(maxsize=2, overflow_policy="drop_oldest")
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)

    f1 = RawFrame("v1", 0, 0.0, "iso", dummy, 10, 10, "fixed", 30.0, "uri")
    f2 = RawFrame("v1", 1, 1.0, "iso", dummy, 10, 10, "fixed", 30.0, "uri")
    f3 = RawFrame("v1", 2, 2.0, "iso", dummy, 10, 10, "fixed", 30.0, "uri")

    await q.put(f1)
    await q.put(f2)
    # Adding 3rd item with maxsize=2 and drop_oldest should drop f1
    await q.put(f3)

    out = await q.get()
    assert out.frame_index == 1
    assert q.frames_dropped == 1

