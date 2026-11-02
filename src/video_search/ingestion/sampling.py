"""
Frame sampling strategies for Module 1.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import cv2


class SamplingGate(ABC):
    """Abstract sampling gate determining whether a candidate frame should be processed."""

    @abstractmethod
    def should_sample(
        self,
        frame_index: int,
        timestamp_sec: float,
        frame_image: np.ndarray,
        is_keyframe: bool = False,
    ) -> bool:
        """Returns True if the frame should be emitted downstream."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal state for a new video stream."""
        pass


class FixedFPSSampling(SamplingGate):
    """
    Samples frames at a fixed temporal interval (e.g. 1 frame every sample_interval_sec).
    """

    def __init__(self, sample_interval_sec: float = 1.0):
        self.sample_interval_sec = max(sample_interval_sec, 0.01)
        self.last_sampled_time: Optional[float] = None

    def should_sample(
        self,
        frame_index: int,
        timestamp_sec: float,
        frame_image: np.ndarray,
        is_keyframe: bool = False,
    ) -> bool:
        if self.last_sampled_time is None:
            self.last_sampled_time = timestamp_sec
            return True

        if (timestamp_sec - self.last_sampled_time) >= self.sample_interval_sec:
            self.last_sampled_time = timestamp_sec
            return True
        return False

    def reset(self) -> None:
        self.last_sampled_time = None


class KeyframeSampling(SamplingGate):
    """Samples only keyframes (I-frames)."""

    def should_sample(
        self,
        frame_index: int,
        timestamp_sec: float,
        frame_image: np.ndarray,
        is_keyframe: bool = False,
    ) -> bool:
        return is_keyframe

    def reset(self) -> None:
        pass


class SceneChangeSampling(SamplingGate):
    """
    Samples frames when the difference between consecutive frame color histograms
    exceeds a threshold.
    """

    def __init__(self, scene_threshold: float = 0.3):
        self.scene_threshold = scene_threshold
        self.prev_hist: Optional[np.ndarray] = None

    def _calc_hist(self, frame_image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def should_sample(
        self,
        frame_index: int,
        timestamp_sec: float,
        frame_image: np.ndarray,
        is_keyframe: bool = False,
    ) -> bool:
        if self.prev_hist is None:
            self.prev_hist = self._calc_hist(frame_image)
            return True

        curr_hist = self._calc_hist(frame_image)
        # Correlation comparison: 1.0 is identical, < 1-threshold indicates scene change
        score = cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        delta = 1.0 - score
        if delta >= self.scene_threshold:
            self.prev_hist = curr_hist
            return True
        return False

    def reset(self) -> None:
        self.prev_hist = None


class AdaptiveSampling(SamplingGate):
    """
    Combines scene change detection with a minimum interval floor and maximum interval ceiling.
    """

    def __init__(
        self,
        scene_threshold: float = 0.3,
        min_interval_sec: float = 0.5,
        max_interval_sec: float = 3.0,
    ):
        self.scene_gate = SceneChangeSampling(scene_threshold=scene_threshold)
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec
        self.last_sampled_time: Optional[float] = None

    def should_sample(
        self,
        frame_index: int,
        timestamp_sec: float,
        frame_image: np.ndarray,
        is_keyframe: bool = False,
    ) -> bool:
        if self.last_sampled_time is None:
            self.last_sampled_time = timestamp_sec
            self.scene_gate.should_sample(frame_index, timestamp_sec, frame_image, is_keyframe)
            return True

        elapsed = timestamp_sec - self.last_sampled_time
        if elapsed < self.min_interval_sec:
            return False

        # If max interval reached, force sample
        if elapsed >= self.max_interval_sec:
            self.last_sampled_time = timestamp_sec
            self.scene_gate.should_sample(frame_index, timestamp_sec, frame_image, is_keyframe)
            return True

        # Check scene change
        if self.scene_gate.should_sample(frame_index, timestamp_sec, frame_image, is_keyframe):
            self.last_sampled_time = timestamp_sec
            return True

        return False

    def reset(self) -> None:
        self.last_sampled_time = None
        self.scene_gate.reset()


def create_sampling_gate(
    strategy: str = "adaptive",
    interval_sec: float = 1.0,
    scene_threshold: float = 0.25,
    min_interval_sec: float = 0.5,
    max_interval_sec: float = 3.0,
) -> SamplingGate:
    strat = strategy.lower()
    if strat == "fixed_fps":
        return FixedFPSSampling(sample_interval_sec=interval_sec)
    elif strat == "keyframe":
        return KeyframeSampling()
    elif strat == "scene_change":
        return SceneChangeSampling(scene_threshold=scene_threshold)
    elif strat == "adaptive":
        return AdaptiveSampling(
            scene_threshold=scene_threshold,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
        )
    return AdaptiveSampling(
        scene_threshold=scene_threshold,
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
    )

