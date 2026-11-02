"""
YOLO Object Detection sub-module.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from ultralytics import YOLO

from ..models import Detection, RawFrame
from ..config import DetectionSettings
from ..utils.logging import setup_logger
from ..utils.gpu import get_optimal_device

logger = setup_logger("analysis.detector")


class ObjectDetector:
    """
    Batched object detector using Ultralytics YOLO models.
    """

    def __init__(self, settings: Optional[DetectionSettings] = None):
        self.settings = settings or DetectionSettings()
        self.device = get_optimal_device(self.settings.device)
        logger.info(f"Loading YOLO detector '{self.settings.model}' on {self.device}...")
        self.model = YOLO(self.settings.model)
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move YOLO model to {self.device}, running default: {e}")

    def detect_batch(self, frames: List[RawFrame]) -> List[List[Detection]]:
        """
        Runs object detection on a batch of RawFrames and returns
        a list of Detection lists corresponding to each frame.
        """
        if not frames:
            return []

        images = [f.image for f in frames]
        try:
            results = self.model.predict(
                source=images,
                conf=self.settings.confidence_threshold,
                iou=self.settings.iou_threshold,
                imgsz=self.settings.input_size,
                classes=self.settings.target_classes,
                device=str(self.device),
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO detection error: {e}", exc_info=True)
            return [[] for _ in frames]

        batch_detections: List[List[Detection]] = []
        for idx, res in enumerate(results):
            frame_dets: List[Detection] = []
            orig_h, orig_w = frames[idx].height, frames[idx].width

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                names = res.names

                for box, conf, cls_id in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = [float(b) for b in box]
                    norm_box = [
                        max(0.0, min(1.0, x1 / orig_w)),
                        max(0.0, min(1.0, y1 / orig_h)),
                        max(0.0, min(1.0, x2 / orig_w)),
                        max(0.0, min(1.0, y2 / orig_h)),
                    ]
                    class_name = names.get(cls_id, str(cls_id))
                    frame_dets.append(
                        Detection(
                            class_name=class_name,
                            class_id=int(cls_id),
                            confidence=float(conf),
                            bbox_xyxy=[x1, y1, x2, y2],
                            bbox_normalized=norm_box,
                        )
                    )
            batch_detections.append(frame_dets)

        return batch_detections

