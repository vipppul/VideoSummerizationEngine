"""
TensorRT export script for Edge Hardware Acceleration.
Exports YOLOv8 detection and CLIP encoders to TensorRT engines.
"""

from __future__ import annotations
import argparse
from ultralytics import YOLO
from video_search.utils.logging import setup_logger

logger = setup_logger("scripts.export_tensorrt")


def export_yolo_tensorrt(model_name: str = "yolov8n.pt", imgsz: int = 640, half: bool = True):
    logger.info(f"Exporting {model_name} to TensorRT engine (half={half}, imgsz={imgsz})...")
    model = YOLO(model_name)
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=0,
    )
    logger.info(f"Successfully generated TensorRT engine at: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export models to TensorRT")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    args = parser.parse_args()

    export_yolo_tensorrt(args.model, args.imgsz, args.fp16)


if __name__ == "__main__":
    main()

