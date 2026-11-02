"""
ONNX export script for cross-platform model deployment.
"""

from __future__ import annotations
import argparse
from ultralytics import YOLO
from video_search.utils.logging import setup_logger

logger = setup_logger("scripts.export_onnx")


def export_yolo_onnx(model_name: str = "yolov8n.pt", imgsz: int = 640):
    logger.info(f"Exporting {model_name} to ONNX (imgsz={imgsz})...")
    model = YOLO(model_name)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        simplify=True,
    )
    logger.info(f"Successfully generated ONNX model at: {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model", default="yolov8n.pt", help="Model checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    export_yolo_onnx(args.model, args.imgsz)


if __name__ == "__main__":
    main()

