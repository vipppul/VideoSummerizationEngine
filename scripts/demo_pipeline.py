"""
End-to-End Demo Script for Multimodal Video Search Engine.
Creates a synthetic sample video, processes it through the pipeline,
and runs semantic search and RAG question answering.
"""

from __future__ import annotations
import asyncio
import os
import cv2
import numpy as np

from video_search.config import load_config
from video_search.pipeline import PipelineOrchestrator
from video_search.query.engine import RAGQueryEngine
from video_search.utils.logging import setup_logger

logger = setup_logger("scripts.demo")


def create_sample_video(output_path: str = "./data/videos/sample_demo.mp4", duration_sec: int = 10, fps: int = 15):
    """
    Creates a synthetic 10-second test video with moving objects and text
    simulating a traffic scene with license plates and signage.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    logger.info(f"Generating synthetic video: '{output_path}' ({total_frames} frames)...")

    for i in range(total_frames):
        # Create road / background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80
        # Draw road
        cv2.rectangle(frame, (0, 200), (width, 480), (40, 40, 40), -1)
        # Road lane markers
        for lane_x in range(0, width, 60):
            cv2.line(frame, (lane_x, 340), (lane_x + 30, 340), (255, 255, 255), 2)

        # Scene 1 (0 to 4s): Car moving with license plate "ABC-1234"
        if i < fps * 4:
            car_x = int(50 + (i / (fps * 4)) * 300)
            car_y = 280
            # Draw blue car
            cv2.rectangle(frame, (car_x, car_y), (car_x + 120, car_y + 60), (200, 50, 50), -1)
            cv2.rectangle(frame, (car_x + 20, car_y - 25), (car_x + 100, car_y), (220, 80, 80), -1)
            # License plate
            cv2.rectangle(frame, (car_x + 30, car_y + 35), (car_x + 90, car_y + 55), (255, 255, 255), -1)
            cv2.putText(frame, "ABC-1234", (car_x + 32, car_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            # Road sign
            cv2.rectangle(frame, (480, 80), (580, 140), (0, 180, 0), -1)
            cv2.putText(frame, "EXIT 14B", (490, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Scene 2 (4 to 7s): Red truck and STOP sign
        elif i < fps * 7:
            # STOP Sign
            cv2.circle(frame, (520, 140), 45, (0, 0, 200), -1)
            cv2.putText(frame, "STOP", (490, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # Red truck
            truck_x = int(350 - ((i - fps * 4) / (fps * 3)) * 250)
            cv2.rectangle(frame, (truck_x, 260), (truck_x + 140, 350), (30, 30, 220), -1)
            cv2.putText(frame, "TRUCK-99", (truck_x + 15, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Scene 3 (7 to 10s): Pedestrian and bicycle
        else:
            ped_x = int(100 + ((i - fps * 7) / (fps * 3)) * 150)
            # Pedestrian
            cv2.circle(frame, (ped_x, 230), 15, (200, 180, 150), -1)
            cv2.line(frame, (ped_x, 245), (ped_x, 290), (0, 150, 0), 4)
            cv2.putText(frame, "PEDESTRIAN", (ped_x - 30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add timestamp overlay
        time_sec = i / fps
        cv2.putText(
            frame,
            f"T: {time_sec:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        out.write(frame)

    out.release()
    logger.info(f"Synthetic video successfully saved to {output_path}")
    return output_path


async def main():
    config = load_config()
    video_path = create_sample_video()

    logger.info("Initializing Pipeline Orchestrator...")
    orchestrator = PipelineOrchestrator(config)

    video_id = "test_traffic_cam_01"
    logger.info(f"Ingesting and indexing video '{video_id}'...")
    await orchestrator.process_video_async(source_uri=video_path, video_id=video_id)

    # Wait for processing to complete
    while True:
        status = orchestrator.get_job_status(video_id)
        logger.info(f"Job Status: {status['status']} | Frames Indexed: {status.get('frames_indexed', 0)}")
        if status["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(1.0)

    if status["status"] == "failed":
        logger.error(f"Processing failed: {status.get('error')}")
        return

    # Fetch and display video narrative summary
    summary = orchestrator.summary_store.get_summary(video_id)
    print("\n" + "="*70)
    print("GLOBAL VIDEO CHRONOLOGICAL SUMMARY:")
    print("="*70)
    print(summary.narrative if summary else "No summary generated.")
    print("="*70 + "\n")

    # Initialize RAG Query Engine
    logger.info("Initializing RAG Query Engine...")
    query_engine = RAGQueryEngine(
        config=config,
        vector_store=orchestrator.vector_store,
        summary_store=orchestrator.summary_store,
        encoder=orchestrator.encoder,
        vlm=orchestrator.vlm,
    )

    test_queries = [
        "When does the vehicle with license plate ABC-1234 appear?",
        "Is there a red truck or stop sign in the video?",
        "What happens between 7 and 10 seconds?",
    ]

    for q in test_queries:
        print(f"\n[QUERY]: '{q}'")
        res = query_engine.ask(q, video_id=video_id)
        print(f"[ANSWER]:\n{res.answer}")
        print(f"[CONFIDENCE]: {res.confidence:.2f}")
        print(f"[CITED TIMESTAMPS]:")
        for ts in res.timestamps:
            print(f"  - [{ts.time_formatted}] ({ts.time_sec:.2f}s): {ts.description}")
        print(f"[EVIDENCE FRAMES RETRIEVED]: {len(res.evidence_frames)}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())

