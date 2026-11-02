"""
CLI Tool to Ingest, Summarize, and Interactively Query Any Video.
Automatically detects if the FastAPI server is running on localhost:8000
and uses it; otherwise runs directly in standalone mode.

Usage:
    python scripts/analyze_video.py --video "path/to/your_video.mp4"
    python scripts/analyze_video.py --video "rtsp://..." --video-id "stream_01"
"""

from __future__ import annotations
import argparse
import asyncio
import os
import sys

# Windows UTF-8 console output protection
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
from video_search.config import load_config
from video_search.utils.logging import setup_logger

logger = setup_logger("cli.analyze")


def is_server_running(url: str = "http://localhost:8000/health") -> bool:
    try:
        r = httpx.get(url, timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


async def run_via_server(api_base: str, video_path: str, video_id: str, sample_interval: float):
    print(f"[SERVER] Connected to running Video Search Engine server at {api_base}")
    client = httpx.AsyncClient(base_url=api_base, timeout=120.0)

    # 1. Ingest
    print(f"[INGEST] Submitting video '{video_path}' for ingestion...")
    abs_path = os.path.abspath(video_path) if not video_path.startswith("rtsp://") else video_path
    ingest_payload = {
        "source_uri": abs_path,
        "video_id": video_id,
        "sample_strategy": "fixed_fps",
        "sample_interval_sec": sample_interval,
    }
    resp = await client.post("/api/v1/videos/ingest", json=ingest_payload)
    if resp.status_code != 200:
        print(f"\n[ERROR] Ingestion error: {resp.text}")
        return

    data = resp.json()
    vid_id = data["video_id"]

    # 2. Poll status
    while True:
        stat_resp = await client.get(f"/api/v1/videos/{vid_id}/status")
        if stat_resp.status_code == 200:
            stat = stat_resp.json()
            frames_done = stat.get("frames_indexed", 0)
            sys.stdout.write(f"\r[STATUS] {stat['status']} | Frames Indexed: {frames_done}")
            sys.stdout.flush()
            if stat["status"] in ("completed", "failed"):
                print()
                break
        await asyncio.sleep(1.0)

    # 3. Retrieve Summary
    summary_resp = await client.get(f"/api/v1/videos/{vid_id}/summary")
    if summary_resp.status_code == 200:
        summary_data = summary_resp.json()
        print("\n" + "=" * 70)
        print(f"GLOBAL VIDEO SUMMARY FOR: {vid_id}")
        print("=" * 70)
        print(summary_data.get("narrative", "No summary text generated."))
        print("=" * 70)

    # 4. Interactive Q&A
    print("\n[SEARCH] Interactive Question Answering & Semantic Search")
    print("Type your question below (or 'quit' / 'exit' to stop).\n")

    while True:
        try:
            query = input("\n[QUERY] Ask about the video: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("[EXIT] Exiting video search.")
                break

            ask_resp = await client.post("/api/v1/ask", json={"query": query, "video_id": vid_id})
            if ask_resp.status_code == 200:
                res = ask_resp.json()
                print("\n" + "-" * 60)
                print(f"[ANSWER]:\n{res.get('answer', '')}\n")
                timestamps = res.get("timestamps", [])
                if timestamps:
                    print("[TIMESTAMPS]:")
                    for ts in timestamps:
                        print(f"  * [{ts.get('time_formatted', '')}] ({ts.get('time_sec', 0.0):.2f}s) - {ts.get('description', '')}")
                print(f"[CONFIDENCE]: {res.get('confidence', 0.0) * 100:.1f}% | Evidence Frames: {res.get('evidence_frames_count', 0)}")
                print("-" * 60)
            else:
                print(f"[ERROR] Server error: {ask_resp.text}")

        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Exiting.")
            break


async def run_standalone(video_path: str, video_id: str, sample_interval: float):
    from video_search.pipeline import PipelineOrchestrator
    from video_search.query.engine import RAGQueryEngine

    config = load_config()
    config.ingestion.sample_interval_sec = sample_interval

    vid_id = video_id or os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n[STANDALONE] Running in standalone local mode for: '{video_path}' (ID: '{vid_id}')...")

    orchestrator = PipelineOrchestrator(config)

    print(f"[PROCESS] Extracting frames, detecting objects, and generating captions/OCR...")
    await orchestrator.process_video_async(source_uri=video_path, video_id=vid_id)

    while True:
        status = orchestrator.get_job_status(vid_id)
        frames_done = status.get("frames_indexed", 0)
        sys.stdout.write(f"\r[STATUS] {status['status']} | Frames Indexed: {frames_done}")
        sys.stdout.flush()
        if status["status"] in ("completed", "failed"):
            print()
            break
        await asyncio.sleep(1.0)

    if status["status"] == "failed":
        print(f"\n[ERROR] Pipeline failed: {status.get('error')}")
        return

    summary = orchestrator.summary_store.get_summary(vid_id)
    print("\n" + "=" * 70)
    print(f"GLOBAL VIDEO SUMMARY FOR: {vid_id}")
    print("=" * 70)
    if summary and summary.narrative:
        print(summary.narrative)
    else:
        print("Summary not generated.")
    print("=" * 70)

    query_engine = RAGQueryEngine(
        config=config,
        vector_store=orchestrator.vector_store,
        summary_store=orchestrator.summary_store,
        encoder=orchestrator.encoder,
        vlm=orchestrator.vlm,
    )

    print("\n[SEARCH] Interactive Question Answering & Semantic Search")
    print("Type your question below (or 'quit' / 'exit' to stop).\n")

    while True:
        try:
            query = input("\n[QUERY] Ask about the video: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("[EXIT] Exiting video search.")
                break

            response = query_engine.ask(query, video_id=vid_id)
            print("\n" + "-" * 60)
            print(f"[ANSWER]:\n{response.answer}\n")
            if response.timestamps:
                print("[TIMESTAMPS]:")
                for ts in response.timestamps:
                    print(f"  * [{ts.time_formatted}] ({ts.time_sec:.2f}s) - {ts.description}")
            print(f"[CONFIDENCE]: {response.confidence * 100:.1f}% | Evidence Frames: {len(response.evidence_frames)}")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Analyze and query your video")
    parser.add_argument("--video", "-v", required=True, help="Path to video file (.mp4, .mkv, .avi) or RTSP stream")
    parser.add_argument("--video-id", "-id", default=None, help="Custom identifier for the video")
    parser.add_argument("--sample-interval", "-s", type=float, default=1.0, help="Frame sample interval in seconds (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.video) and not args.video.startswith("rtsp://"):
        print(f"\n[ERROR] Video file '{args.video}' does not exist on disk.")
        sys.exit(1)

    vid_id = args.video_id or os.path.splitext(os.path.basename(args.video))[0]

    if is_server_running():
        asyncio.run(run_via_server("http://localhost:8000", args.video, vid_id, args.sample_interval))
    else:
        asyncio.run(run_standalone(args.video, vid_id, args.sample_interval))


if __name__ == "__main__":
    main()
