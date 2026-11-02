"""
FastAPI route definitions for Video Search and Retrieval Engine.
"""

from __future__ import annotations
import os
import re
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response, status, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse

from .schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchItem,
    AskRequest,
    AskResponse,
    TimestampItem,
    VideoStatusResponse,
    VideoSummaryResponse,
)
from ..models import IngestConfig
from ..pipeline import PipelineOrchestrator
from ..query.engine import RAGQueryEngine
from ..utils.ffmpeg import format_timestamp
from ..utils.logging import setup_logger

logger = setup_logger("api.routes")
router = APIRouter()

# Globals injected during application startup
orchestrator: Optional[PipelineOrchestrator] = None
query_engine: Optional[RAGQueryEngine] = None


def set_services(orch: PipelineOrchestrator, q_engine: RAGQueryEngine) -> None:
    global orchestrator, query_engine
    orchestrator = orch
    query_engine = q_engine


@router.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "video-search-engine"}


@router.get("/api/v1/videos", tags=["Videos"])
def list_videos() -> List[Dict[str, Any]]:
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")
    return orchestrator.summary_store.list_all()


@router.post("/api/v1/videos/upload", tags=["Ingestion"])
async def upload_and_ingest(
    file: UploadFile = File(...),
    video_id: Optional[str] = Form(None),
    sample_interval_sec: float = Form(1.0),
):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")

    upload_dir = Path("./data/videos")
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "uploaded_video.mp4"
    save_path = upload_dir / filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vid_id = video_id or Path(filename).stem
    config = IngestConfig(
        sample_strategy="fixed_fps",
        sample_interval_sec=sample_interval_sec,
    )
    assigned_id = await orchestrator.process_video_async(
        source_uri=str(save_path.resolve()),
        video_id=vid_id,
        ingest_config=config,
    )
    return {
        "video_id": assigned_id,
        "status": "processing",
        "source_uri": str(save_path),
        "message": f"Uploaded and started processing video '{assigned_id}'",
    }


@router.get("/api/v1/videos/{video_id}/stream", tags=["Videos"])
def stream_video(video_id: str):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")

    summary = orchestrator.summary_store.get_summary(video_id)
    if summary and summary.source_uri and os.path.exists(summary.source_uri):
        return FileResponse(summary.source_uri, media_type="video/mp4")

    # Fallback: check data/videos folder for matching files
    candidates = list(Path("./data/videos").glob(f"{video_id}.*")) + list(Path("./data/videos").glob(f"*{video_id}*"))
    for c in candidates:
        if c.suffix.lower() in (".mp4", ".mkv", ".avi", ".webm", ".mov") and c.exists():
            return FileResponse(str(c), media_type="video/mp4")

    raise HTTPException(status_code=404, detail=f"Video stream for '{video_id}' not found")


@router.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
@router.get("/", response_class=HTMLResponse, tags=["Web UI"])
def get_ui_page():
    ui_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if ui_path.exists():
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
            )
    return HTMLResponse(content="<h1>UI Template Not Found</h1>", status_code=404)


@router.post("/api/v1/videos/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_video(req: IngestRequest) -> IngestResponse:
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")

    if not os.path.exists(req.source_uri) and not req.source_uri.startswith("rtsp://"):
        raise HTTPException(
            status_code=400,
            detail=f"Video source URI '{req.source_uri}' does not exist on disk and is not an RTSP stream.",
        )

    config = IngestConfig(
        sample_strategy=req.sample_strategy or "fixed_fps",
        sample_interval_sec=req.sample_interval_sec or 1.0,
    )
    video_id = await orchestrator.process_video_async(
        source_uri=req.source_uri,
        video_id=req.video_id,
        ingest_config=config,
    )
    return IngestResponse(
        video_id=video_id,
        status="processing",
        message=f"Ingestion started in background for video '{video_id}'",
    )


@router.get("/api/v1/videos/{video_id}/status", response_model=VideoStatusResponse, tags=["Ingestion"])
def get_video_status(video_id: str) -> VideoStatusResponse:
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")
    stat = orchestrator.get_job_status(video_id)
    if stat["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return VideoStatusResponse(
        video_id=stat["video_id"],
        status=stat["status"],
        frames_indexed=stat.get("frames_indexed", 0),
        summary_available=stat.get("summary_available", False),
    )


@router.get("/api/v1/videos/{video_id}/summary", response_model=VideoSummaryResponse, tags=["Summaries"])
def get_video_summary(video_id: str) -> VideoSummaryResponse:
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")
    summary = orchestrator.summary_store.get_summary(video_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"No summary found for video '{video_id}'")

    # Collect unique cleaned entities across all segments
    all_entities = set()
    for s in summary.segments:
        for ent in s.key_entities:
            if not ent:
                continue
            cleaned = ent
            if cleaned.startswith("Text:'"):
                cleaned = cleaned[6:].rstrip("'")
            if cleaned.startswith("</s>"):
                cleaned = cleaned[4:].strip()
            # Skip pure date-time timestamps
            if re.match(r"^\d{2}-\d{2}-\d{4}", cleaned):
                continue
            if len(cleaned) > 1:
                all_entities.add(cleaned)

    # Produce executive short summary
    short_summary = ""
    if query_engine:
        short_summary = query_engine.get_short_summary(summary)

    if not short_summary:
        if summary.segments:
            sentences = []
            for seg in summary.segments:
                txt = seg.summary_text
                txt = re.sub(r"^The image (shows|is a screenshot from)\s*", "", txt, flags=re.IGNORECASE)
                first_s = txt.split(".")[0].strip()
                if first_s and first_s not in sentences:
                    sentences.append(first_s[:1].upper() + first_s[1:])
            short_summary = ". ".join(sentences[:2]) + "."
        else:
            short_summary = summary.narrative[:220]

    return VideoSummaryResponse(
        video_id=summary.video_id,
        source_uri=summary.source_uri,
        duration_sec=summary.total_duration_sec,
        total_frames=summary.total_frames_processed,
        narrative=summary.narrative,
        segments=[s.__dict__ for s in summary.segments],
        short_summary=short_summary,
        key_entities=sorted(list(all_entities))[:15],
    )


@router.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
def search_videos(req: SearchRequest) -> SearchResponse:
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")

    analyzed = query_engine.analyzer.analyze(
        query=req.query,
        video_id=req.video_id,
        top_k=req.top_k or 20,
    )
    results = query_engine.searcher.search(analyzed)
    reranked = query_engine.reranker.rerank(req.query, results, top_n=req.top_k or 20)

    items: List[SearchItem] = []
    for r in reranked:
        p = r.payload
        ts = float(p.get("timestamp_sec", 0.0))
        items.append(
            SearchItem(
                point_id=r.point_id,
                score=r.score,
                video_id=p.get("video_id", ""),
                frame_index=p.get("frame_index", 0),
                timestamp_sec=ts,
                timestamp_formatted=format_timestamp(ts),
                frame_caption=p.get("frame_caption", ""),
                detection_summary=p.get("detection_summary", ""),
                ocr_full=p.get("ocr_full", ""),
                frame_image_path=p.get("frame_image_path"),
            )
        )

    return SearchResponse(
        query=req.query,
        total_results=len(items),
        results=items,
    )


@router.post("/api/v1/ask", response_model=AskResponse, tags=["RAG Engine"])
def ask_question(req: AskRequest) -> AskResponse:
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")

    ans = query_engine.ask(
        query=req.query,
        video_id=req.video_id,
        top_k=req.top_k,
    )

    timestamps = [
        TimestampItem(
            time_sec=t.time_sec,
            time_formatted=t.time_formatted,
            description=t.description,
            frame_index=t.frame_index,
        )
        for t in ans.timestamps
    ]

    return AskResponse(
        query=ans.query,
        answer=ans.answer,
        timestamps=timestamps,
        confidence=ans.confidence,
        follow_up_suggestions=ans.follow_up_suggestions,
        evidence_frames_count=len(ans.evidence_frames),
        search_metadata=ans.search_metadata,
    )


@router.get("/api/v1/frames/{video_id}/{frame_index}/image", tags=["Frames"])
def get_frame_image(video_id: str, frame_index: int):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")
    path = orchestrator.frame_store.get_frame_path(video_id, frame_index)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image for frame {frame_index} not found")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/api/v1/videos/{video_id}", tags=["Videos"])
def delete_video(video_id: str):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized")
    orchestrator.delete_video(video_id)
    return {"status": "success", "message": f"Deleted video '{video_id}' and all associated assets"}

