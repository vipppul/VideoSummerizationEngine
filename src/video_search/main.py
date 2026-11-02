"""
Main application entry point for Video Search Engine FastAPI service.
"""

from __future__ import annotations
import os
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import load_config, SystemConfig
from .pipeline import PipelineOrchestrator
from .query.engine import RAGQueryEngine
from .api.routes import router, set_services
from .utils.logging import setup_logger

logger = setup_logger("main")


def create_app(config: SystemConfig = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing Video Search Engine services...")
        orch = PipelineOrchestrator(cfg)
        q_engine = RAGQueryEngine(
            config=cfg,
            vector_store=orch.vector_store,
            summary_store=orch.summary_store,
            encoder=orch.encoder,
            vlm=orch.vlm,
        )
        set_services(orch, q_engine)
        logger.info("System ready to accept requests!")
        yield
        logger.info("Shutting down Video Search Engine services...")

    app = FastAPI(
        title="Multimodal Video Summarization and Semantic Search API",
        version="0.1.0",
        description="End-to-End Multimodal Video Pipeline: Ingestion, YOLO, Florence-2 VLM/OCR, CLIP/SigLIP, Qdrant, RAG Engine",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


def run():
    parser = argparse.ArgumentParser(description="Run Video Search Engine API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config
    uvicorn.run("video_search.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    run()

