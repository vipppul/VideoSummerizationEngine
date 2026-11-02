# Multimodal Video Summarization and Semantic Search Engine

An end-to-end, query-agnostic multimodal video analysis, summarization, and semantic search engine built for edge computer vision and scalable retrieval.

---

## 🌟 Key Features

- **Module 1 — Video Ingestion Pipeline**: Low-latency video ingestion (files & RTSP streams) via FFmpeg and OpenCV. Supports 4 sampling strategies (`fixed_fps`, `keyframe`, `scene_change`, `adaptive`) and maintains exact PTS timestamps and frame indices.
- **Module 2 — Dense Scene Understanding & OCR**: Object detection with YOLOv8, dense region captioning, full-frame descriptions, and OCR text extraction with Florence-2 (with graceful fallback). Generates chronological video narratives.
- **Module 3 — Multimodal Vectorization**: Shared visual-text embedding space via `open_clip` (CLIP / SigLIP) with configurable fusion strategies (`weighted_average`, `image_only`, `text_only`, `concatenate`).
- **Module 4 — Vector Database & Schema**: Qdrant vector database with triple named vectors (`fused`, `image`, `text`), INT8 scalar quantization support, and high-speed payload indexing. Includes companion SQLite narrative summary store.
- **Module 5 — RAG Query Engine**: Arbitrary natural-language queries, automated filter parsing (temporal range, video ID, OCR text, entity types), hybrid search with Reciprocal Rank Fusion, cross-encoder re-ranking, and grounded LLM synthesis citing exact timestamps.
- **Module 6 — Edge & Hardware Optimization**: TensorRT and ONNX export scripts, CUDA accelerated decoding, and dynamic batching.
- **FastAPI REST API**: Production-grade async HTTP service exposing ingestion, retrieval, frame serving, and RAG endpoints.

---

## 🚀 Quick Start

### 1. Environment & Setup

Ensure you are using the `anpr` Miniconda environment:

```bash
conda activate anpr
```

Install local package in editable mode:

```bash
pip install -e .
```

Copy `.env.example` to `.env` and configure your API keys (e.g. `GOOGLE_API_KEY` for Gemini):

```bash
copy .env.example .env
```

### 2. Run the End-to-End Demo

Run the self-contained demo script which automatically generates a synthetic traffic video, indexes it through all 5 modules, and runs semantic search queries:

```bash
python scripts/demo_pipeline.py
```

### 3. Start the FastAPI Web Service

```bash
python -m video_search.main --host 0.0.0.0 --port 8000 --config config/default.yaml
```

Interactive OpenAPI Swagger UI is available at:  
👉 **`http://localhost:8000/docs`**

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/videos/ingest` | Ingest video file or RTSP stream |
| `GET` | `/api/v1/videos/{video_id}/status` | Check video ingestion & indexing status |
| `GET` | `/api/v1/videos/{video_id}/summary` | Retrieve global chronological narrative |
| `POST` | `/api/v1/search` | Search video frames via vector similarity + filters |
| `POST` | `/api/v1/ask` | Natural language question answering (RAG) |
| `GET` | `/api/v1/frames/{video_id}/{frame_index}/image` | Fetch extracted frame JPEG image |
| `DELETE` | `/api/v1/videos/{video_id}` | Delete video and all indexed assets |
| `GET` | `/health` | Health check endpoint |

---

## 🧪 Running Tests

Run the test suite using pytest:

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration

Configurations can be customized via YAML files in `config/`:

- `config/default.yaml`: Base configuration for server / high-accuracy mode.
- `config/edge.yaml`: Overrides for resource-constrained edge devices (reduced batch size, smaller input resolutions).
- `config/dev.yaml`: Overrides for local rapid development (in-memory Qdrant, higher sampling intervals).

Override settings at runtime via environment variables or CLI flags:
```bash
python -m video_search.main --config config/edge.yaml
```

