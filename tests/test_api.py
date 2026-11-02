"""
Tests for FastAPI HTTP endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from video_search.main import create_app
from video_search.config import load_config


@pytest.fixture(scope="module")
def client():
    config = load_config()
    config.vector_db.storage_path = ":memory:"
    config.vector_db.use_memory = True
    app = create_app(config)
    with TestClient(app) as test_c:
        yield test_c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "video-search-engine"


def test_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AETHER VIDEO" in response.text
    assert "<video" in response.text


def test_api_search_and_ask(client):
    # Test search endpoint
    search_res = client.post(
        "/api/v1/search",
        json={"query": "traffic and pedestrian", "top_k": 5},
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "results" in search_data
    assert "total_results" in search_data

    # Test ask endpoint
    ask_res = client.post(
        "/api/v1/ask",
        json={"query": "What happens in the video?", "top_k": 5},
    )
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert "answer" in ask_data
    assert "timestamps" in ask_data

