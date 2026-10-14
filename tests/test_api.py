"""
API integration tests.

Tests the FastAPI app end-to-end using TestClient (no need to start
a real server). These tests verify the HTTP layer, middleware, and
error handling.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Set DEV_MODE=true so auth is bypassed in tests
    import os
    os.environ["DEV_MODE"] = "true"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only-32chars"

    from api.main import app

    with TestClient(app) as client:
        yield client


def test_root_endpoint(client):
    """Test the root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "docs" in data
    assert "health" in data


def test_health_endpoint(client):
    """Test the liveness health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data


def test_chat_demo_mode(client):
    """Test chat endpoint returns demo response when no model is loaded."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_streaming_demo(client):
    """Test streaming chat endpoint returns SSE format."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    # SSE responses have content-type text/event-stream
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_list_models(client):
    """Test the models listing endpoint."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_invalid_model_name(client):
    """Test chat with non-existent model returns demo response."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    # Falls back to demo mode, not an error
    assert response.status_code == 200


def test_chat_validation_error(client):
    """Test that invalid request body returns 422 validation error."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test",
            # Missing required 'messages' field
        },
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_chat_invalid_temperature(client):
    """Test that temperature > 2.0 is rejected."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 5.0,  # Out of range (max 2.0)
        },
    )
    assert response.status_code == 422


def test_chat_invalid_role(client):
    """Test that invalid message role is rejected."""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "invalid_role", "content": "hi"}],
        },
    )
    assert response.status_code == 422
