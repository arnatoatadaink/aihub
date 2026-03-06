"""E2E API tests for FastAPI endpoints (using TestClient)."""
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _install_fake_genai():
    """Insert stub google.generativeai into sys.modules before any import triggers it."""
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    genai_mod = MagicMock(name="google.generativeai")
    genai_mod.types = MagicMock()
    genai_mod.types.GenerationConfig = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("google", google_pkg)
    sys.modules["google.generativeai"] = genai_mod
    return genai_mod


# Install before any backend import so providers/__init__.py won't hit the real package
_GENAI_MOCK = _install_fake_genai()


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client with mocked providers."""
    with patch.dict("os.environ", {
        "GEMINI_API_KEY": "fake-key",
        "OPENAI_API_KEY": "sk-fake",
        "ANTHROPIC_API_KEY": "sk-ant-fake",
    }):
        from backend.main import app
        yield TestClient(app)


# ======================================================================
# Health
# ======================================================================

class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ======================================================================
# Models
# ======================================================================

class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert any("gemini" in m for m in model_ids)
        assert any("gpt" in m for m in model_ids)


# ======================================================================
# Chat completions (non-streaming)
# ======================================================================

class TestChatCompletions:
    def test_chat_completion_gemini(self, client):
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        _GENAI_MOCK.GenerativeModel.return_value = mock_model

        resp = client.post("/v1/chat/completions", json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "gemini",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["object"] == "chat.completion"

    def test_chat_completion_unknown_provider(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "nonexistent",
            "stream": False,
        })
        assert resp.status_code == 400

    def test_chat_completion_streaming(self, client):
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = iter([chunk1, chunk2])
        _GENAI_MOCK.GenerativeModel.return_value = mock_model

        resp = client.post("/v1/chat/completions", json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "gemini",
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data:" in body
        assert "[DONE]" in body


# ======================================================================
# Templates
# ======================================================================

class TestTemplates:
    def test_list_templates(self, client):
        resp = client.get("/v1/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        names = [t.get("name") for t in data["templates"]]
        assert "General Assistant" in names

    def test_get_template(self, client):
        resp = client.get("/v1/templates/general_assistant")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "General Assistant"
        assert "system_prompt" in data

    def test_get_template_not_found(self, client):
        resp = client.get("/v1/templates/nonexistent_template")
        assert resp.status_code == 404

    def test_save_and_delete_template(self, client):
        resp = client.post("/v1/templates/test_temp", json={
            "name": "Test Template",
            "description": "For testing",
            "system_prompt": "You are a tester.",
            "tags": ["test"],
        })
        assert resp.status_code == 200

        resp = client.get("/v1/templates/test_temp")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test Template"

        resp = client.delete("/v1/templates/test_temp")
        assert resp.status_code == 200

        resp = client.get("/v1/templates/test_temp")
        assert resp.status_code == 404


# ======================================================================
# Media endpoints
# ======================================================================

class TestMedia:
    def test_image_generate_unknown_provider(self, client):
        resp = client.post("/v1/media/images/generate", json={
            "prompt": "a cat",
            "provider": "nonexistent",
        })
        assert resp.status_code == 400

    def test_video_generate_unknown_provider(self, client):
        resp = client.post("/v1/media/videos/generate", json={
            "prompt": "a sunset",
            "provider": "nonexistent",
        })
        assert resp.status_code == 400

    def test_music_generate_unknown_provider(self, client):
        resp = client.post("/v1/media/music/generate", json={
            "prompt": "jazz music",
            "provider": "nonexistent",
        })
        assert resp.status_code == 400


# ======================================================================
# Training jobs (without Celery — expect 503)
# ======================================================================

class TestTrainingJobs:
    def test_create_job_without_celery(self, client):
        resp = client.post("/v1/training/jobs", json={
            "model_name_or_path": "google/gemma-2-2b-it",
        })
        assert resp.status_code == 503
