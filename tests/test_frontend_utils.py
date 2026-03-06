"""Tests for frontend/utils.py — shared HTTP helpers and provider config."""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestProviderModelMap:
    def test_all_text_providers_present(self):
        from frontend.utils import PROVIDER_MODEL_MAP
        assert "gemini" in PROVIDER_MODEL_MAP
        assert "openai" in PROVIDER_MODEL_MAP
        assert "anthropic" in PROVIDER_MODEL_MAP

    def test_each_provider_has_at_least_one_model(self):
        from frontend.utils import PROVIDER_MODEL_MAP
        for provider, models in PROVIDER_MODEL_MAP.items():
            assert len(models) >= 1, f"{provider} has no models"

    def test_gemini_includes_flash(self):
        from frontend.utils import PROVIDER_MODEL_MAP
        assert any("flash" in m for m in PROVIDER_MODEL_MAP["gemini"])

    def test_openai_includes_gpt4o(self):
        from frontend.utils import PROVIDER_MODEL_MAP
        assert "gpt-4o" in PROVIDER_MODEL_MAP["openai"]

    def test_anthropic_includes_sonnet(self):
        from frontend.utils import PROVIDER_MODEL_MAP
        assert any("sonnet" in m for m in PROVIDER_MODEL_MAP["anthropic"])


class TestApiGet:
    def test_returns_json_on_success(self):
        from frontend.utils import api_get
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status.return_value = None

        with patch("frontend.utils.httpx.get", return_value=mock_resp) as mock_get:
            result = api_get("/health")

        mock_get.assert_called_once()
        assert result == {"status": "ok"}

    def test_returns_error_dict_on_exception(self):
        from frontend.utils import api_get
        with patch("frontend.utils.httpx.get", side_effect=Exception("connection refused")):
            result = api_get("/health")
        assert "error" in result
        assert "connection refused" in result["error"]

    def test_uses_backend_url_from_env(self):
        with patch.dict(os.environ, {"BACKEND_URL": "http://test-backend:9999"}):
            import importlib
            import frontend.utils as mod
            importlib.reload(mod)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {}
            mock_resp.raise_for_status.return_value = None
            with patch("frontend.utils.httpx.get", return_value=mock_resp) as mock_get:
                mod.api_get("/ping")
            call_url = mock_get.call_args[0][0]
            assert "test-backend:9999" in call_url


class TestApiPost:
    def test_returns_json_on_success(self):
        from frontend.utils import api_post
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "abc"}
        mock_resp.raise_for_status.return_value = None

        with patch("frontend.utils.httpx.post", return_value=mock_resp):
            result = api_post("/v1/chat/completions", {"model": "gemini"})

        assert result == {"id": "abc"}

    def test_returns_error_dict_on_exception(self):
        from frontend.utils import api_post
        with patch("frontend.utils.httpx.post", side_effect=Exception("timeout")):
            result = api_post("/v1/chat/completions", {})
        assert "error" in result
        assert "timeout" in result["error"]


class TestApiDelete:
    def test_returns_json_on_success(self):
        from frontend.utils import api_delete
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status.return_value = None

        with patch("frontend.utils.httpx.delete", return_value=mock_resp):
            result = api_delete("/v1/rag/documents/abc123")

        assert result == {"status": "ok"}

    def test_returns_error_dict_on_exception(self):
        from frontend.utils import api_delete
        with patch("frontend.utils.httpx.delete", side_effect=Exception("404")):
            result = api_delete("/v1/rag/documents/missing")
        assert "error" in result
