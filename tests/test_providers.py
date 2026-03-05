"""Mock tests for AI providers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# BaseProvider
# ---------------------------------------------------------------------------

def test_base_provider_is_abstract():
    from backend.providers.base import BaseProvider
    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def test_get_models_returns_list(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("google.generativeai.configure"):
                from backend.providers.gemini import GeminiProvider
                provider = GeminiProvider()
                models = provider.get_models()
                assert isinstance(models, list)
                assert len(models) > 0

    def test_validate_key_returns_false_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib, backend.providers.gemini as mod
            importlib.reload(mod)
            provider = mod.GeminiProvider()
            provider.api_key = ""
            assert provider.validate_key() is False

    @pytest.mark.asyncio
    async def test_generate_calls_model(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                    mock_model = MagicMock()
                    mock_response = MagicMock()
                    mock_response.text = "Hello from Gemini"
                    mock_model.generate_content.return_value = mock_response
                    mock_model_cls.return_value = mock_model

                    from backend.providers.gemini import GeminiProvider
                    provider = GeminiProvider()
                    result = await provider.generate(
                        [{"role": "user", "content": "Hi"}],
                        {"model": "gemini-1.5-flash", "temperature": 0.7, "max_tokens": 256, "top_p": 1.0},
                    )
                    assert result == "Hello from Gemini"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                    chunk1 = MagicMock()
                    chunk1.text = "Hello "
                    chunk2 = MagicMock()
                    chunk2.text = "world"
                    mock_model = MagicMock()
                    mock_model.generate_content.return_value = iter([chunk1, chunk2])
                    mock_model_cls.return_value = mock_model

                    from backend.providers.gemini import GeminiProvider
                    provider = GeminiProvider()
                    chunks = []
                    async for chunk in provider.stream(
                        [{"role": "user", "content": "Hi"}],
                        {"model": "gemini-1.5-flash", "temperature": 0.7, "max_tokens": 256, "top_p": 1.0},
                    ):
                        chunks.append(chunk)
                    assert chunks == ["Hello ", "world"]


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_get_models_returns_list(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            from backend.providers.openai_prov import OpenAIProvider
            provider = OpenAIProvider()
            models = provider.get_models()
            assert isinstance(models, list)
            assert "gpt-4o" in models

    def test_validate_key_returns_false_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib, backend.providers.openai_prov as mod
            importlib.reload(mod)
            provider = mod.OpenAIProvider()
            provider.api_key = ""
            assert provider.validate_key() is False

    @pytest.mark.asyncio
    async def test_generate_returns_content(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            with patch("openai.AsyncOpenAI") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value = mock_client
                mock_choice = MagicMock()
                mock_choice.message.content = "Hello from OpenAI"
                mock_response = MagicMock()
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

                from backend.providers.openai_prov import OpenAIProvider
                provider = OpenAIProvider()
                provider.client = mock_client
                result = await provider.generate(
                    [{"role": "user", "content": "Hi"}],
                    {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 256, "top_p": 1.0},
                )
                assert result == "Hello from OpenAI"
