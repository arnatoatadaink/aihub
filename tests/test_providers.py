"""Mock tests for AI providers."""
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — inject fake google.generativeai into sys.modules so providers
# that do `import google.generativeai as genai` get the mock instead of the
# real (potentially broken) package.
# ---------------------------------------------------------------------------

def _fake_genai():
    """Return a MagicMock that quacks like google.generativeai."""
    genai = MagicMock()
    genai.types = MagicMock()
    genai.types.GenerationConfig = MagicMock(return_value=MagicMock())
    return genai


def _install_fake_genai():
    """Insert stub modules so `import google.generativeai` succeeds."""
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    genai_mod = MagicMock(name="google.generativeai")
    genai_mod.types = MagicMock()
    genai_mod.types.GenerationConfig = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("google", google_pkg)
    sys.modules["google.generativeai"] = genai_mod
    return genai_mod


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
    def setup_method(self):
        # Ensure a fresh module each test by removing cached import
        sys.modules.pop("backend.providers.gemini", None)

    def test_get_models_returns_list(self):
        genai_mock = _install_fake_genai()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            from backend.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            models = provider.get_models()
            assert isinstance(models, list)
            assert len(models) > 0

    def test_validate_key_returns_false_when_no_key(self):
        _install_fake_genai()
        with patch.dict("os.environ", {}, clear=True):
            from backend.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            provider.api_key = ""
            assert provider.validate_key() is False

    @pytest.mark.asyncio
    async def test_generate_calls_model(self):
        genai_mock = _install_fake_genai()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        genai_mock.GenerativeModel.return_value = mock_model

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            from backend.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            result = await provider.generate(
                [{"role": "user", "content": "Hi"}],
                {"model": "gemini-1.5-flash", "temperature": 0.7, "max_tokens": 256, "top_p": 1.0},
            )
            assert result == "Hello from Gemini"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        genai_mock = _install_fake_genai()
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = iter([chunk1, chunk2])
        genai_mock.GenerativeModel.return_value = mock_model

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
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


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_get_models_returns_list(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
            from backend.providers.anthropic_prov import AnthropicProvider
            provider = AnthropicProvider()
            models = provider.get_models()
            assert isinstance(models, list)
            assert "claude-sonnet-4-6" in models

    def test_validate_key_returns_false_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib, backend.providers.anthropic_prov as mod
            importlib.reload(mod)
            provider = mod.AnthropicProvider()
            provider.api_key = ""
            assert provider.validate_key() is False

    @pytest.mark.asyncio
    async def test_generate_returns_content(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
            with patch("anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value = mock_client
                mock_block = MagicMock()
                mock_block.text = "Hello from Claude"
                mock_response = MagicMock()
                mock_response.content = [mock_block]
                mock_client.messages.create = AsyncMock(return_value=mock_response)

                from backend.providers.anthropic_prov import AnthropicProvider
                provider = AnthropicProvider()
                provider.client = mock_client
                result = await provider.generate(
                    [{"role": "user", "content": "Hi"}],
                    {"model": "claude-sonnet-4-6", "temperature": 0.7, "max_tokens": 256},
                )
                assert result == "Hello from Claude"


# ---------------------------------------------------------------------------
# DallEProvider
# ---------------------------------------------------------------------------

class TestDallEProvider:
    def test_get_models_returns_list(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            from backend.providers.dalle import DallEProvider
            provider = DallEProvider()
            models = provider.get_models()
            assert isinstance(models, list)
            assert "dall-e-3" in models

    def test_validate_key_returns_false_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib, backend.providers.dalle as mod
            importlib.reload(mod)
            provider = mod.DallEProvider()
            provider.api_key = ""
            assert provider.validate_key() is False


# ---------------------------------------------------------------------------
# VoiceVoxProvider
# ---------------------------------------------------------------------------

class TestVoiceVoxProvider:
    def test_get_models(self):
        from backend.providers.voicevox import VoiceVoxProvider
        provider = VoiceVoxProvider()
        assert provider.get_models() == ["voicevox"]

    def test_modal_type_is_audio(self):
        from backend.providers.voicevox import VoiceVoxProvider
        provider = VoiceVoxProvider()
        assert provider.modal_type == "audio"


# ---------------------------------------------------------------------------
# ImagenProvider
# ---------------------------------------------------------------------------

class TestImagenProvider:
    def test_get_models(self):
        _install_fake_genai()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            from backend.providers.imagen import ImagenProvider
            provider = ImagenProvider()
            models = provider.get_models()
            assert isinstance(models, list)
            assert any("imagen" in m for m in models)

    def test_modal_type_is_image(self):
        _install_fake_genai()
        from backend.providers.imagen import ImagenProvider
        assert ImagenProvider().modal_type == "image"


# ---------------------------------------------------------------------------
# VeoProvider
# ---------------------------------------------------------------------------

class TestVeoProvider:
    def test_get_models(self):
        from backend.providers.veo import VeoProvider
        provider = VeoProvider()
        assert provider.get_models() == ["veo-2.0-generate-001"]

    def test_modal_type_is_video(self):
        from backend.providers.veo import VeoProvider
        assert VeoProvider().modal_type == "video"


# ---------------------------------------------------------------------------
# MusicFXProvider
# ---------------------------------------------------------------------------

class TestMusicFXProvider:
    def test_get_models(self):
        from backend.providers.musicfx import MusicFXProvider
        provider = MusicFXProvider()
        assert provider.get_models() == ["musicfx"]

    def test_modal_type_is_audio(self):
        from backend.providers.musicfx import MusicFXProvider
        assert MusicFXProvider().modal_type == "audio"
