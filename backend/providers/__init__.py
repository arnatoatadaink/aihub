from .base import BaseProvider
from .gemini import GeminiProvider
from .openai_prov import OpenAIProvider
from .anthropic_prov import AnthropicProvider
from .voicevox import VoiceVoxProvider
from .imagen import ImagenProvider
from .dalle import DallEProvider
from .veo import VeoProvider
from .musicfx import MusicFXProvider
from .custom import CustomProvider, build_custom_provider

PROVIDER_MAP = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "voicevox": VoiceVoxProvider,
    "imagen": ImagenProvider,
    "dalle": DallEProvider,
    "veo": VeoProvider,
    "musicfx": MusicFXProvider,
}

__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "VoiceVoxProvider",
    "ImagenProvider",
    "DallEProvider",
    "VeoProvider",
    "MusicFXProvider",
    "CustomProvider",
    "build_custom_provider",
    "PROVIDER_MAP",
]
