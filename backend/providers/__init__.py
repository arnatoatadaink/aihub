from .base import BaseProvider
from .gemini import GeminiProvider
from .openai_prov import OpenAIProvider
from .anthropic_prov import AnthropicProvider
from .voicevox import VoiceVoxProvider

PROVIDER_MAP = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "voicevox": VoiceVoxProvider,
}

__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "VoiceVoxProvider",
    "PROVIDER_MAP",
]
