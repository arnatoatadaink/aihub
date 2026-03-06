"""OpenAI DALL-E image generation provider."""
from __future__ import annotations

import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .base import BaseProvider

AVAILABLE_MODELS = [
    "dall-e-3",
    "dall-e-2",
]


class DallEProvider(BaseProvider):
    modal_type: str = "image"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    def _get_client(self) -> AsyncOpenAI:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not set")
        return self.client

    async def generate(self, messages: list, params: dict) -> str:
        """Generate an image and return the URL."""
        client = self._get_client()
        raw = messages[-1].get("content", "") if messages else ""
        prompt = raw if isinstance(raw, str) else "\n".join(
            p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text"
        )
        model = params.get("model", "dall-e-3")
        size = params.get("size", "1024x1024")
        quality = params.get("quality", "standard")

        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
            response_format="url",
        )
        return response.data[0].url or ""

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        result = await self.generate(messages, params)
        yield result

    def get_models(self) -> list[str]:
        return AVAILABLE_MODELS

    def validate_key(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            client.models.list()
            return True
        except Exception:
            return False
