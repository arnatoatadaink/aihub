"""Google Veo video generation provider.

Uses the Gemini API's video generation capabilities (Veo model).
Veo is an async API — generation is submitted, then polled until ready.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

from .base import BaseProvider

AVAILABLE_MODELS = [
    "veo-2.0-generate-001",
]

POLL_INTERVAL = 5   # seconds
MAX_POLLS = 120     # 10 minutes max


class VeoProvider(BaseProvider):
    modal_type: str = "video"

    def __init__(self):
        import google.generativeai as genai
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def generate(self, messages: list, params: dict) -> str:
        """Submit video generation and poll until complete. Returns video URI."""
        import google.generativeai as genai

        prompt = messages[-1].get("content", "") if messages else ""
        model_name = params.get("model", AVAILABLE_MODELS[0])

        client = genai.Client(api_key=self.api_key)
        operation = client.models.generate_videos(
            model=model_name,
            prompt=prompt,
            config=genai.types.GenerateVideosConfig(
                number_of_videos=params.get("n", 1),
                duration_seconds=params.get("duration", 5),
                aspect_ratio=params.get("aspect_ratio", "16:9"),
            ),
        )

        # Poll for completion
        for _ in range(MAX_POLLS):
            if operation.done:
                break
            await asyncio.sleep(POLL_INTERVAL)
            operation = client.operations.get(operation)

        if not operation.done:
            raise TimeoutError("Video generation timed out")

        if operation.result and operation.result.generated_videos:
            video = operation.result.generated_videos[0]
            return video.uri or ""

        return ""

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        result = await self.generate(messages, params)
        yield result

    def get_models(self) -> list[str]:
        return AVAILABLE_MODELS

    def validate_key(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.generativeai as genai
            list(genai.list_models())
            return True
        except Exception:
            return False
