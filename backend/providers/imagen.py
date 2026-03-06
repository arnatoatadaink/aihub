"""Google Imagen image generation provider."""
from __future__ import annotations

import base64
import os
from typing import AsyncGenerator

from .base import BaseProvider

AVAILABLE_MODELS = [
    "imagen-3.0-generate-002",
    "imagen-3.0-fast-generate-001",
]


class ImagenProvider(BaseProvider):
    modal_type: str = "image"

    def __init__(self):
        import google.generativeai as genai
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def generate(self, messages: list, params: dict) -> str:
        """Generate an image and return a base64-encoded data URI."""
        import google.generativeai as genai

        raw = messages[-1].get("content", "") if messages else ""
        prompt = raw if isinstance(raw, str) else "\n".join(
            p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text"
        )
        model_name = params.get("model", AVAILABLE_MODELS[0])

        model = genai.ImageGenerationModel.from_pretrained(model_name)
        response = model.generate_images(
            prompt=prompt,
            number_of_images=params.get("n", 1),
            aspect_ratio=params.get("aspect_ratio", "1:1"),
        )

        if not response.images:
            return ""

        img = response.images[0]
        b64 = base64.b64encode(img._image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

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
