"""Google MusicFX music generation provider.

MusicFX is accessed via the Generative Language API (AI Studio).
This provider wraps the REST endpoint for audio generation.
"""
from __future__ import annotations

import base64
import os
from typing import AsyncGenerator

import aiohttp

from .base import BaseProvider

AVAILABLE_MODELS = [
    "musicfx",
]


class MusicFXProvider(BaseProvider):
    modal_type: str = "audio"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate(self, messages: list, params: dict) -> str:
        """Generate music from a text prompt. Returns base64 audio data URI."""
        raw = messages[-1].get("content", "") if messages else ""
        prompt = raw if isinstance(raw, str) else "\n".join(
            p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text"
        )
        duration = params.get("duration", 30)

        url = f"{self.base_url}/models/musicfx:generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "audio/mp3",
                "duration": duration,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                params={"key": self.api_key},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise RuntimeError(f"MusicFX API error ({resp.status}): {error_body}")
                data = await resp.json()

        # Extract audio bytes from response
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                mime = part["inlineData"].get("mimeType", "audio/mp3")
                b64 = part["inlineData"]["data"]
                return f"data:{mime};base64,{b64}"

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
            genai.configure(api_key=self.api_key)
            list(genai.list_models())
            return True
        except Exception:
            return False
