import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .base import BaseProvider


def _normalize_messages(messages: list) -> list:
    """Ensure content list items are plain dicts (not Pydantic objects)."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part)
                else:
                    # Pydantic ContentPart object
                    if part.type == "text":
                        parts.append({"type": "text", "text": part.text or ""})
                    elif part.type == "image_url" and part.image_url:
                        parts.append({"type": "image_url", "image_url": part.image_url})
            result.append({"role": msg["role"], "content": parts})
        else:
            result.append(msg)
    return result

AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


class OpenAIProvider(BaseProvider):
    modal_type: str = "text"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    def _get_client(self) -> AsyncOpenAI:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not set")
        return self.client

    async def generate(self, messages: list, params: dict) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=params.get("model", "gpt-4o-mini"),
            messages=_normalize_messages(messages),
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 2048),
            top_p=params.get("top_p", 1.0),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=params.get("model", "gpt-4o-mini"),
            messages=_normalize_messages(messages),
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 2048),
            top_p=params.get("top_p", 1.0),
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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
