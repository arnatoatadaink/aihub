import os
from typing import AsyncGenerator

import anthropic

from .base import BaseProvider

AVAILABLE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider(BaseProvider):
    modal_type: str = "text"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return self.client

    @staticmethod
    def _convert_content(content) -> str | list:
        """Convert OpenAI-style content to Anthropic format.

        str  → str (unchanged)
        list → list of Anthropic content blocks
        """
        if isinstance(content, str):
            return content
        blocks = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "text")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        })
            else:
                # Pydantic ContentPart object
                if part.type == "text":
                    blocks.append({"type": "text", "text": part.text or ""})
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url.get("url", "")
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        })
        return blocks

    def _split_messages(self, messages: list) -> tuple[str | None, list]:
        """Extract system prompt and convert to Anthropic message format."""
        system_prompt = None
        conversation = []
        for msg in messages:
            if msg.get("role") == "system":
                raw = msg.get("content", "")
                system_prompt = raw if isinstance(raw, str) else ""
            else:
                conversation.append({
                    "role": msg["role"],
                    "content": self._convert_content(msg["content"]),
                })
        return system_prompt, conversation

    async def generate(self, messages: list, params: dict) -> str:
        client = self._get_client()
        system_prompt, conversation = self._split_messages(messages)

        kwargs = dict(
            model=params.get("model", "claude-sonnet-4-6"),
            messages=conversation,
            max_tokens=params.get("max_tokens", 2048),
            temperature=params.get("temperature", 0.7),
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)
        return response.content[0].text

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        client = self._get_client()
        system_prompt, conversation = self._split_messages(messages)

        kwargs = dict(
            model=params.get("model", "claude-sonnet-4-6"),
            messages=conversation,
            max_tokens=params.get("max_tokens", 2048),
            temperature=params.get("temperature", 0.7),
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def get_models(self) -> list[str]:
        return AVAILABLE_MODELS

    def validate_key(self) -> bool:
        if not self.api_key:
            return False
        try:
            sync_client = anthropic.Anthropic(api_key=self.api_key)
            sync_client.models.list()
            return True
        except Exception:
            return False
