"""Custom server provider — connects to any OpenAI-compatible endpoint.

Supports Ollama, LM Studio, vLLM, LocalAI, etc.
Configuration is stored in data/custom_providers.json and managed via
the /v1/custom_providers REST API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncGenerator

from .base import BaseProvider

_STORE_PATH = Path(os.getenv("CUSTOM_PROVIDERS_PATH", "data/custom_providers.json"))


# ---------------------------------------------------------------------------
# Config store helpers
# ---------------------------------------------------------------------------

def _load_store() -> dict[str, dict]:
    """Load custom provider configs from disk. Returns {id: config}."""
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text())
    except Exception:
        return {}


def _save_store(store: dict[str, dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def list_custom_providers() -> list[dict]:
    return list(_load_store().values())


def get_custom_provider_config(provider_id: str) -> dict | None:
    return _load_store().get(provider_id)


def save_custom_provider(config: dict) -> dict:
    """Upsert a custom provider config. Returns the saved config."""
    store = _load_store()
    store[config["id"]] = config
    _save_store(store)
    return config


def delete_custom_provider(provider_id: str) -> bool:
    store = _load_store()
    if provider_id not in store:
        return False
    del store[provider_id]
    _save_store(store)
    return True


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class CustomProvider(BaseProvider):
    """OpenAI-compatible provider pointing at a user-configured endpoint."""

    modal_type: str = "text"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        models: list[str] | None = None,
        name: str = "custom",
    ):
        from openai import AsyncOpenAI

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "dummy"  # some local servers require a non-empty key
        self._models = models or []
        self.name = name

        # Append /v1 only when not already present
        api_base = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=api_base)

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    async def generate(self, messages: list, params: dict) -> str:
        model = params.get("model") or (self._models[0] if self._models else "default")
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 2048),
            top_p=params.get("top_p", 1.0),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        model = params.get("model") or (self._models[0] if self._models else "default")
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
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
        return self._models

    def validate_key(self) -> bool:
        """Try to fetch the model list from the remote server."""
        import httpx
        try:
            url = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key != "dummy" else {}
            resp = httpx.get(f"{url}/models", headers=headers, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory — create a CustomProvider from a stored config ID
# ---------------------------------------------------------------------------

def build_custom_provider(provider_id: str) -> CustomProvider:
    config = get_custom_provider_config(provider_id)
    if config is None:
        raise ValueError(f"Custom provider not found: {provider_id!r}")
    return CustomProvider(
        base_url=config["base_url"],
        api_key=config.get("api_key", ""),
        models=config.get("models", []),
        name=config.get("name", provider_id),
    )
