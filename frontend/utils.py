"""Shared frontend utilities: HTTP client helpers and provider/model config."""
import os

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Single source of truth for provider → model list mapping used across tabs.
PROVIDER_MODEL_MAP: dict[str, list[str]] = {
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
}


def api_get(path: str, timeout: int = 15) -> dict:
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, payload: dict | None = None, timeout: int = 60, **kwargs) -> dict:
    try:
        resp = httpx.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_all_providers() -> dict[str, list[str]]:
    """Return provider→models map including saved custom providers."""
    result = {**PROVIDER_MODEL_MAP}
    custom = api_get("/v1/custom_providers")
    for p in custom.get("providers", []):
        pid = p["id"]
        result[pid] = p.get("models", []) or ["default"]
    return result


def api_delete(path: str, timeout: int = 15) -> dict:
    try:
        resp = httpx.delete(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
