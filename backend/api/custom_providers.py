"""Custom provider management endpoints.

Endpoints:
  GET    /v1/custom_providers               — list all custom providers
  POST   /v1/custom_providers               — create / update a custom provider
  GET    /v1/custom_providers/{id}          — get a single config
  DELETE /v1/custom_providers/{id}          — remove a custom provider
  GET    /v1/custom_providers/{id}/validate — test connectivity
  POST   /v1/custom_providers/fetch_models  — probe a server for its model list
"""
from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.providers.custom import (
    build_custom_provider,
    delete_custom_provider,
    get_custom_provider_config,
    list_custom_providers,
    save_custom_provider,
)

router = APIRouter(prefix="/v1/custom_providers", tags=["custom_providers"])


class CustomProviderCreate(BaseModel):
    id: str = Field(default="", description="省略時は自動生成")
    name: str
    base_url: str = Field(..., description="例: http://localhost:11434  (Ollama)")
    api_key: str = ""
    models: list[str] = Field(default_factory=list, description="モデル名の一覧 (空でも可)")
    description: str = ""


class FetchModelsRequest(BaseModel):
    base_url: str
    api_key: str = ""


@router.get("")
async def list_providers():
    return {"providers": list_custom_providers()}


@router.post("")
async def create_provider(body: CustomProviderCreate):
    config = body.model_dump()
    if not config["id"]:
        config["id"] = f"custom_{uuid.uuid4().hex[:8]}"
    saved = save_custom_provider(config)
    return {"provider": saved}


@router.get("/{provider_id}")
async def get_provider(provider_id: str):
    config = get_custom_provider_config(provider_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")
    return {"provider": config}


@router.delete("/{provider_id}")
async def remove_provider(provider_id: str):
    if not delete_custom_provider(provider_id):
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")
    return {"deleted": provider_id}


@router.post("/fetch_models")
async def fetch_models(body: FetchModelsRequest):
    """Probe a remote OpenAI-compatible server's /v1/models endpoint.

    Returns the list of available model IDs so the user can paste them into
    the custom provider form without having to type them manually.
    """
    api_base = body.base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    headers: dict[str, str] = {}
    if body.api_key:
        headers["Authorization"] = f"Bearer {body.api_key}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(f"{api_base}/models", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI format: {"data": [{"id": "...", ...}]}
        if "data" in data:
            models = [m["id"] for m in data["data"] if "id" in m]
        else:
            # Some servers return {"models": [...]} or a plain list
            raw = data.get("models", data) if isinstance(data, dict) else data
            models = [m["id"] if isinstance(m, dict) else str(m) for m in (raw or [])]
        return {"models": sorted(models)}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Server returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach server: {e}")


@router.get("/{provider_id}/validate")
async def validate_provider(provider_id: str):
    config = get_custom_provider_config(provider_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")
    try:
        provider = build_custom_provider(provider_id)
        ok = provider.validate_key()
        return {"provider_id": provider_id, "reachable": ok}
    except Exception as e:
        return {"provider_id": provider_id, "reachable": False, "error": str(e)}
