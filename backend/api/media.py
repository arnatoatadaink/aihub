"""Multimodal media generation endpoints (image / video / music)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.providers import PROVIDER_MAP

router = APIRouter(prefix="/v1/media", tags=["media"])


# ======================================================================
# Request schemas
# ======================================================================

class ImageGenerateRequest(BaseModel):
    prompt: str
    provider: str = "imagen"        # "imagen" | "dalle"
    model: str = ""                 # empty → provider default
    n: int = 1
    size: str = "1024x1024"         # DALL-E size
    aspect_ratio: str = "1:1"       # Imagen aspect ratio
    quality: str = "standard"       # DALL-E quality


class VideoGenerateRequest(BaseModel):
    prompt: str
    provider: str = "veo"
    model: str = ""
    n: int = 1
    duration: int = 5               # seconds
    aspect_ratio: str = "16:9"


class MusicGenerateRequest(BaseModel):
    prompt: str
    provider: str = "musicfx"
    model: str = ""
    duration: int = 30              # seconds


# ======================================================================
# Endpoints
# ======================================================================

@router.post("/images/generate")
async def generate_image(req: ImageGenerateRequest):
    provider_name = req.provider
    if provider_name not in PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

    provider = PROVIDER_MAP[provider_name]()
    messages = [{"role": "user", "content": req.prompt}]
    params = {
        "model": req.model or provider.get_models()[0],
        "n": req.n,
        "size": req.size,
        "aspect_ratio": req.aspect_ratio,
        "quality": req.quality,
    }
    try:
        result = await provider.generate(messages, params)
        return {"url": result, "provider": provider_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos/generate")
async def generate_video(req: VideoGenerateRequest):
    provider_name = req.provider
    if provider_name not in PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

    provider = PROVIDER_MAP[provider_name]()
    messages = [{"role": "user", "content": req.prompt}]
    params = {
        "model": req.model or provider.get_models()[0],
        "n": req.n,
        "duration": req.duration,
        "aspect_ratio": req.aspect_ratio,
    }
    try:
        result = await provider.generate(messages, params)
        return {"url": result, "provider": provider_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/music/generate")
async def generate_music(req: MusicGenerateRequest):
    provider_name = req.provider
    if provider_name not in PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

    provider = PROVIDER_MAP[provider_name]()
    messages = [{"role": "user", "content": req.prompt}]
    params = {
        "model": req.model or provider.get_models()[0],
        "duration": req.duration,
    }
    try:
        result = await provider.generate(messages, params)
        return {"url": result, "provider": provider_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
