import time

from fastapi import APIRouter

from backend.providers import PROVIDER_MAP

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    models = []
    for provider_name, provider_cls in PROVIDER_MAP.items():
        provider = provider_cls()
        for model_id in provider.get_models():
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name,
                }
            )
    return {"object": "list", "data": models}
