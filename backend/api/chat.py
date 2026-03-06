import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.providers import PROVIDER_MAP

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-1.5-flash"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False
    provider: str = "gemini"


def _detect_provider(model: str, explicit: str) -> str:
    if explicit and (explicit in PROVIDER_MAP or explicit.startswith("custom_")):
        return explicit
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("gpt"):
        return "openai"
    return explicit or "gemini"


def _instantiate_provider(provider_name: str):
    """Return an instantiated provider, supporting custom_ prefix."""
    if provider_name.startswith("custom_"):
        from backend.providers.custom import build_custom_provider
        return build_custom_provider(provider_name)
    if provider_name not in PROVIDER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")
    return PROVIDER_MAP[provider_name]()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    provider_name = _detect_provider(request.model, request.provider)
    provider = _instantiate_provider(provider_name)
    messages = [m.model_dump() for m in request.messages]
    params = {
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
    }

    if request.stream:
        async def event_stream() -> AsyncGenerator[str, None]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            async for chunk in provider.stream(messages, params):
                data = (
                    f'data: {{"id":"{completion_id}","object":"chat.completion.chunk",'
                    f'"choices":[{{"delta":{{"content":{chunk!r}}},"index":0}}]}}\n\n'
                )
                yield data
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    content = await provider.generate(messages, params)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }
