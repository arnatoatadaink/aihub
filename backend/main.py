import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.models import router as models_router
from backend.api.rag import router as rag_router
from backend.api.templates import router as templates_router
from backend.api.jobs import router as jobs_router
from backend.api.media import router as media_router

load_dotenv()

app = FastAPI(
    title="AI Hub API",
    description="Unified AI provider gateway with OpenAI-compatible endpoints",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(rag_router)
app.include_router(templates_router)
app.include_router(jobs_router)
app.include_router(media_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
