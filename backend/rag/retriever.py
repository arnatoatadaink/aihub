"""RAG retrieval logic: embed query → search FAISS → format context."""
import os
import uuid
from pathlib import Path

import numpy as np

from .faiss_store import Document, FAISSStore


def _get_embedder():
    """Return an embedding function depending on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return _openai_embed
    if os.getenv("GEMINI_API_KEY"):
        return _gemini_embed
    raise RuntimeError("No embedding provider available. Set OPENAI_API_KEY or GEMINI_API_KEY.")


def _openai_embed(texts: list[str]) -> np.ndarray:
    import openai
    client = openai.OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in response.data], dtype=np.float32)


def _gemini_embed(texts: list[str]) -> np.ndarray:
    import google.generativeai as genai
    results = []
    for text in texts:
        res = genai.embed_content(model="models/embedding-001", content=text)
        results.append(res["embedding"])
    return np.array(results, dtype=np.float32)


class Retriever:
    def __init__(self, store: FAISSStore | None = None):
        self.store = store or FAISSStore()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, metadata: dict | None = None, chunk_size: int = 500) -> int:
        """Split text into chunks, embed, and store. Returns chunk count."""
        chunks = _chunk_text(text, chunk_size)
        embedder = _get_embedder()
        embeddings = embedder(chunks)
        docs = [
            Document(id=str(uuid.uuid4()), text=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
        self.store.add(docs, embeddings)
        return len(docs)

    def ingest_file(self, file_path: str, metadata: dict | None = None) -> int:
        """Read a text file and ingest its contents."""
        text = Path(file_path).read_text(encoding="utf-8")
        meta = {"source": file_path, **(metadata or {})}
        return self.ingest_text(text, metadata=meta)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant documents for a query."""
        embedder = _get_embedder()
        query_vec = embedder([query])[0]
        results = self.store.search(query_vec, top_k=top_k)
        return [
            {"text": doc.text, "metadata": doc.metadata, "score": dist}
            for doc, dist in results
        ]

    def build_context(self, query: str, top_k: int = 5) -> str:
        """Return a formatted context string ready to inject into a prompt."""
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        parts = [f"[Context {i + 1}]\n{d['text']}" for i, d in enumerate(docs)]
        return "\n\n".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Simple sentence-aware chunker."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = sentence + ". "
        else:
            current += sentence + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]
