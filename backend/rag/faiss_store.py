"""FAISS-backed vector store for document ingestion and retrieval."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

EMBED_DIM = 1536  # OpenAI text-embedding-3-small / Gemini embedding-001 default


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class FAISSStore:
    """Manages a FAISS flat-L2 index with an associated document store."""

    def __init__(self, index_path: str | None = None):
        self.index_path = Path(
            index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        )
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: faiss.IndexFlatL2 | None = None
        self._docs: list[Document] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _index_file(self) -> Path:
        return self.index_path / "index.faiss"

    def _meta_file(self) -> Path:
        return self.index_path / "metadata.json"

    def _load(self):
        if self._index_file().exists() and self._meta_file().exists():
            self._index = faiss.read_index(str(self._index_file()))
            with open(self._meta_file()) as f:
                raw = json.load(f)
            self._docs = [Document(**d) for d in raw]
        else:
            self._index = faiss.IndexFlatL2(EMBED_DIM)
            self._docs = []

    def save(self):
        faiss.write_index(self._index, str(self._index_file()))
        with open(self._meta_file(), "w") as f:
            json.dump(
                [{"id": d.id, "text": d.text, "metadata": d.metadata} for d in self._docs],
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, docs: list[Document], embeddings: np.ndarray):
        """Add documents with their pre-computed embeddings."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self._index.add(embeddings.astype(np.float32))
        self._docs.extend(docs)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Document, float]]:
        """Return top-k documents with L2 distances."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self._index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._docs[idx], float(dist)))
        return results

    def delete(self, doc_id: str) -> bool:
        """Remove document by id (rebuilds index)."""
        before = len(self._docs)
        keep_indices = [i for i, d in enumerate(self._docs) if d.id != doc_id]
        if len(keep_indices) == before:
            return False

        # Rebuild index from kept embeddings.
        # NOTE: faiss.IndexFlatL2 does not support direct deletion; we reconstruct.
        # reconstruct_n fetches all vectors in one call (O(n)) instead of per-item.
        new_index = faiss.IndexFlatL2(EMBED_DIM)
        if keep_indices:
            all_vecs = self._index.reconstruct_n(0, self._index.ntotal)
            kept_vecs = all_vecs[keep_indices].astype(np.float32)
            new_index.add(kept_vecs)
        self._index = new_index
        self._docs = [self._docs[i] for i in keep_indices]
        self.save()
        return True

    def list_documents(self) -> list[Document]:
        return list(self._docs)

    @property
    def count(self) -> int:
        return len(self._docs)
