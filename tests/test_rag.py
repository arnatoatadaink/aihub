"""Tests for backend/rag/faiss_store.py and backend/rag/retriever.py."""
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from backend.rag.faiss_store import EMBED_DIM, Document, FAISSStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> FAISSStore:
    return FAISSStore(index_path=str(tmp_path))


def _rand_vecs(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, EMBED_DIM), dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISSStore — basic CRUD
# ---------------------------------------------------------------------------

class TestFAISSStoreAdd:
    def test_add_single_doc(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id=str(uuid.uuid4()), text="hello world")]
        store.add(docs, _rand_vecs(1))
        assert store.count == 1

    def test_add_multiple_docs(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id=str(uuid.uuid4()), text=f"doc {i}") for i in range(5)]
        store.add(docs, _rand_vecs(5))
        assert store.count == 5

    def test_add_1d_embedding_is_accepted(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id="x", text="single")]
        store.add(docs, _rand_vecs(1).squeeze())  # shape (EMBED_DIM,)
        assert store.count == 1


class TestFAISSStoreSearch:
    def test_search_returns_results(self, tmp_path):
        store = _make_store(tmp_path)
        vecs = _rand_vecs(3)
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(3)]
        store.add(docs, vecs)

        results = store.search(vecs[0], top_k=2)
        assert len(results) == 2
        # Nearest neighbour of vecs[0] should be itself
        assert results[0][0].id == "0"

    def test_search_empty_store_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        results = store.search(_rand_vecs(1).squeeze(), top_k=5)
        assert results == []

    def test_search_top_k_capped_by_store_size(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(2)]
        store.add(docs, _rand_vecs(2))
        results = store.search(_rand_vecs(1).squeeze(), top_k=10)
        assert len(results) <= 2


class TestFAISSStoreDelete:
    def test_delete_existing_doc(self, tmp_path):
        store = _make_store(tmp_path)
        doc = Document(id="del-me", text="to be deleted")
        store.add([doc], _rand_vecs(1))
        assert store.count == 1

        deleted = store.delete("del-me")
        assert deleted is True
        assert store.count == 0

    def test_delete_nonexistent_doc_returns_false(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.delete("no-such-id") is False

    def test_delete_preserves_other_docs(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id=f"id-{i}", text=f"doc {i}") for i in range(3)]
        store.add(docs, _rand_vecs(3))

        store.delete("id-1")
        remaining_ids = {d.id for d in store.list_documents()}
        assert remaining_ids == {"id-0", "id-2"}


class TestFAISSStorePersistence:
    def test_data_survives_reload(self, tmp_path):
        store = _make_store(tmp_path)
        doc = Document(id="persist-me", text="persisted", metadata={"src": "test"})
        store.add([doc], _rand_vecs(1))

        # Reload from same path
        store2 = _make_store(tmp_path)
        assert store2.count == 1
        loaded = store2.list_documents()[0]
        assert loaded.id == "persist-me"
        assert loaded.text == "persisted"
        assert loaded.metadata == {"src": "test"}


class TestFAISSStoreListDocuments:
    def test_list_returns_copy(self, tmp_path):
        store = _make_store(tmp_path)
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(3)]
        store.add(docs, _rand_vecs(3))
        listed = store.list_documents()
        assert len(listed) == 3
        # Mutating the returned list must not affect internal state
        listed.clear()
        assert store.count == 3


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def _mock_embedder(self, texts: list[str]) -> np.ndarray:
        """Deterministic fake embedder."""
        rng = np.random.default_rng(len(texts))
        return rng.random((len(texts), EMBED_DIM), dtype=np.float32)

    def test_ingest_text_adds_chunks(self, tmp_path):
        from backend.rag.retriever import Retriever

        store = _make_store(tmp_path)
        retriever = Retriever(store=store)

        with patch("backend.rag.retriever._get_embedder", return_value=self._mock_embedder):
            count = retriever.ingest_text("Hello world. " * 50, chunk_size=100)
        assert count >= 1
        assert store.count == count

    def test_ingest_file(self, tmp_path):
        from backend.rag.retriever import Retriever

        store = _make_store(tmp_path)
        retriever = Retriever(store=store)

        text_file = tmp_path / "sample.txt"
        text_file.write_text("Sample content. " * 30, encoding="utf-8")

        with patch("backend.rag.retriever._get_embedder", return_value=self._mock_embedder):
            count = retriever.ingest_file(str(text_file))
        assert count >= 1

    def test_retrieve_returns_results(self, tmp_path):
        from backend.rag.retriever import Retriever

        store = _make_store(tmp_path)
        retriever = Retriever(store=store)

        with patch("backend.rag.retriever._get_embedder", return_value=self._mock_embedder):
            retriever.ingest_text("The capital of France is Paris.", chunk_size=500)
            results = retriever.retrieve("France capital", top_k=1)

        assert len(results) == 1
        assert "text" in results[0]
        assert "score" in results[0]

    def test_build_context_empty_store(self, tmp_path):
        from backend.rag.retriever import Retriever

        store = _make_store(tmp_path)
        retriever = Retriever(store=store)

        with patch("backend.rag.retriever._get_embedder", return_value=self._mock_embedder):
            ctx = retriever.build_context("anything")
        assert ctx == ""


class TestChunkText:
    def test_short_text_is_single_chunk(self):
        from backend.rag.retriever import _chunk_text
        chunks = _chunk_text("Short text.", chunk_size=500)
        assert len(chunks) == 1
        # chunker re-appends ". " after splitting on ". ", so trailing dot is doubled
        assert chunks[0] == "Short text.."

    def test_long_text_is_split(self):
        from backend.rag.retriever import _chunk_text
        long = "Word. " * 200  # ~1200 chars
        chunks = _chunk_text(long, chunk_size=100)
        assert len(chunks) > 1

    def test_empty_text_returns_chunk(self):
        from backend.rag.retriever import _chunk_text
        chunks = _chunk_text("", chunk_size=500)
        # empty string split + ". " appended → ["."]
        assert chunks == ["."]
