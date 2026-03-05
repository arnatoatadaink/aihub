"""RAG management endpoints."""
import os
import tempfile
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.rag.faiss_store import FAISSStore
from backend.rag.retriever import Retriever

router = APIRouter(prefix="/v1/rag", tags=["rag"])

_store = FAISSStore()
_retriever = Retriever(store=_store)


class IngestTextRequest(BaseModel):
    text: str
    metadata: dict = {}
    chunk_size: int = 500


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/ingest/text")
async def ingest_text(req: IngestTextRequest):
    try:
        count = _retriever.ingest_text(req.text, metadata=req.metadata, chunk_size=req.chunk_size)
        return {"status": "ok", "chunks_added": count, "total_docs": _store.count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "upload")[1] or ".txt"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        count = _retriever.ingest_file(tmp_path, metadata={"filename": file.filename})
        os.unlink(tmp_path)
        return {"status": "ok", "chunks_added": count, "total_docs": _store.count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query(req: QueryRequest):
    try:
        results = _retriever.retrieve(req.query, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    docs = _store.list_documents()
    return {
        "total": len(docs),
        "documents": [
            {"id": d.id, "preview": d.text[:120], "metadata": d.metadata}
            for d in docs
        ],
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted = _store.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "ok", "total_docs": _store.count}
