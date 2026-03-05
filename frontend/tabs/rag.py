"""RAG Management tab — document ingestion, query testing, document list."""
import os

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _get(path: str) -> dict:
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, **kwargs) -> dict:
    try:
        resp = httpx.post(f"{BACKEND_URL}{path}", timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# Action functions
# ------------------------------------------------------------------

def ingest_text(text: str, chunk_size: int) -> tuple[str, str]:
    if not text.strip():
        return "テキストを入力してください", refresh_doc_list()
    result = _post("/v1/rag/ingest/text", json={"text": text, "chunk_size": int(chunk_size)})
    if "error" in result:
        return f"Error: {result['error']}", refresh_doc_list()
    return (
        f"完了: {result['chunks_added']} チャンク追加 (合計 {result['total_docs']} docs)",
        refresh_doc_list(),
    )


def ingest_file(file) -> tuple[str, str]:
    if file is None:
        return "ファイルを選択してください", refresh_doc_list()
    try:
        with open(file.name, "rb") as f:
            result = _post(
                "/v1/rag/ingest/file",
                files={"file": (os.path.basename(file.name), f, "text/plain")},
            )
        if "error" in result:
            return f"Error: {result['error']}", refresh_doc_list()
        return (
            f"完了: {result['chunks_added']} チャンク追加 (合計 {result['total_docs']} docs)",
            refresh_doc_list(),
        )
    except Exception as e:
        return f"Error: {e}", refresh_doc_list()


def run_query(query: str, top_k: int) -> str:
    if not query.strip():
        return "クエリを入力してください"
    result = _post("/v1/rag/query", json={"query": query, "top_k": int(top_k)})
    if "error" in result:
        return f"Error: {result['error']}"
    if not result.get("results"):
        return "結果なし"
    lines = []
    for i, r in enumerate(result["results"], 1):
        lines.append(f"**[{i}] Score: {r['score']:.4f}**")
        if r.get("metadata"):
            lines.append(f"Metadata: {r['metadata']}")
        lines.append(r["text"])
        lines.append("---")
    return "\n\n".join(lines)


def refresh_doc_list() -> str:
    result = _get("/v1/rag/documents")
    if "error" in result:
        return f"Error: {result['error']}"
    docs = result.get("documents", [])
    if not docs:
        return "ドキュメントなし"
    lines = [f"**合計 {result['total']} docs**\n"]
    for d in docs:
        meta_str = str(d["metadata"]) if d["metadata"] else ""
        lines.append(f"- `{d['id'][:8]}...` {meta_str}\n  {d['preview']}")
    return "\n".join(lines)


def delete_doc(doc_id: str) -> tuple[str, str]:
    doc_id = doc_id.strip()
    if not doc_id:
        return "Doc IDを入力してください", refresh_doc_list()
    try:
        resp = httpx.delete(f"{BACKEND_URL}/v1/rag/documents/{doc_id}", timeout=15)
        if resp.status_code == 404:
            return "ドキュメントが見つかりません", refresh_doc_list()
        resp.raise_for_status()
        return "削除完了", refresh_doc_list()
    except Exception as e:
        return f"Error: {e}", refresh_doc_list()


# ------------------------------------------------------------------
# Tab builder
# ------------------------------------------------------------------

def build_rag_tab() -> gr.Tab:
    with gr.Tab("RAG") as tab:
        gr.Markdown("## RAG 管理")

        with gr.Row():
            # Left: ingestion + query
            with gr.Column(scale=2):
                with gr.Accordion("テキスト投入", open=True):
                    raw_text = gr.Textbox(
                        label="テキスト", lines=6, placeholder="ドキュメント本文を貼り付け..."
                    )
                    chunk_size = gr.Slider(
                        label="Chunk Size", minimum=100, maximum=2000, value=500, step=50
                    )
                    ingest_text_btn = gr.Button("テキストを投入", variant="primary")
                    ingest_text_status = gr.Textbox(label="ステータス", interactive=False)

                with gr.Accordion("ファイル投入", open=False):
                    upload_file = gr.File(label="テキストファイル (.txt, .md)", file_types=[".txt", ".md"])
                    ingest_file_btn = gr.Button("ファイルを投入", variant="primary")
                    ingest_file_status = gr.Textbox(label="ステータス", interactive=False)

                gr.Markdown("### クエリテスト")
                query_input = gr.Textbox(label="検索クエリ", placeholder="質問を入力...")
                top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
                query_btn = gr.Button("検索", variant="primary")
                query_result = gr.Markdown()

            # Right: document list
            with gr.Column(scale=1):
                gr.Markdown("### ドキュメント一覧")
                refresh_btn = gr.Button("更新")
                doc_list = gr.Markdown()
                gr.Markdown("### 削除")
                delete_id = gr.Textbox(label="Doc ID (フル)", placeholder="UUID...")
                delete_btn = gr.Button("削除", variant="stop")
                delete_status = gr.Textbox(label="ステータス", interactive=False)

        # Events
        ingest_text_btn.click(
            fn=ingest_text,
            inputs=[raw_text, chunk_size],
            outputs=[ingest_text_status, doc_list],
        )
        ingest_file_btn.click(
            fn=ingest_file,
            inputs=[upload_file],
            outputs=[ingest_file_status, doc_list],
        )
        query_btn.click(fn=run_query, inputs=[query_input, top_k], outputs=query_result)
        refresh_btn.click(fn=refresh_doc_list, inputs=[], outputs=doc_list)
        delete_btn.click(fn=delete_doc, inputs=[delete_id], outputs=[delete_status, doc_list])

        # Load doc list on tab open
        tab.select(fn=refresh_doc_list, inputs=[], outputs=doc_list)

    return tab
