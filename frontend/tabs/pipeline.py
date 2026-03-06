"""Pipeline tab — build and run multi-step LLM pipelines.

UI layout:
  Left  : Pipeline builder (up to 5 steps via form + JSON editor)
  Right : Run panel — input text → step-by-step results
"""
from __future__ import annotations

import json

import gradio as gr

from frontend.utils import BACKEND_URL, api_get, api_post

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLE_PIPELINE = {
    "name": "翻訳→要約パイプライン",
    "description": "日本語を英語に翻訳してから3行で要約します",
    "steps": [
        {
            "id": "translate",
            "name": "翻訳",
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "system_prompt": "あなたは翻訳者です。入力を英語に翻訳してください。翻訳文のみを返してください。",
            "input_template": "{{input}}",
            "params": {"temperature": 0.2, "max_tokens": 2048},
        },
        {
            "id": "summarize",
            "name": "要約",
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "system_prompt": "Summarize the following text in 3 concise bullet points.",
            "input_template": "{{prev_output}}",
            "params": {"temperature": 0.5, "max_tokens": 512},
        },
    ],
}

_TEMPLATE_PRESETS = {
    "翻訳→要約": _EXAMPLE_PIPELINE,
    "添削→フォーマット": {
        "name": "添削→フォーマットパイプライン",
        "description": "文章を校正してから箇条書きに整形します",
        "steps": [
            {
                "id": "proofread",
                "name": "添削",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "system_prompt": "入力テキストの誤字・文法を修正してください。修正文のみを返してください。",
                "input_template": "{{input}}",
                "params": {"temperature": 0.2, "max_tokens": 2048},
            },
            {
                "id": "format",
                "name": "箇条書き化",
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "system_prompt": "入力を箇条書き（- で始まる行）にまとめてください。",
                "input_template": "{{prev_output}}",
                "params": {"temperature": 0.3, "max_tokens": 1024},
            },
        ],
    },
    "質問生成→回答": {
        "name": "質問生成→回答パイプライン",
        "description": "テキストから質問を生成し、その質問に自分で回答します",
        "steps": [
            {
                "id": "gen_questions",
                "name": "質問生成",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "与えられたテキストについて重要な質問を3つ生成してください。番号付きリストで返してください。",
                "input_template": "{{input}}",
                "params": {"temperature": 0.7, "max_tokens": 512},
            },
            {
                "id": "answer",
                "name": "回答",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "以下の質問にそれぞれ詳細に回答してください。",
                "input_template": "元テキスト:\n{{input}}\n\n質問:\n{{prev_output}}",
                "params": {"temperature": 0.7, "max_tokens": 2048},
            },
        ],
    },
}


def _format_step_results(results: list[dict]) -> str:
    """Convert step results list to a readable markdown string."""
    lines = []
    for r in results:
        status = "✅" if not r.get("error") else "❌"
        lines.append(f"## {status} Step {r['step_index'] + 1}: {r['step_name']}")
        lines.append(f"**Provider:** `{r['provider']}`  **Model:** `{r['model']}`")
        lines.append("")
        lines.append("**Input sent to LLM:**")
        lines.append(f"```\n{r['input']}\n```")
        lines.append("")
        if r.get("error"):
            lines.append(f"**Error:** {r['error']}")
        else:
            lines.append("**Output:**")
            lines.append(r["output"])
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action functions
# ---------------------------------------------------------------------------

def load_preset(preset_name: str) -> str:
    preset = _TEMPLATE_PRESETS.get(preset_name, _EXAMPLE_PIPELINE)
    return json.dumps(preset, indent=2, ensure_ascii=False)


def load_saved_pipelines() -> tuple[list[str], str]:
    result = api_get("/v1/pipelines")
    pipelines = result.get("pipelines", [])
    choices = [f"{p['id']} — {p['name']}" for p in pipelines]
    if not choices:
        return [], "保存済みパイプラインなし"
    return choices, f"{len(choices)} 件のパイプラインが保存されています"


def save_pipeline_action(pipeline_json: str) -> str:
    try:
        definition = json.loads(pipeline_json)
    except json.JSONDecodeError as e:
        return f"JSONパースエラー: {e}"
    result = api_post("/v1/pipelines", definition)
    if "error" in result:
        return f"保存失敗: {result['error']}"
    pid = result.get("pipeline", {}).get("id", "?")
    return f"保存しました (ID: {pid})"


def load_selected_pipeline(selection: str) -> str:
    if not selection:
        return ""
    pipeline_id = selection.split(" — ")[0].strip()
    result = api_get(f"/v1/pipelines/{pipeline_id}")
    if "error" in result:
        return f"// 取得失敗: {result['error']}"
    return json.dumps(result.get("pipeline", {}), indent=2, ensure_ascii=False)


def delete_selected_pipeline(selection: str) -> tuple[str, list[str], str]:
    if not selection:
        return "IDを選択してください", [], ""
    pipeline_id = selection.split(" — ")[0].strip()
    result = api_post(f"/v1/pipelines/{pipeline_id}", payload=None)
    # use DELETE via httpx directly since api_post wraps POST
    import httpx
    try:
        resp = httpx.delete(f"{BACKEND_URL}/v1/pipelines/{pipeline_id}", timeout=10)
        resp.raise_for_status()
        choices, msg = load_saved_pipelines()
        return f"削除しました: {pipeline_id}", choices, msg
    except Exception as e:
        return f"削除失敗: {e}", [], ""


def run_pipeline_action(pipeline_json: str, user_input: str) -> tuple[str, str, str]:
    """Return (step_results_md, final_output, status)."""
    if not user_input.strip():
        return "", "", "入力テキストを入力してください"
    try:
        definition = json.loads(pipeline_json)
    except json.JSONDecodeError as e:
        return "", "", f"JSONパースエラー: {e}"

    payload = {"input": user_input, "definition": definition}
    result = api_post("/v1/pipelines/run", payload, timeout=120)

    if "error" in result:
        return "", "", f"実行エラー: {result['error']}"

    results = result.get("results", [])
    final = result.get("final_output", "")
    step_md = _format_step_results(results)
    status = f"✅ 完了 ({len(results)} ステップ)"
    return step_md, final, status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_pipeline_tab() -> gr.Tab:
    with gr.Tab("Pipeline") as tab:
        gr.Markdown("## Pipeline — 複数プロバイダーを連結して処理")
        gr.Markdown(
            "LLMのステップを直列に連結します。各ステップの出力を次のステップへ渡せます。  \n"
            "`{{input}}` = 最初のユーザー入力 / `{{prev_output}}` = 直前ステップの出力 / "
            "`{{step:<id>}}` = 特定ステップの出力"
        )

        with gr.Row():
            # ----------------------------------------------------------------
            # Left — Pipeline Builder
            # ----------------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### パイプライン定義 (JSON)")

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="テンプレートから読み込む",
                        choices=list(_TEMPLATE_PRESETS.keys()),
                        value=None,
                    )
                    load_preset_btn = gr.Button("読み込む", size="sm")

                pipeline_json = gr.Code(
                    label="Pipeline JSON",
                    language="json",
                    value=json.dumps(_EXAMPLE_PIPELINE, indent=2, ensure_ascii=False),
                    lines=25,
                )

                with gr.Row():
                    save_btn = gr.Button("保存", variant="primary")
                    save_status = gr.Textbox(label="", interactive=False, scale=3)

                gr.Markdown("#### 保存済みパイプライン")
                with gr.Row():
                    saved_list = gr.Dropdown(label="保存済み", choices=[], value=None, scale=3)
                    refresh_btn = gr.Button("更新", size="sm")

                with gr.Row():
                    load_saved_btn = gr.Button("読み込む", size="sm")
                    delete_saved_btn = gr.Button("削除", size="sm", variant="stop")

                saved_status = gr.Textbox(label="", interactive=False)

            # ----------------------------------------------------------------
            # Right — Run Panel
            # ----------------------------------------------------------------
            with gr.Column(scale=3):
                gr.Markdown("### 実行")

                user_input = gr.Textbox(
                    label="入力テキスト",
                    placeholder="パイプラインへの最初の入力を入力...",
                    lines=5,
                )
                run_btn = gr.Button("パイプライン実行", variant="primary")
                run_status = gr.Textbox(label="ステータス", interactive=False)

                gr.Markdown("#### 最終出力")
                final_output = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=8,
                )

                gr.Markdown("#### ステップ別結果")
                step_results = gr.Markdown()

        # ----------------------------------------------------------------
        # Event wiring
        # ----------------------------------------------------------------
        load_preset_btn.click(fn=load_preset, inputs=preset_dropdown, outputs=pipeline_json)

        save_btn.click(fn=save_pipeline_action, inputs=pipeline_json, outputs=save_status)

        def _refresh():
            choices, msg = load_saved_pipelines()
            return gr.Dropdown(choices=choices, value=None), msg

        refresh_btn.click(fn=_refresh, outputs=[saved_list, saved_status])
        tab.select(fn=_refresh, outputs=[saved_list, saved_status])

        load_saved_btn.click(
            fn=load_selected_pipeline, inputs=saved_list, outputs=pipeline_json
        )

        def _delete(sel):
            msg, choices, list_msg = delete_selected_pipeline(sel)
            return msg, gr.Dropdown(choices=choices, value=None), list_msg

        delete_saved_btn.click(
            fn=_delete, inputs=saved_list, outputs=[save_status, saved_list, saved_status]
        )

        run_btn.click(
            fn=run_pipeline_action,
            inputs=[pipeline_json, user_input],
            outputs=[step_results, final_output, run_status],
        )

    return tab
