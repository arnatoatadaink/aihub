import json
import os

import gradio as gr
import httpx

from frontend.utils import BACKEND_URL, api_delete, api_get, api_post


def save_settings(gemini_key: str, openai_key: str, anthropic_key: str) -> str:
    """Save API keys to environment variables (runtime only)."""
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    return "Settings saved (runtime only). Restart the backend to apply changes."


def check_backend_health() -> str:
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            return "Backend: OK"
        return f"Backend: ERROR (status {resp.status_code})"
    except Exception as e:
        return f"Backend: Unreachable ({e})"


# ---------------------------------------------------------------------------
# Custom provider helpers
# ---------------------------------------------------------------------------

def _list_custom_providers() -> tuple[list[str], str]:
    result = api_get("/v1/custom_providers")
    providers = result.get("providers", [])
    choices = [f"{p['id']} — {p['name']} ({p['base_url']})" for p in providers]
    summary = f"{len(choices)} 件のカスタムプロバイダーが登録されています" if choices else "カスタムプロバイダーなし"
    return choices, summary


def save_custom_provider(name: str, base_url: str, api_key: str, models_raw: str, description: str) -> str:
    if not name.strip() or not base_url.strip():
        return "名前とBase URLは必須です"
    models = [m.strip() for m in models_raw.splitlines() if m.strip()]
    payload = {
        "name": name.strip(),
        "base_url": base_url.strip(),
        "api_key": api_key.strip(),
        "models": models,
        "description": description.strip(),
    }
    result = api_post("/v1/custom_providers", payload)
    if "error" in result:
        return f"保存失敗: {result['error']}"
    pid = result.get("provider", {}).get("id", "?")
    return f"保存しました (ID: {pid})"


def validate_custom_provider(selection: str) -> str:
    if not selection:
        return "プロバイダーを選択してください"
    pid = selection.split(" — ")[0].strip()
    result = api_get(f"/v1/custom_providers/{pid}/validate")
    if result.get("reachable"):
        return f"✅ 接続OK: {pid}"
    err = result.get("error", "unreachable")
    return f"❌ 接続失敗: {err}"


def delete_custom_provider(selection: str) -> tuple[str, list[str], str]:
    if not selection:
        return "プロバイダーを選択してください", [], ""
    pid = selection.split(" — ")[0].strip()
    result = api_delete(f"/v1/custom_providers/{pid}")
    if "error" in result:
        return f"削除失敗: {result['error']}", [], ""
    choices, summary = _list_custom_providers()
    return f"削除しました: {pid}", choices, summary


def load_provider_detail(selection: str) -> tuple[str, str, str, str, str]:
    """Load a saved custom provider into the form fields."""
    if not selection:
        return "", "", "", "", ""
    pid = selection.split(" — ")[0].strip()
    result = api_get(f"/v1/custom_providers/{pid}")
    if "error" in result:
        return "", "", "", "", f"取得失敗: {result['error']}"
    p = result.get("provider", {})
    models_text = "\n".join(p.get("models", []))
    return p.get("name", ""), p.get("base_url", ""), p.get("api_key", ""), models_text, p.get("description", "")


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_settings_tab() -> gr.Tab:
    with gr.Tab("Settings") as tab:

        # ---- API Keys -------------------------------------------------------
        gr.Markdown("## API Key Settings")
        gr.Markdown(
            "APIキーはバックエンドの環境変数として適用されます。"
            "ここで入力した値はランタイムのみ有効で、ファイルには保存されません。"
        )
        with gr.Group():
            gemini_key = gr.Textbox(
                label="Google Gemini API Key",
                placeholder="AIza...",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
            )
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
            )
            anthropic_key = gr.Textbox(
                label="Anthropic API Key",
                placeholder="sk-ant-...",
                type="password",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
            )

        save_btn = gr.Button("Save Settings", variant="primary")
        status_msg = gr.Textbox(label="Status", interactive=False)

        save_btn.click(
            fn=save_settings,
            inputs=[gemini_key, openai_key, anthropic_key],
            outputs=status_msg,
        )

        # ---- Custom Providers -----------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## カスタムプロバイダー設定")
        gr.Markdown(
            "OllamaやLM Studio、vLLM等のOpenAI互換エンドポイントを追加できます。  \n"
            "追加したプロバイダーはPlaygroundやPipelineタブで `custom_<id>` として利用できます。"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 新規登録 / 編集")
                cp_name = gr.Textbox(label="名前", placeholder="Ollama Local")
                cp_url = gr.Textbox(
                    label="Base URL",
                    placeholder="http://localhost:11434",
                )
                cp_key = gr.Textbox(
                    label="API Key (不要な場合は空欄)",
                    placeholder="",
                    type="password",
                )
                cp_models = gr.Textbox(
                    label="モデル一覧 (1行1モデル)",
                    placeholder="llama3\nmistral\ngemma3",
                    lines=4,
                )
                cp_desc = gr.Textbox(label="説明 (任意)", placeholder="Local Ollama instance")
                cp_save_btn = gr.Button("登録 / 更新", variant="primary")
                cp_save_status = gr.Textbox(label="", interactive=False)

            with gr.Column():
                gr.Markdown("### 登録済みプロバイダー")
                cp_list = gr.Dropdown(label="一覧", choices=[], value=None)
                with gr.Row():
                    cp_refresh_btn = gr.Button("更新", size="sm")
                    cp_load_btn = gr.Button("フォームに読み込む", size="sm")
                    cp_validate_btn = gr.Button("接続テスト", size="sm")
                    cp_delete_btn = gr.Button("削除", size="sm", variant="stop")
                cp_list_status = gr.Textbox(label="", interactive=False)

        # ---- Backend Health -------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## Backend Status")
        check_btn = gr.Button("Check Backend Health")
        health_msg = gr.Textbox(label="Health", interactive=False)
        check_btn.click(fn=check_backend_health, inputs=[], outputs=health_msg)

        # ---- Event wiring ---------------------------------------------------

        cp_save_btn.click(
            fn=save_custom_provider,
            inputs=[cp_name, cp_url, cp_key, cp_models, cp_desc],
            outputs=cp_save_status,
        )

        def _refresh():
            choices, msg = _list_custom_providers()
            return gr.Dropdown(choices=choices, value=None), msg

        cp_refresh_btn.click(fn=_refresh, outputs=[cp_list, cp_list_status])
        tab.select(fn=_refresh, outputs=[cp_list, cp_list_status])

        cp_load_btn.click(
            fn=load_provider_detail,
            inputs=cp_list,
            outputs=[cp_name, cp_url, cp_key, cp_models, cp_desc],
        )

        cp_validate_btn.click(
            fn=validate_custom_provider, inputs=cp_list, outputs=cp_list_status
        )

        def _delete(sel):
            msg, choices, list_msg = delete_custom_provider(sel)
            return msg, gr.Dropdown(choices=choices, value=None), list_msg

        cp_delete_btn.click(
            fn=_delete,
            inputs=cp_list,
            outputs=[cp_save_status, cp_list, cp_list_status],
        )

    return tab
