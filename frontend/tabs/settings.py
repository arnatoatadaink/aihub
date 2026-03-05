import os

import gradio as gr
import httpx


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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


def build_settings_tab() -> gr.Tab:
    with gr.Tab("Settings") as tab:
        gr.Markdown("## API Key Settings")
        gr.Markdown(
            "API keys are stored as environment variables on the backend only. "
            "Keys entered here are applied at runtime and are not persisted to disk."
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

        gr.Markdown("## Backend Status")
        check_btn = gr.Button("Check Backend Health")
        health_msg = gr.Textbox(label="Health", interactive=False)

        save_btn.click(
            fn=save_settings,
            inputs=[gemini_key, openai_key, anthropic_key],
            outputs=status_msg,
        )
        check_btn.click(fn=check_backend_health, inputs=[], outputs=health_msg)

    return tab
