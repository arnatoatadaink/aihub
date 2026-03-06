import os

from dotenv import load_dotenv
import gradio as gr

from frontend.tabs.playground import build_playground_tab
from frontend.tabs.rag import build_rag_tab
from frontend.tabs.evals import build_evals_tab
from frontend.tabs.training import build_training_tab
from frontend.tabs.media import build_media_tab
from frontend.tabs.pipeline import build_pipeline_tab
from frontend.tabs.settings import build_settings_tab
from frontend.version import APP_VERSION

load_dotenv()


def build_app() -> gr.Blocks:
    with gr.Blocks(title=f"AI Hub v{APP_VERSION}", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            f"""
            # AI Hub &nbsp; <sub>`v{APP_VERSION}`</sub>
            Unified interface for Google Gemini, OpenAI GPT-4o, Anthropic Claude, and more.
            """
        )
        build_playground_tab()
        build_rag_tab()
        build_evals_tab()
        build_training_tab()
        build_media_tab()
        build_pipeline_tab()
        build_settings_tab()

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
