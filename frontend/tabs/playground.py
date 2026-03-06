"""Prompt Playground tab.

Supports built-in providers (gemini / openai / anthropic) and any registered
custom providers.  Use the "Refresh Providers" button to load newly registered
custom providers into the Provider dropdown.

Multimodal support:
- Vision: attach an image (jpeg/png/webp/gif) to be sent alongside the message.
- STT: record audio or upload a file; transcription is fetched from the backend
  and pasted into the text input before sending.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from typing import Generator

import gradio as gr
import httpx

from frontend.utils import BACKEND_URL, PROVIDER_MODEL_MAP, api_get


# ---------------------------------------------------------------------------
# Provider / model helpers
# ---------------------------------------------------------------------------

def _fetch_full_provider_map() -> dict[str, list[str]]:
    """Return built-in + custom provider → model lists."""
    full = dict(PROVIDER_MODEL_MAP)
    result = api_get("/v1/custom_providers")
    for p in result.get("providers", []):
        pid = p["id"]
        full[pid] = p.get("models") or ["default"]
    return full


def get_models_for_provider(provider: str, provider_map: dict) -> gr.Dropdown:
    models = provider_map.get(provider, [])
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else None,
        allow_custom_value=True,
    )


# ---------------------------------------------------------------------------
# Vision helper
# ---------------------------------------------------------------------------

# Providers that support vision input
VISION_PROVIDERS = {"gemini", "openai", "anthropic"}

def _image_to_data_url(filepath: str) -> str | None:
    """Read an image file and return a data URL (base64-encoded)."""
    if not filepath:
        return None
    mime, _ = mimetypes.guess_type(filepath)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _build_user_content(text: str, image_path: str | None) -> str | list:
    """Return str content or multipart list if an image is attached."""
    if not image_path:
        return text
    data_url = _image_to_data_url(image_path)
    if not data_url:
        return text
    parts: list = []
    if text.strip():
        parts.append({"type": "text", "text": text})
    parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return parts


# ---------------------------------------------------------------------------
# STT helper
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """Call backend STT endpoint and return transcription text."""
    if not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as f:
            resp = httpx.post(
                f"{BACKEND_URL}/v1/audio/transcriptions",
                files={"file": (audio_path, f)},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
    except Exception as e:
        return f"[STT Error: {e}]"


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------

def chat(
    message: str | list,
    history: list,
    provider: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> Generator[str, None, None]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True,
        "provider": provider,
    }

    accumulated = ""
    try:
        with httpx.stream(
            "POST",
            f"{BACKEND_URL}/v1/chat/completions",
            json=payload,
            timeout=60,
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    accumulated += delta
                    yield accumulated
                except Exception:
                    continue
    except Exception as e:
        yield f"Error: {e}"


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_playground_tab() -> gr.Tab:
    with gr.Tab("Playground") as tab:
        gr.Markdown("## Prompt Playground")

        # State holds the full provider→models map (updated on refresh)
        provider_map_state = gr.State(value=dict(PROVIDER_MODEL_MAP))

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460, label="Chat")

                # ── Text input row ─────────────────────────────────────
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="メッセージを入力...",
                        show_label=False,
                        scale=9,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # ── Vision: image upload (accordion) ──────────────────
                with gr.Accordion("画像を添付（Vision）", open=False):
                    image_input = gr.Image(
                        label="画像ファイル（jpeg / png / webp / gif）",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )
                    gr.Markdown(
                        "_Vision対応モデル: Gemini全モデル / gpt-4o系 / Claude Sonnet・Opus_"
                    )

                # ── STT: audio input (accordion) ───────────────────────
                with gr.Accordion("音声入力（STT）", open=False):
                    audio_input = gr.Audio(
                        label="マイク録音 または 音声ファイルをアップロード",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )
                    with gr.Row():
                        transcribe_btn = gr.Button("文字起こし", scale=2)
                        stt_status = gr.Textbox(
                            label="", placeholder="ステータス", interactive=False, scale=8
                        )

            with gr.Column(scale=1):
                gr.Markdown("### Parameters")

                with gr.Row():
                    provider = gr.Dropdown(
                        label="Provider",
                        choices=list(PROVIDER_MODEL_MAP.keys()),
                        value="gemini",
                        allow_custom_value=True,
                        scale=4,
                    )
                    refresh_prov_btn = gr.Button("🔄", size="sm", scale=1)

                model = gr.Dropdown(
                    label="Model",
                    choices=PROVIDER_MODEL_MAP["gemini"],
                    value="gemini-2.0-flash",
                    allow_custom_value=True,
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful assistant.",
                    lines=4,
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.05
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=64, maximum=8192, value=2048, step=64
                )
                top_p = gr.Slider(
                    label="Top P", minimum=0.0, maximum=1.0, value=1.0, step=0.01
                )
                clear_btn = gr.Button("Clear Chat")

        # ── Event handlers ───────────────────────────────────────────────

        # Refresh providers (load custom providers from backend)
        def _refresh_providers():
            full_map = _fetch_full_provider_map()
            provider_ids = list(full_map.keys())
            return (
                full_map,
                gr.update(choices=provider_ids),
            )

        refresh_prov_btn.click(
            fn=_refresh_providers,
            outputs=[provider_map_state, provider],
        )

        # Also refresh on tab select so custom providers are visible immediately
        tab.select(
            fn=_refresh_providers,
            outputs=[provider_map_state, provider],
        )

        # Update model list when provider changes
        def _on_provider_change(prov, pmap):
            return get_models_for_provider(prov, pmap)

        provider.change(
            fn=_on_provider_change,
            inputs=[provider, provider_map_state],
            outputs=model,
        )

        # STT: transcribe audio → paste into text input
        def _do_transcribe(audio_path: str):
            if not audio_path:
                return gr.update(), "音声ファイルを選択してください"
            text = transcribe_audio(audio_path)
            if text.startswith("[STT Error"):
                return gr.update(), text
            return gr.update(value=text), f"文字起こし完了（{len(text)}文字）"

        transcribe_btn.click(
            fn=_do_transcribe,
            inputs=[audio_input],
            outputs=[msg_input, stt_status],
        )

        # Chat submit / send  (now passes image_path through)
        def user_submit(message: str, image_path: str | None, history: list):
            content = _build_user_content(message, image_path)
            # Display text in chatbot (image preview is shown as "[画像添付]" label)
            display_text = message if isinstance(content, str) else f"[画像添付] {message}".strip()
            return "", None, history + [[display_text, None]], content

        # pending_content holds the actual content (str or list) to be sent
        pending_content = gr.State(value=None)

        def bot_respond(
            history, pending, prov, mod, sys_prompt, temp, max_tok, tp
        ):
            content = pending if pending is not None else (history[-1][0] if history else "")
            history[-1][1] = ""
            for partial in chat(
                content, history[:-1], prov, mod, sys_prompt, temp, max_tok, tp
            ):
                history[-1][1] = partial
                yield history

        msg_input.submit(
            fn=user_submit,
            inputs=[msg_input, image_input, chatbot],
            outputs=[msg_input, image_input, chatbot, pending_content],
        ).then(
            fn=bot_respond,
            inputs=[chatbot, pending_content, provider, model, system_prompt, temperature, max_tokens, top_p],
            outputs=chatbot,
        )

        send_btn.click(
            fn=user_submit,
            inputs=[msg_input, image_input, chatbot],
            outputs=[msg_input, image_input, chatbot, pending_content],
        ).then(
            fn=bot_respond,
            inputs=[chatbot, pending_content, provider, model, system_prompt, temperature, max_tokens, top_p],
            outputs=chatbot,
        )

        clear_btn.click(fn=lambda: [], outputs=chatbot)

    return tab
