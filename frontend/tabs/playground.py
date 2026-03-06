"""Prompt Playground tab.

Supports built-in providers (gemini / openai / anthropic) and any registered
custom providers.  Use the "Refresh Providers" button to load newly registered
custom providers into the Provider dropdown.
"""
from __future__ import annotations

import json
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
# Chat streaming
# ---------------------------------------------------------------------------

def chat(
    message: str,
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
                chatbot = gr.Chatbot(height=500, label="Chat")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="メッセージを入力...",
                        show_label=False,
                        scale=9,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

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

        # Chat submit / send
        def user_submit(message, history):
            return "", history + [[message, None]]

        def bot_respond(history, provider, model, system_prompt, temperature, max_tokens, top_p):
            user_msg = history[-1][0]
            history[-1][1] = ""
            for partial in chat(
                user_msg, history[:-1], provider, model,
                system_prompt, temperature, max_tokens, top_p,
            ):
                history[-1][1] = partial
                yield history

        msg_input.submit(
            fn=user_submit,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            fn=bot_respond,
            inputs=[chatbot, provider, model, system_prompt, temperature, max_tokens, top_p],
            outputs=chatbot,
        )

        send_btn.click(
            fn=user_submit,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            fn=bot_respond,
            inputs=[chatbot, provider, model, system_prompt, temperature, max_tokens, top_p],
            outputs=chatbot,
        )

        clear_btn.click(fn=lambda: [], outputs=chatbot)

    return tab
