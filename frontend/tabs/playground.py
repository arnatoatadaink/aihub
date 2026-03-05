import json
from typing import Generator

import gradio as gr
import httpx

from frontend.utils import BACKEND_URL, PROVIDER_MODEL_MAP


def get_models_for_provider(provider: str) -> gr.Dropdown:
    models = PROVIDER_MODEL_MAP.get(provider, [])
    return gr.Dropdown(choices=models, value=models[0] if models else None)


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


def build_playground_tab() -> gr.Tab:
    with gr.Tab("Playground") as tab:
        gr.Markdown("## Prompt Playground")
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
                provider = gr.Dropdown(
                    label="Provider",
                    choices=list(PROVIDER_MODEL_MAP.keys()),
                    value="gemini",
                )
                model = gr.Dropdown(
                    label="Model",
                    choices=PROVIDER_MODEL_MAP["gemini"],
                    value="gemini-2.0-flash",
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

        provider.change(fn=get_models_for_provider, inputs=provider, outputs=model)

        def user_submit(message, history):
            return "", history + [[message, None]]

        def bot_respond(history, provider, model, system_prompt, temperature, max_tokens, top_p):
            user_msg = history[-1][0]
            history[-1][1] = ""
            for partial in chat(
                user_msg, history[:-1], provider, model,
                system_prompt, temperature, max_tokens, top_p
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
