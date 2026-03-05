"""Eval Dashboard tab — run prompt evaluations across multiple providers/models."""
import json
import time
from typing import Generator

import gradio as gr
import httpx

from frontend.utils import BACKEND_URL, PROVIDER_MODEL_MAP


def _chat(provider: str, model: str, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, float]:
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "provider": provider,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = httpx.post(f"{BACKEND_URL}/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        return content, elapsed
    except Exception as e:
        return f"Error: {e}", time.time() - t0


def run_eval(
    system_prompt: str,
    test_cases_json: str,
    eval_models: list[str],
    temperature: float,
    max_tokens: int,
) -> Generator[tuple, None, None]:
    """Run each test case against selected models and yield results table."""
    try:
        test_cases = json.loads(test_cases_json)
    except Exception as e:
        yield (
            [[f"JSON parse error: {e}", "", "", ""]],
            "パースエラー",
        )
        return

    if not test_cases:
        yield ([[]], "テストケースなし")
        return

    rows = []
    for tc_idx, tc in enumerate(test_cases):
        user_input = tc.get("input", "")
        expected = tc.get("expected", "")
        for model_str in eval_models:
            parts = model_str.split("/", 1)
            if len(parts) != 2:
                continue
            provider, model = parts
            output, elapsed = _chat(
                provider, model, system_prompt, user_input, temperature, max_tokens
            )
            rows.append([
                f"TC{tc_idx + 1}",
                user_input[:80],
                expected[:80],
                model_str,
                output[:200],
                f"{elapsed:.2f}s",
            ])
            yield (rows, f"実行中… TC{tc_idx + 1}/{len(test_cases)} — {model_str}")

    yield (rows, f"完了: {len(rows)} 件の評価")


def build_evals_tab() -> gr.Tab:
    with gr.Tab("Evals") as tab:
        gr.Markdown("## Eval Dashboard")
        gr.Markdown(
            "複数モデルに同じプロンプトを投入し、出力を横断比較します。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful assistant.",
                    lines=3,
                )
                test_cases = gr.Textbox(
                    label='Test Cases (JSON) — [{"input": "...", "expected": "..."}, ...]',
                    value=json.dumps(
                        [
                            {"input": "What is 2+2?", "expected": "4"},
                            {"input": "Capital of Japan?", "expected": "Tokyo"},
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    lines=10,
                )

            with gr.Column(scale=1):
                gr.Markdown("### モデル選択")
                model_checkboxes = gr.CheckboxGroup(
                    label="評価するモデル",
                    choices=[
                        f"{p}/{m}"
                        for p, models in PROVIDER_MODEL_MAP.items()
                        for m in models
                    ],
                    value=["gemini/gemini-2.0-flash"],
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.05
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=64, maximum=4096, value=512, step=64
                )
                run_btn = gr.Button("評価実行", variant="primary")

        status_msg = gr.Textbox(label="ステータス", interactive=False)
        results_table = gr.Dataframe(
            headers=["TC", "Input", "Expected", "Model", "Output", "Latency"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="評価結果",
            wrap=True,
        )

        run_btn.click(
            fn=run_eval,
            inputs=[system_prompt, test_cases, model_checkboxes, temperature, max_tokens],
            outputs=[results_table, status_msg],
        )

    return tab
