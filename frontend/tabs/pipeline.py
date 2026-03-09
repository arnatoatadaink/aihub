"""Pipeline tab — form-based multi-step LLM pipeline builder.

Each step is configured via GUI controls (no JSON editing required).
Up to MAX_STEPS steps can be added / removed dynamically.
A collapsible JSON preview shows the generated definition for power users.
"""
from __future__ import annotations

import base64
import io
import json

import gradio as gr

from frontend.utils import BACKEND_URL, PROVIDER_MODEL_MAP, api_delete, api_get, api_post

MAX_STEPS = 5

_BUILTIN_PROVIDERS = list(PROVIDER_MODEL_MAP.keys())

_PRESETS: dict[str, dict] = {
    "翻訳→要約": {
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
    },
    "添削→箇条書き": {
        "name": "添削→箇条書きパイプライン",
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

_FIELDS_PER_STEP = 7  # name, provider, model, system_prompt, input_template, temperature, max_tokens


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_all_provider_ids() -> list[str]:
    """Built-in providers + registered custom provider IDs."""
    ids = list(_BUILTIN_PROVIDERS)
    result = api_get("/v1/custom_providers")
    for p in result.get("providers", []):
        ids.append(p["id"])
    return ids


def _build_definition(name: str, desc: str, count: int, *flat_fields) -> dict:
    """Construct a pipeline definition dict from flattened form field values."""
    steps = []
    for i in range(int(count)):
        b = i * _FIELDS_PER_STEP
        default_template = "{{input}}" if i == 0 else "{{prev_output}}"
        steps.append({
            "id": f"step{i + 1}",
            "name": flat_fields[b] or f"Step {i + 1}",
            "provider": flat_fields[b + 1] or "gemini",
            "model": flat_fields[b + 2] or "",
            "system_prompt": flat_fields[b + 3] or "",
            "input_template": flat_fields[b + 4] or default_template,
            "params": {
                "temperature": float(flat_fields[b + 5] or 0.7),
                "max_tokens": int(flat_fields[b + 6] or 2048),
            },
        })
    return {"name": name or "My Pipeline", "description": desc or "", "steps": steps}


def _definition_to_form(defn: dict) -> list:
    """Unpack a pipeline definition dict into the flat list of form values.

    Returns: [pipeline_name, pipeline_desc, step_count]
              + [7 fields × MAX_STEPS]
              + [gr.update(visible=...) × MAX_STEPS]
    """
    steps = (defn.get("steps") or [])[:MAX_STEPS]
    count = max(len(steps), 1)
    out: list = [defn.get("name", ""), defn.get("description", ""), count]
    for i in range(MAX_STEPS):
        default_template = "{{input}}" if i == 0 else "{{prev_output}}"
        if i < len(steps):
            s = steps[i]
            p = s.get("params", {})
            out.extend([
                s.get("name", f"Step {i + 1}"),
                s.get("provider", "gemini"),
                s.get("model", ""),
                s.get("system_prompt", ""),
                s.get("input_template") or default_template,
                p.get("temperature", 0.7),
                p.get("max_tokens", 2048),
            ])
        else:
            out.extend([f"Step {i + 1}", "gemini", "", "", default_template, 0.7, 2048])
    out.extend([gr.update(visible=i < count) for i in range(MAX_STEPS)])
    return out


def _format_step_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        icon = "✅" if not r.get("error") else "❌"
        lines.append(f"### {icon} Step {r['step_index'] + 1}: {r['step_name']}")
        lines.append(f"**Provider:** `{r['provider']}`　**Model:** `{r['model']}`")
        lines.append("")
        lines.append("**LLMへの入力:**")
        lines.append(f"```\n{r['input']}\n```")
        if r.get("error"):
            lines.append(f"**エラー:** {r['error']}")
        else:
            lines.append("**出力:**")
            out = r["output"]
            if out.startswith("data:image/") or (
                out.startswith("http") and any(ext in out for ext in [".png", ".jpg", ".webp", ".gif"])
            ):
                lines.append(f"![生成画像]({out})")
            else:
                lines.append(out)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_pipeline_tab() -> gr.Tab:
    with gr.Tab("Pipeline") as tab:
        gr.Markdown(
            "## Pipeline — 複数プロバイダーを連結して処理\n"
            "ステップを順番に実行し、前のステップの出力を次のステップへ渡せます。  \n"
            "`{{input}}` = 最初の入力　`{{prev_output}}` = 直前ステップの出力　"
            "`{{step:<id>}}` = 特定ステップの出力"
        )

        step_count = gr.State(value=1)

        # ── Pipeline metadata ──────────────────────────────────────────────
        with gr.Row():
            pipeline_name = gr.Textbox(
                label="パイプライン名", placeholder="My Pipeline", scale=3
            )
            pipeline_desc = gr.Textbox(label="説明", placeholder="何をするパイプラインか…", scale=4)

        with gr.Row():
            # ── Left: Step builder ─────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("### ステップ設定")
                    refresh_prov_btn = gr.Button("🔄 プロバイダー更新", size="sm")
                    prov_status = gr.Textbox(label="", interactive=False, scale=3)

                # ── Step blocks (pre-created, shown/hidden via gr.Group) ───
                step_groups: list[gr.Group] = []
                step_comps: list[list] = []  # [name, provider, model, system, template, temp, tokens]

                for i in range(MAX_STEPS):
                    with gr.Group(visible=(i == 0)) as grp:
                        with gr.Accordion(f"Step {i + 1}", open=True):
                            with gr.Row():
                                s_name = gr.Textbox(
                                    label="ステップ名",
                                    value=f"Step {i + 1}",
                                    scale=2,
                                )
                                s_provider = gr.Dropdown(
                                    label="プロバイダー",
                                    choices=_BUILTIN_PROVIDERS,
                                    value="gemini",
                                    allow_custom_value=True,
                                    scale=2,
                                )
                                s_model = gr.Textbox(
                                    label="モデル",
                                    placeholder="空欄=デフォルト (例: gemini-2.0-flash)",
                                    scale=3,
                                )
                            s_system = gr.Textbox(
                                label="システムプロンプト",
                                placeholder="あなたは…",
                                lines=3,
                            )
                            s_template = gr.Textbox(
                                label="入力テンプレート",
                                value="{{input}}" if i == 0 else "{{prev_output}}",
                                info="変数: {{input}} / {{prev_output}} / {{step:<id>}}",
                            )
                            with gr.Row():
                                s_temp = gr.Number(
                                    label="Temperature",
                                    value=0.7,
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.05,
                                )
                                s_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=2048,
                                    minimum=64,
                                    maximum=32768,
                                    step=64,
                                )
                    step_groups.append(grp)
                    step_comps.append([s_name, s_provider, s_model, s_system, s_template, s_temp, s_tokens])

                all_step_inputs: list = [c for cs in step_comps for c in cs]

                # ── Add / Remove step ──────────────────────────────────────
                with gr.Row():
                    add_step_btn = gr.Button("＋ ステップ追加", size="sm")
                    del_step_btn = gr.Button(
                        "－ 最後のステップを削除", size="sm", variant="secondary"
                    )

                gr.Markdown("---")

                # ── Presets & saved pipelines ─────────────────────────────
                with gr.Accordion("プリセット / 保存済みパイプライン", open=False):
                    gr.Markdown("**プリセット**")
                    with gr.Row():
                        preset_dd = gr.Dropdown(
                            label="テンプレート",
                            choices=list(_PRESETS.keys()),
                            value=None,
                            scale=3,
                        )
                        load_preset_btn = gr.Button("読み込む", size="sm", scale=1)

                    gr.Markdown("**保存済みパイプライン**")
                    with gr.Row():
                        saved_dd = gr.Dropdown(
                            label="一覧", choices=[], value=None, scale=4
                        )
                        refresh_saved_btn = gr.Button("🔄", size="sm", scale=1)
                    with gr.Row():
                        load_saved_btn = gr.Button("読み込む", size="sm")
                        delete_saved_btn = gr.Button("削除", size="sm", variant="stop")
                    saved_status = gr.Textbox(label="", interactive=False)

                gr.Markdown("---")

                # ── Save + JSON preview ───────────────────────────────────
                with gr.Row():
                    save_btn = gr.Button("💾 パイプラインを保存", variant="primary")
                    save_status = gr.Textbox(label="", interactive=False, scale=3)

                with gr.Accordion("生成された JSON (プレビュー)", open=False):
                    preview_btn = gr.Button("JSON を生成", size="sm")
                    json_preview = gr.Code(
                        label="",
                        language="json",
                        interactive=False,
                        lines=12,
                    )

            # ── Right: Run panel ───────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 実行")
                input_type_radio = gr.Radio(
                    choices=["テキストのみ", "テキスト＋画像"],
                    value="テキストのみ",
                    label="入力タイプ",
                )
                user_input_box = gr.Textbox(
                    label="入力テキスト",
                    placeholder="パイプラインへの最初の入力を入力...",
                    lines=4,
                )
                input_image_box = gr.Image(
                    label="画像入力",
                    type="pil",  # PIL.Image — stays in memory, no disk I/O
                    visible=False,
                )
                run_btn = gr.Button("▶ パイプライン実行", variant="primary", size="lg")
                run_status = gr.Textbox(label="ステータス", interactive=False)

                gr.Markdown("#### 最終出力")
                final_output = gr.Textbox(label="", interactive=False, lines=8)

                gr.Markdown("#### ステップ別詳細")
                step_results = gr.Markdown()

        # ── Common input sets ────────────────────────────────────────────────
        builder_inputs = [pipeline_name, pipeline_desc, step_count] + all_step_inputs
        form_outputs = [pipeline_name, pipeline_desc, step_count] + all_step_inputs + step_groups
        provider_dropdowns = [cs[1] for cs in step_comps]

        # ── Event handlers ───────────────────────────────────────────────────

        # Refresh provider dropdowns in all step forms
        def _refresh_providers():
            ids = _get_all_provider_ids()
            updates = [gr.update(choices=ids) for _ in range(MAX_STEPS)]
            return [f"✅ {len(ids)} 件のプロバイダーを取得"] + updates

        refresh_prov_btn.click(
            fn=_refresh_providers,
            outputs=[prov_status] + provider_dropdowns,
        )

        # Add / remove step
        def _add(count):
            new = min(int(count) + 1, MAX_STEPS)
            return [new] + [gr.update(visible=i < new) for i in range(MAX_STEPS)]

        def _remove(count):
            new = max(int(count) - 1, 1)
            return [new] + [gr.update(visible=i < new) for i in range(MAX_STEPS)]

        add_step_btn.click(fn=_add, inputs=step_count, outputs=[step_count] + step_groups)
        del_step_btn.click(fn=_remove, inputs=step_count, outputs=[step_count] + step_groups)

        # Load preset into form
        def _load_preset(name):
            return _definition_to_form(_PRESETS.get(name, list(_PRESETS.values())[0]))

        load_preset_btn.click(fn=_load_preset, inputs=preset_dd, outputs=form_outputs)

        # Saved pipeline list management
        def _refresh_saved():
            r = api_get("/v1/pipelines")
            choices = [f"{p['id']} — {p['name']}" for p in r.get("pipelines", [])]
            msg = f"{len(choices)} 件の保存済みパイプライン" if choices else "保存済みパイプラインなし"
            return gr.update(choices=choices, value=None), msg

        def _load_saved(sel):
            if not sel:
                return _definition_to_form({})
            pid = sel.split(" — ")[0].strip()
            r = api_get(f"/v1/pipelines/{pid}")
            if "error" in r:
                return _definition_to_form({})
            return _definition_to_form(r.get("pipeline", {}))

        def _delete_saved(sel):
            if not sel:
                return "パイプラインを選択してください", gr.update(), ""
            pid = sel.split(" — ")[0].strip()
            api_delete(f"/v1/pipelines/{pid}")
            dd_update, msg = _refresh_saved()
            return f"削除しました: {pid}", dd_update, msg

        refresh_saved_btn.click(fn=_refresh_saved, outputs=[saved_dd, saved_status])
        tab.select(fn=_refresh_saved, outputs=[saved_dd, saved_status])
        load_saved_btn.click(fn=_load_saved, inputs=saved_dd, outputs=form_outputs)
        delete_saved_btn.click(
            fn=_delete_saved, inputs=saved_dd, outputs=[saved_status, saved_dd, saved_status]
        )

        # Save pipeline
        def _save(*all_args):
            name, desc, count = all_args[0], all_args[1], all_args[2]
            flat = all_args[3:]
            defn = _build_definition(name, desc, count, *flat)
            r = api_post("/v1/pipelines", defn)
            if "error" in r:
                return f"保存失敗: {r['error']}"
            pid = r.get("pipeline", {}).get("id", "?")
            return f"✅ 保存しました (ID: {pid})"

        save_btn.click(fn=_save, inputs=builder_inputs, outputs=save_status)

        # JSON preview
        def _preview(*all_args):
            name, desc, count = all_args[0], all_args[1], all_args[2]
            flat = all_args[3:]
            defn = _build_definition(name, desc, count, *flat)
            return json.dumps(defn, indent=2, ensure_ascii=False)

        preview_btn.click(fn=_preview, inputs=builder_inputs, outputs=json_preview)

        # Show/hide image input based on input type selection
        input_type_radio.change(
            fn=lambda t: gr.update(visible=(t == "テキスト＋画像")),
            inputs=input_type_radio,
            outputs=input_image_box,
        )

        # Run pipeline
        def _run(*all_args):
            name, desc, count = all_args[0], all_args[1], all_args[2]
            flat = all_args[3:-3]
            user_input_text = all_args[-3]
            user_input_image = all_args[-2]
            input_type = all_args[-1]

            if not str(user_input_text).strip() and not user_input_image:
                return "", "", "入力テキストまたは画像を入力してください"

            defn = _build_definition(name, desc, count, *flat)
            payload: dict = {"definition": defn}

            if input_type == "テキスト＋画像" and user_input_image:
                # user_input_image is PIL.Image — encode to PNG in memory, no disk I/O
                try:
                    buf = io.BytesIO()
                    user_input_image.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    data_uri = f"data:image/png;base64,{b64}"
                except Exception as e:
                    return "", "", f"画像のエンコードに失敗しました: {e}"
                payload["input_parts"] = [
                    {"type": "text", "text": str(user_input_text).strip()},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
                payload["input"] = ""
            else:
                payload["input"] = str(user_input_text).strip()

            r = api_post("/v1/pipelines/run", payload, timeout=180)
            if "error" in r:
                return "", "", f"実行エラー: {r['error']}"
            results = r.get("results", [])
            final = r.get("final_output", "")
            return _format_step_results(results), final, f"✅ 完了 ({len(results)} ステップ)"

        run_btn.click(
            fn=_run,
            inputs=builder_inputs + [user_input_box, input_image_box, input_type_radio],
            outputs=[step_results, final_output, run_status],
        )

    return tab
