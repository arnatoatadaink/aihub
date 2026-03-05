"""Training GUI tab — GRPO local training and Vertex AI job submission."""
from __future__ import annotations

import json
import os
import time

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ======================================================================
# Backend helpers
# ======================================================================

def _post(path: str, payload: dict) -> dict:
    try:
        resp = httpx.post(f"{BACKEND_URL}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _get(path: str) -> dict:
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Action functions
# ======================================================================

def submit_grpo_job(
    model_path: str,
    output_dir: str,
    dataset_path: str,
    max_samples: int,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    num_generations: int,
    beta: float,
    temperature: float,
    reward_model: str,
    use_peft: bool,
    lora_rank: int,
) -> tuple[str, str]:
    payload = {
        "model_name_or_path": model_path,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "max_samples": int(max_samples),
        "num_train_epochs": int(num_epochs),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "learning_rate": learning_rate,
        "num_generations": int(num_generations),
        "beta": beta,
        "temperature": temperature,
        "reward_model": reward_model,
        "use_peft": use_peft,
        "lora_rank": int(lora_rank),
    }
    result = _post("/v1/training/jobs", payload)
    if "error" in result:
        return f"Error: {result['error']}", ""
    job_id = result.get("job_id", "")
    return f"ジョブをキューに追加しました (ID: {job_id})", job_id


def submit_vertex_job(
    project: str,
    location: str,
    staging_bucket: str,
    container_uri: str,
    display_name: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    max_cost_usd: float,
    extra_args: str,
) -> str:
    args = [a.strip() for a in extra_args.split("\n") if a.strip()]
    payload = {
        "project": project,
        "location": location,
        "staging_bucket": staging_bucket,
        "container_uri": container_uri,
        "display_name": display_name,
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": int(accelerator_count),
        "max_cost_usd": max_cost_usd,
        "args": args,
    }
    result = _post("/v1/training/vertex", payload)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Vertex AI ジョブをキューに追加しました (Celery ID: {result.get('job_id')})"


def poll_job_status(job_id: str) -> str:
    job_id = job_id.strip()
    if not job_id:
        return "Job IDを入力してください"
    result = _get(f"/v1/training/jobs/{job_id}")
    if "error" in result:
        return f"Error: {result['error']}"
    return json.dumps(result, indent=2, ensure_ascii=False)


def list_active_jobs() -> str:
    result = _get("/v1/training/jobs")
    if "error" in result:
        return f"Error: {result['error']}"
    jobs = result.get("jobs", [])
    if not jobs:
        return "アクティブなジョブなし"
    lines = [f"- `{j['job_id'][:8]}…` [{j['name']}] worker: {j.get('worker', '?')}" for j in jobs]
    return "\n".join(lines)


# ======================================================================
# Tab builder
# ======================================================================

def build_training_tab() -> gr.Tab:
    with gr.Tab("Training") as tab:
        gr.Markdown("## Training GUI")
        gr.Markdown(
            "GRPOトレーニングをローカルまたはVertex AI上で実行します。  "
            "ジョブはCelery + Redisキュー経由で非同期実行されます。"
        )

        with gr.Tabs():
            # ----------------------------------------------------------
            # Local GRPO
            # ----------------------------------------------------------
            with gr.Tab("GRPO (Local)"):
                gr.Markdown("### ローカル GRPO トレーニング")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**モデル・データ設定**")
                        model_path = gr.Textbox(
                            label="Model (HF Hub ID or local path)",
                            value="google/gemma-2-2b-it",
                        )
                        output_dir = gr.Textbox(
                            label="Output Directory", value="./outputs/grpo"
                        )
                        dataset_path = gr.Textbox(
                            label="Dataset Path (JSONL)",
                            placeholder="/path/to/data.jsonl  ※空白で合成デモデータ使用",
                        )
                        max_samples = gr.Number(
                            label="Max Samples (-1 = all)", value=-1, precision=0
                        )

                    with gr.Column():
                        gr.Markdown("**ハイパーパラメータ**")
                        num_epochs = gr.Slider(
                            label="Epochs", minimum=1, maximum=10, value=1, step=1
                        )
                        batch_size = gr.Slider(
                            label="Batch Size / device", minimum=1, maximum=32, value=4, step=1
                        )
                        grad_accum = gr.Slider(
                            label="Gradient Accumulation", minimum=1, maximum=32, value=4, step=1
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate", value=5e-6, precision=10
                        )
                        num_generations = gr.Slider(
                            label="Num Generations (G)", minimum=2, maximum=16, value=4, step=1
                        )
                        beta = gr.Slider(
                            label="Beta (KL coeff)", minimum=0.0, maximum=0.5, value=0.01, step=0.005
                        )
                        temperature = gr.Slider(
                            label="Sampling Temperature", minimum=0.1, maximum=2.0, value=0.9, step=0.05
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**報酬モデル設定**")
                        reward_model = gr.Radio(
                            label="Reward Model",
                            choices=["gemini", "rule_based"],
                            value="gemini",
                        )

                    with gr.Column():
                        gr.Markdown("**LoRA設定**")
                        use_peft = gr.Checkbox(label="Use LoRA (PEFT)", value=True)
                        lora_rank = gr.Slider(
                            label="LoRA Rank", minimum=4, maximum=64, value=8, step=4
                        )

                grpo_submit_btn = gr.Button("トレーニング開始", variant="primary")
                grpo_status = gr.Textbox(label="ステータス", interactive=False)
                grpo_job_id = gr.Textbox(label="Job ID", interactive=False)

                grpo_submit_btn.click(
                    fn=submit_grpo_job,
                    inputs=[
                        model_path, output_dir, dataset_path, max_samples,
                        num_epochs, batch_size, grad_accum, learning_rate,
                        num_generations, beta, temperature,
                        reward_model, use_peft, lora_rank,
                    ],
                    outputs=[grpo_status, grpo_job_id],
                )

            # ----------------------------------------------------------
            # Vertex AI
            # ----------------------------------------------------------
            with gr.Tab("Vertex AI"):
                gr.Markdown("### Vertex AI カスタムトレーニングジョブ")
                gr.Markdown(
                    "> **コスト上限必須**: Max Cost (USD) を必ず設定してください。"
                    "Vertex AIには直接の予算上限機能がないため、GCPの"
                    "[請求アラート](https://console.cloud.google.com/billing)も合わせて設定してください。"
                )
                with gr.Row():
                    with gr.Column():
                        vx_project = gr.Textbox(
                            label="GCP Project ID",
                            placeholder=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                        )
                        vx_location = gr.Textbox(label="Region", value="us-central1")
                        vx_bucket = gr.Textbox(
                            label="Staging Bucket (gs://...)",
                            placeholder="gs://your-bucket/staging",
                        )
                        vx_container = gr.Textbox(
                            label="Container URI",
                            placeholder="gcr.io/your-project/aihub-training:latest",
                        )
                        vx_display_name = gr.Textbox(
                            label="Job Display Name", value="aihub-grpo-training"
                        )

                    with gr.Column():
                        vx_machine = gr.Dropdown(
                            label="Machine Type",
                            choices=["n1-standard-8", "n1-standard-16", "a2-highgpu-1g"],
                            value="n1-standard-8",
                        )
                        vx_accel = gr.Dropdown(
                            label="Accelerator",
                            choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_A100_80GB", ""],
                            value="NVIDIA_TESLA_T4",
                        )
                        vx_accel_count = gr.Slider(
                            label="Accelerator Count", minimum=0, maximum=8, value=1, step=1
                        )
                        vx_max_cost = gr.Number(
                            label="Max Cost (USD) ※必須", value=50.0, precision=2
                        )
                        vx_args = gr.Textbox(
                            label="Extra Args (1行1引数)",
                            placeholder="--model_name google/gemma-2-2b-it\n--num_epochs 3",
                            lines=4,
                        )

                vx_submit_btn = gr.Button("Vertex AI ジョブ送信", variant="primary")
                vx_status = gr.Textbox(label="ステータス", interactive=False)

                vx_submit_btn.click(
                    fn=submit_vertex_job,
                    inputs=[
                        vx_project, vx_location, vx_bucket, vx_container,
                        vx_display_name, vx_machine, vx_accel, vx_accel_count,
                        vx_max_cost, vx_args,
                    ],
                    outputs=vx_status,
                )

            # ----------------------------------------------------------
            # Job Monitor
            # ----------------------------------------------------------
            with gr.Tab("Job Monitor"):
                gr.Markdown("### ジョブモニター")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**ジョブステータス確認**")
                        poll_job_id = gr.Textbox(
                            label="Job ID", placeholder="Celery Task UUID"
                        )
                        poll_btn = gr.Button("ステータス確認")
                        poll_result = gr.Code(language="json", label="結果")

                    with gr.Column():
                        gr.Markdown("**アクティブジョブ一覧**")
                        list_btn = gr.Button("更新")
                        active_jobs = gr.Markdown()

                poll_btn.click(fn=poll_job_status, inputs=[poll_job_id], outputs=poll_result)
                list_btn.click(fn=list_active_jobs, inputs=[], outputs=active_jobs)
                tab.select(fn=list_active_jobs, inputs=[], outputs=active_jobs)

    return tab
