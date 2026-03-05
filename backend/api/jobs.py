"""Training job management endpoints.

Endpoints:
  POST /v1/training/jobs          — enqueue a training job
  GET  /v1/training/jobs          — list recent jobs
  GET  /v1/training/jobs/{job_id} — get job status
  POST /v1/training/vertex        — submit to Vertex AI
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/training", tags=["training"])


# ======================================================================
# Request / Response schemas
# ======================================================================

class GRPOJobRequest(BaseModel):
    model_name_or_path: str = "google/gemma-2-2b-it"
    output_dir: str = "./outputs/grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_seq_length: int = 512
    max_new_tokens: int = 256
    num_generations: int = 4
    beta: float = 0.01
    temperature: float = 0.9
    dataset_path: str = ""
    max_samples: int = -1
    reward_model: str = "gemini"
    use_peft: bool = True
    lora_rank: int = 8
    seed: int = 42


class VertexJobRequest(BaseModel):
    project: str = ""
    location: str = "us-central1"
    staging_bucket: str
    container_uri: str
    display_name: str = "aihub-grpo-training"
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 1
    args: list[str] = Field(default_factory=list)
    max_cost_usd: float = 50.0
    service_account: str = ""


# ======================================================================
# Helpers
# ======================================================================

def _get_celery():
    try:
        from backend.worker import celery_app
        return celery_app
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Celery worker unavailable: {e}. Make sure Redis and the worker are running.",
        )


def _task_info(task_id: str) -> dict:
    from celery.result import AsyncResult
    celery = _get_celery()
    res = AsyncResult(task_id, app=celery)
    info: dict = {
        "job_id": task_id,
        "state": res.state,
    }
    if res.state == "PROGRESS":
        info["message"] = (res.info or {}).get("message", "")
    elif res.state == "SUCCESS":
        info["result"] = res.result
    elif res.state == "FAILURE":
        info["error"] = str(res.result)
    return info


# ======================================================================
# Endpoints
# ======================================================================

@router.post("/jobs")
async def create_grpo_job(req: GRPOJobRequest):
    """Enqueue a GRPO training job via Celery."""
    celery = _get_celery()
    task = celery.send_task("training.grpo", kwargs={"config_dict": req.model_dump()})
    return {"job_id": task.id, "status": "queued"}


@router.get("/jobs")
async def list_jobs():
    """Return recent job IDs from the Celery inspect API."""
    try:
        from backend.worker import celery_app
        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}

        jobs = []
        for worker, tasks in {**active, **reserved}.items():
            for t in tasks:
                jobs.append({"job_id": t["id"], "name": t["name"], "worker": worker})
        return {"jobs": jobs}
    except Exception as e:
        return {"jobs": [], "warning": str(e)}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Return status of a specific Celery task."""
    _get_celery()  # ensure celery is reachable
    return _task_info(job_id)


@router.post("/vertex")
async def submit_vertex_job(req: VertexJobRequest):
    """Submit a Vertex AI Custom Training job (async via Celery)."""
    celery = _get_celery()
    task = celery.send_task(
        "training.vertex_submit",
        kwargs={"config_dict": req.model_dump()},
    )
    return {"job_id": task.id, "status": "queued"}
