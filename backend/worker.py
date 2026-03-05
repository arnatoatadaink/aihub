"""Celery application and training task definitions.

All long-running training operations run through this worker so that
the Gradio event loop is never blocked.

Start with:
    celery -A backend.worker worker --loglevel=info
"""
from __future__ import annotations

import logging
import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "aihub",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,   # 24 h
    task_track_started=True,
    worker_prefetch_multiplier=1,   # one task at a time (GPU jobs)
)


# ======================================================================
# Tasks
# ======================================================================

@celery_app.task(bind=True, name="training.grpo")
def run_grpo_task(self, config_dict: dict) -> dict:
    """Run a GRPO training job.

    Args:
        config_dict: Serialised GRPOConfig fields.

    Returns:
        Serialised GRPOTrainResult fields.
    """
    from backend.training.grpo import GRPOConfig, GRPOTrainer

    def progress(msg: str):
        self.update_state(state="PROGRESS", meta={"message": msg})
        logger.info("[grpo] %s", msg)

    config = GRPOConfig(**config_dict)
    trainer = GRPOTrainer(config, progress_cb=progress)
    result = trainer.train()
    return {
        "status": result.status,
        "epochs_done": result.epochs_done,
        "steps_done": result.steps_done,
        "final_loss": result.final_loss,
        "output_dir": result.output_dir,
        "error": result.error,
    }


@celery_app.task(bind=True, name="training.vertex_submit")
def submit_vertex_job_task(self, config_dict: dict) -> dict:
    """Submit a Vertex AI Custom Training job.

    Args:
        config_dict: Serialised VertexJobConfig fields.

    Returns:
        Serialised VertexJobResult fields.
    """
    from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter

    self.update_state(state="PROGRESS", meta={"message": "Submitting Vertex AI job…"})
    config = VertexJobConfig(**config_dict)
    submitter = VertexJobSubmitter(config)
    result = submitter.submit()
    return {
        "status": result.status,
        "job_name": result.job_name,
        "job_id": result.job_id,
        "console_url": result.console_url,
        "error": result.error,
    }
