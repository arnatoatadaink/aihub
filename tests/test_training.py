"""Tests for backend/training/ modules (no torch/peft required — lazy imports)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# GRPOConfig
# ---------------------------------------------------------------------------

class TestGRPOConfig:
    def test_defaults(self):
        from backend.training.grpo import GRPOConfig
        cfg = GRPOConfig()
        assert cfg.model_name_or_path == "google/gemma-2-2b-it"
        assert cfg.num_train_epochs == 1
        assert cfg.num_generations == 4
        assert cfg.beta == pytest.approx(0.01)
        assert cfg.reward_model == "gemini"
        assert cfg.use_peft is True
        assert cfg.lora_rank == 8

    def test_custom_values(self):
        from backend.training.grpo import GRPOConfig
        cfg = GRPOConfig(
            model_name_or_path="mistralai/Mistral-7B-v0.1",
            num_train_epochs=3,
            reward_model="rule_based",
            use_peft=False,
        )
        assert cfg.model_name_or_path == "mistralai/Mistral-7B-v0.1"
        assert cfg.num_train_epochs == 3
        assert cfg.reward_model == "rule_based"
        assert cfg.use_peft is False


class TestGRPOTrainResult:
    def test_default_status(self):
        from backend.training.grpo import GRPOTrainResult
        r = GRPOTrainResult(status="failed")
        assert r.status == "failed"
        assert r.epochs_done == 0
        assert r.error == ""
        assert r.logs == []


class TestGRPOTrainerInit:
    def test_instantiation(self):
        from backend.training.grpo import GRPOConfig, GRPOTrainer
        cfg = GRPOConfig()
        trainer = GRPOTrainer(cfg)
        assert trainer.config is cfg

    def test_custom_progress_callback(self):
        from backend.training.grpo import GRPOConfig, GRPOTrainer
        messages = []
        trainer = GRPOTrainer(GRPOConfig(), progress_cb=messages.append)
        trainer.progress_cb("hello")
        assert messages == ["hello"]

    def test_train_fails_gracefully_without_deps(self):
        """Training should return a failed result when trl/torch are not installed."""
        from backend.training.grpo import GRPOConfig, GRPOTrainer

        cfg = GRPOConfig(dataset_path="")
        trainer = GRPOTrainer(cfg)

        with patch("backend.training.grpo.GRPOTrainer._load_dataset") as mock_ds:
            mock_ds.side_effect = ImportError("No module named 'trl'")
            result = trainer.train()

        assert result.status == "failed"
        assert "trl" in result.error


# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------

class TestLoRAConfig:
    def test_defaults(self):
        from backend.training.lora import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == pytest.approx(0.05)
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"
        assert cfg.target_modules is None

    def test_apply_lora_raises_without_peft(self):
        from backend.training.lora import apply_lora
        mock_model = MagicMock()
        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises(ImportError, match="peft"):
                apply_lora(mock_model)

    def test_merge_and_save_raises_on_failure(self, tmp_path):
        from backend.training.lora import merge_and_save
        mock_model = MagicMock()
        mock_model.merge_and_unload.side_effect = RuntimeError("merge failed")
        with pytest.raises(RuntimeError, match="merge failed"):
            merge_and_save(mock_model, str(tmp_path))


# ---------------------------------------------------------------------------
# VertexJobConfig & VertexJobSubmitter
# ---------------------------------------------------------------------------

class TestVertexJobConfig:
    def test_defaults(self):
        from backend.training.vertex_job import VertexJobConfig
        cfg = VertexJobConfig()
        assert cfg.location == "us-central1"
        assert cfg.machine_type == "n1-standard-8"
        assert cfg.accelerator_type == "NVIDIA_TESLA_T4"
        assert cfg.replica_count == 1
        assert cfg.max_cost_usd == pytest.approx(50.0)

    def test_cost_cap_is_required_field(self):
        from backend.training.vertex_job import VertexJobConfig
        cfg = VertexJobConfig(max_cost_usd=10.0)
        assert cfg.max_cost_usd == pytest.approx(10.0)


class TestVertexJobSubmitter:
    def test_submit_fails_without_project(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(project="", staging_bucket="gs://b", container_uri="gcr.io/p/img")
        submitter = VertexJobSubmitter(cfg)
        submitter.project = ""  # ensure no env fallback
        result = submitter.submit()
        assert result.status == "failed"
        assert "GOOGLE_CLOUD_PROJECT" in result.error

    def test_submit_fails_without_staging_bucket(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(project="my-proj", staging_bucket="", container_uri="gcr.io/p/img")
        submitter = VertexJobSubmitter(cfg)
        result = submitter.submit()
        assert result.status == "failed"
        assert "staging_bucket" in result.error

    def test_submit_fails_without_container_uri(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(project="my-proj", staging_bucket="gs://b", container_uri="")
        submitter = VertexJobSubmitter(cfg)
        result = submitter.submit()
        assert result.status == "failed"
        assert "container_uri" in result.error

    def test_submit_fails_gracefully_without_sdk(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(
            project="my-proj",
            staging_bucket="gs://b",
            container_uri="gcr.io/p/img",
        )
        submitter = VertexJobSubmitter(cfg)
        with patch.dict("sys.modules", {"google.cloud.aiplatform": None}):
            result = submitter.submit()
        assert result.status == "failed"
        assert "google-cloud-aiplatform" in result.error

    def test_build_worker_pool_spec_with_accelerator(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(
            project="p", staging_bucket="gs://b", container_uri="gcr.io/p/img",
            accelerator_type="NVIDIA_TESLA_T4", accelerator_count=2,
        )
        submitter = VertexJobSubmitter(cfg)
        spec = submitter._build_worker_pool_spec(cfg)
        assert spec[0]["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"
        assert spec[0]["machine_spec"]["accelerator_count"] == 2

    def test_build_worker_pool_spec_without_accelerator(self):
        from backend.training.vertex_job import VertexJobConfig, VertexJobSubmitter
        cfg = VertexJobConfig(
            project="p", staging_bucket="gs://b", container_uri="gcr.io/p/img",
            accelerator_type="",
        )
        submitter = VertexJobSubmitter(cfg)
        spec = submitter._build_worker_pool_spec(cfg)
        assert "accelerator_type" not in spec[0]["machine_spec"]
