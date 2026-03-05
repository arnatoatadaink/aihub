"""GRPO (Group Relative Policy Optimization) training loop.

This module implements the MED framework's reinforcement-learning-from-AI-feedback
pipeline where a Teacher model (Gemini) generates reward signals for a Student model.

Architecture:
  Teacher (Gemini API) → reward scoring
  Student (7-8B OSS, HuggingFace) → policy updates via GRPO

Reference: DeepSeekMath GRPO paper + TRL GRPOTrainer
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    # Model
    model_name_or_path: str = "google/gemma-2-2b-it"
    output_dir: str = "./outputs/grpo"

    # Training hyperparameters
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_seq_length: int = 512
    max_new_tokens: int = 256

    # GRPO-specific
    num_generations: int = 4        # G — samples per prompt
    beta: float = 0.01              # KL divergence coefficient
    temperature: float = 0.9

    # Data
    dataset_path: str = ""          # JSONL path: {"prompt": "...", "answer": "..."}
    max_samples: int = -1           # -1 = all

    # Reward
    reward_model: str = "gemini"    # "gemini" | "rule_based"
    reward_prompt_template: str = (
        "Rate the following response to the question on a scale of 0.0 to 1.0.\n"
        "Question: {question}\nResponse: {response}\n"
        "Output ONLY a number between 0.0 and 1.0."
    )

    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 10

    # Extras
    use_peft: bool = True           # Apply LoRA via PEFT
    lora_rank: int = 8
    seed: int = 42


@dataclass
class GRPOTrainResult:
    status: str                     # "completed" | "failed"
    epochs_done: int = 0
    steps_done: int = 0
    final_loss: float = 0.0
    output_dir: str = ""
    error: str = ""
    logs: list[dict] = field(default_factory=list)


class GRPOTrainer:
    """Wraps TRL's GRPOTrainer with Gemini-based reward and progress callbacks."""

    def __init__(self, config: GRPOConfig, progress_cb: Callable[[str], None] | None = None):
        self.config = config
        self.progress_cb = progress_cb or (lambda msg: logger.info(msg))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> GRPOTrainResult:
        cfg = self.config
        result = GRPOTrainResult(status="failed", output_dir=cfg.output_dir)
        try:
            self.progress_cb("Loading dataset…")
            dataset = self._load_dataset()

            self.progress_cb("Loading model and tokenizer…")
            model, tokenizer = self._load_model()

            self.progress_cb("Building reward function…")
            reward_fn = self._build_reward_fn()

            self.progress_cb("Starting GRPO training…")
            trainer = self._build_trainer(model, tokenizer, dataset, reward_fn)
            train_output = trainer.train()

            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            trainer.save_model(cfg.output_dir)

            result.status = "completed"
            result.epochs_done = int(train_output.metrics.get("epoch", 0))
            result.steps_done = train_output.global_step
            result.final_loss = train_output.metrics.get("train_loss", 0.0)
            result.output_dir = cfg.output_dir
            self.progress_cb(f"Training complete. Model saved to {cfg.output_dir}")

        except ImportError as e:
            result.error = f"Missing dependency: {e}. Install: pip install trl transformers peft"
            self.progress_cb(f"ERROR: {result.error}")
        except Exception as e:
            result.error = str(e)
            self.progress_cb(f"ERROR: {e}")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self):
        from datasets import Dataset, load_dataset

        cfg = self.config
        if cfg.dataset_path and Path(cfg.dataset_path).exists():
            records = []
            with open(cfg.dataset_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if cfg.max_samples > 0:
                records = records[: cfg.max_samples]
            return Dataset.from_list(records)
        # Fallback: tiny synthetic dataset for smoke-testing
        self.progress_cb("No dataset_path provided — using synthetic demo data")
        return Dataset.from_list(
            [{"prompt": f"What is {i} + {i}?", "answer": str(i * 2)} for i in range(1, 101)]
        )

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if cfg.use_peft:
            from .lora import apply_lora
            model = apply_lora(model, rank=cfg.lora_rank)
            self.progress_cb(f"LoRA applied (rank={cfg.lora_rank})")

        return model, tokenizer

    def _build_reward_fn(self) -> Callable:
        cfg = self.config

        if cfg.reward_model == "rule_based":
            def rule_reward(prompts, completions, **kwargs):
                rewards = []
                for prompt, completion in zip(prompts, completions):
                    answer = kwargs.get("answer", [""])[0] if kwargs.get("answer") else ""
                    score = 1.0 if answer.strip() in completion else 0.0
                    rewards.append(score)
                return rewards
            return rule_reward

        # Gemini-based reward
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        reward_model = genai.GenerativeModel("gemini-1.5-flash")
        template = cfg.reward_prompt_template

        def gemini_reward(prompts, completions, **kwargs):
            rewards = []
            for prompt, completion in zip(prompts, completions):
                eval_prompt = template.format(question=prompt, response=completion)
                try:
                    resp = reward_model.generate_content(eval_prompt)
                    score = float(resp.text.strip())
                    score = max(0.0, min(1.0, score))
                except Exception:
                    score = 0.0
                rewards.append(score)
                time.sleep(0.1)  # rate-limit guard
            return rewards

        return gemini_reward

    def _build_trainer(self, model, tokenizer, dataset, reward_fn):
        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer as TRLGRPOTrainer

        cfg = self.config
        training_args = TRLGRPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_completion_length=cfg.max_new_tokens,
            num_generations=cfg.num_generations,
            beta=cfg.beta,
            temperature=cfg.temperature,
            save_steps=cfg.save_steps,
            logging_steps=cfg.logging_steps,
            seed=cfg.seed,
            report_to="none",
        )
        return TRLGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=dataset,
        )
