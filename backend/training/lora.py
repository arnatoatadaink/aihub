"""LoRA / TinyLoRA application helpers.

Provides utility functions for applying LoRA adapters to HuggingFace models
via the PEFT library, and for merging / saving adapters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # None → auto-detect
    bias: str = "none"                        # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"


def apply_lora(model, rank: int = 8, alpha: int | None = None, **kwargs):
    """Apply LoRA adapters to a CausalLM model and return the PEFT model."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("peft is required: pip install peft") from e

    cfg = LoRAConfig(rank=rank, lora_alpha=alpha or rank * 2, **kwargs)

    # Auto-detect linear projection layers if not specified
    target_modules = cfg.target_modules or _detect_target_modules(model)

    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: trainable params=%d (%.2f%% of %d)",
        trainable,
        100.0 * trainable / total if total else 0,
        total,
    )
    return peft_model


def merge_and_save(peft_model, output_dir: str) -> str:
    """Merge LoRA weights into base model and save to disk."""
    try:
        merged = peft_model.merge_and_unload()
    except Exception as e:
        raise RuntimeError(f"Failed to merge LoRA weights: {e}") from e

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(path))
    logger.info("Merged model saved to %s", path)
    return str(path)


def load_adapter(base_model, adapter_path: str):
    """Load a saved PEFT adapter onto a base model."""
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is required: pip install peft") from e

    return PeftModel.from_pretrained(base_model, adapter_path)


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

_COMMON_TARGETS = {
    # LLaMA / Mistral / Gemma family
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def _detect_target_modules(model) -> list[str]:
    """Return linear layer names that are commonly targeted by LoRA."""
    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in _COMMON_TARGETS:
            found.add(leaf)
    if not found:
        # Fallback: target all Linear layers (heavier but safe)
        import torch.nn as nn
        found = {
            name.split(".")[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
        }
    return list(found)
