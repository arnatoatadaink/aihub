"""Pipeline execution engine.

A pipeline is a list of LLM steps executed sequentially.
Each step can reference:
  {{input}}       — the original user input (constant across all steps)
  {{prev_output}} — the output of the immediately preceding step
  {{step:<id>}}   — the output of a specific step (by step id)

Pipeline definition schema:
{
  "id": "uuid",
  "name": "My Pipeline",
  "description": "...",
  "steps": [
    {
      "id": "step1",
      "name": "Translation",
      "provider": "gemini",           # built-in key or "custom_<id>"
      "model": "gemini-2.0-flash",
      "system_prompt": "Translate to English.",
      "input_template": "{{input}}",  # default: "{{prev_output}}" for step > 0
      "params": {
        "temperature": 0.3,
        "max_tokens": 1024,
        "top_p": 1.0
      }
    },
    {
      "id": "step2",
      "name": "Summary",
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "system_prompt": "Summarize in 3 bullet points.",
      "input_template": "{{prev_output}}",
      "params": {"temperature": 0.5, "max_tokens": 512}
    }
  ]
}
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

_PIPELINE_STORE = Path(os.getenv("PIPELINE_STORE_PATH", "data/pipelines.json"))


# ---------------------------------------------------------------------------
# Pipeline store helpers
# ---------------------------------------------------------------------------

def _load_store() -> dict[str, dict]:
    if not _PIPELINE_STORE.exists():
        return {}
    try:
        return json.loads(_PIPELINE_STORE.read_text())
    except Exception:
        return {}


def _save_store(store: dict[str, dict]) -> None:
    _PIPELINE_STORE.parent.mkdir(parents=True, exist_ok=True)
    _PIPELINE_STORE.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def load_pipelines() -> list[dict]:
    return list(_load_store().values())


def get_pipeline(pipeline_id: str) -> dict | None:
    return _load_store().get(pipeline_id)


def save_pipeline(definition: dict) -> dict:
    if not definition.get("id"):
        definition["id"] = str(uuid.uuid4())
    store = _load_store()
    store[definition["id"]] = definition
    _save_store(store)
    return definition


def delete_pipeline(pipeline_id: str) -> bool:
    store = _load_store()
    if pipeline_id not in store:
        return False
    del store[pipeline_id]
    _save_store(store)
    return True


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _get_provider(provider_key: str):
    """Return an instantiated BaseProvider. Handles custom_ prefix."""
    if provider_key.startswith("custom_"):
        from backend.providers.custom import build_custom_provider
        return build_custom_provider(provider_key)

    from backend.providers import PROVIDER_MAP
    cls = PROVIDER_MAP.get(provider_key)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_key!r}")
    return cls()


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------

def _content_to_text(value: str | list) -> str:
    """Extract plain text from a string or OpenAI-style content list."""
    if isinstance(value, str):
        return value
    return "\n".join(
        p.get("text", "") for p in value
        if isinstance(p, dict) and p.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render(template: str, context: dict[str, str | list]) -> str | list:
    """Replace {{key}} placeholders with values from context.

    If the template is a single bare placeholder (e.g. "{{input}}") and the
    corresponding context value is a list (multimodal content), the list is
    returned directly so vision-capable providers receive it unchanged.
    """
    # Fast path: single placeholder referencing a list value
    stripped = template.strip()
    m = re.fullmatch(r"\{\{([^}]+)\}\}", stripped)
    if m:
        key = m.group(1).strip()
        val = context.get(key)
        if isinstance(val, list):
            return val  # pass multipart content through unchanged

    # Normal string rendering (stringify lists when embedded in text)
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        val = context.get(key, match.group(0))
        return _content_to_text(val) if isinstance(val, list) else val

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PipelineExecutor:
    """Execute a pipeline definition step by step."""

    async def run(self, definition: dict, user_input: str | list) -> list[dict]:
        """Run all steps and return per-step results.

        Args:
            user_input: Plain string or OpenAI-style content list (multimodal).

        Returns:
            list of dicts:
              {
                "step_index": int,
                "step_id":    str,
                "step_name":  str,
                "provider":   str,
                "model":      str,
                "input":      str,    # rendered input sent to the LLM (serialised)
                "output":     str,    # LLM response
                "error":      str | None
              }
        """
        steps = definition.get("steps", [])
        results: list[dict] = []
        # context for template rendering: step outputs by id + aliases
        # Values may be str (text) or list (multimodal content)
        context: dict[str, str | list] = {"input": user_input}

        for idx, step in enumerate(steps):
            step_id = step.get("id", f"step{idx}")
            step_name = step.get("name", step_id)
            provider_key = step.get("provider", "gemini")
            model = step.get("model", "")
            system_prompt = step.get("system_prompt", "")
            params = dict(step.get("params", {}))
            if model:
                params["model"] = model

            # Default input template: first step uses {{input}}, rest use {{prev_output}}
            default_template = "{{input}}" if idx == 0 else "{{prev_output}}"
            template = step.get("input_template") or default_template

            rendered_input = _render(template, context)

            result: dict = {
                "step_index": idx,
                "step_id": step_id,
                "step_name": step_name,
                "provider": provider_key,
                "model": model,
                # Serialise list content to JSON string for display
                "input": (
                    rendered_input if isinstance(rendered_input, str)
                    else json.dumps(rendered_input, ensure_ascii=False)
                ),
                "output": "",
                "error": None,
            }

            try:
                provider = _get_provider(provider_key)
                messages: list[dict] = []
                if system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                # rendered_input may be str or list; providers handle both
                messages.append({"role": "user", "content": rendered_input})

                output = await provider.generate(messages, params)
                result["output"] = output

                # Wrap image/video generation outputs as image_url content so
                # downstream vision steps can consume them directly
                if getattr(provider, "modal_type", "text") == "image" and output:
                    context_value: str | list = [
                        {"type": "image_url", "image_url": {"url": output}}
                    ]
                else:
                    context_value = output

                context[f"step:{step_id}"] = context_value
                context["prev_output"] = context_value

            except Exception as exc:
                result["error"] = str(exc)
                # On error: propagate empty string so downstream steps can still run
                context[f"step:{step_id}"] = ""
                context["prev_output"] = ""

            results.append(result)

        return results


# Module-level convenience function
async def run_pipeline(definition: dict, user_input: str | list) -> list[dict]:
    return await PipelineExecutor().run(definition, user_input)
