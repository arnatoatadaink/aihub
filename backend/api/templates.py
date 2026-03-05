"""Prompt template library endpoints."""
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1/templates", tags=["templates"])

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def _load_template(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["id"] = path.stem
    return data


@router.get("")
async def list_templates():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    templates = []
    for p in sorted(TEMPLATES_DIR.glob("*.yaml")):
        try:
            templates.append(_load_template(p))
        except Exception:
            continue
    return {"templates": templates}


@router.get("/{template_id}")
async def get_template(template_id: str):
    path = TEMPLATES_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return _load_template(path)


class SaveTemplateRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    user_template: str = ""
    tags: list[str] = []
    default_params: dict = {}


@router.post("/{template_id}")
async def save_template(template_id: str, req: SaveTemplateRequest):
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATES_DIR / f"{template_id}.yaml"
    data = req.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return {"status": "ok", "id": template_id}


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    path = TEMPLATES_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    path.unlink()
    return {"status": "ok"}
