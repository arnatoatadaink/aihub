"""Pipeline management and execution endpoints.

Endpoints:
  GET    /v1/pipelines          — list all saved pipelines
  POST   /v1/pipelines          — create / update a pipeline
  GET    /v1/pipelines/{id}     — get a single pipeline definition
  DELETE /v1/pipelines/{id}     — delete a pipeline
  POST   /v1/pipelines/{id}/run — execute a pipeline with user input
  POST   /v1/pipelines/run      — execute an inline (unsaved) pipeline
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline.executor import (
    delete_pipeline,
    get_pipeline,
    load_pipelines,
    run_pipeline,
    save_pipeline,
)

router = APIRouter(prefix="/v1/pipelines", tags=["pipeline"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PipelineStep(BaseModel):
    id: str = ""
    name: str = ""
    provider: str = "gemini"
    model: str = ""
    system_prompt: str = ""
    input_template: str = ""
    params: dict = Field(default_factory=dict)


class PipelineDefinition(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    steps: list[PipelineStep] = Field(default_factory=list)


class PipelineRunRequest(BaseModel):
    input: str
    definition: PipelineDefinition | None = None  # for inline (unsaved) pipelines


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def list_all_pipelines():
    return {"pipelines": load_pipelines()}


@router.post("")
async def create_pipeline(body: PipelineDefinition):
    definition = body.model_dump()
    saved = save_pipeline(definition)
    return {"pipeline": saved}


@router.get("/{pipeline_id}")
async def get_one_pipeline(pipeline_id: str):
    p = get_pipeline(pipeline_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")
    return {"pipeline": p}


@router.delete("/{pipeline_id}")
async def remove_pipeline(pipeline_id: str):
    if not delete_pipeline(pipeline_id):
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")
    return {"deleted": pipeline_id}


# ---------------------------------------------------------------------------
# Execution endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_inline_pipeline(body: PipelineRunRequest):
    """Execute an unsaved pipeline definition directly."""
    if body.definition is None:
        raise HTTPException(status_code=400, detail="definition is required for inline runs")
    definition = body.definition.model_dump()
    try:
        results = await run_pipeline(definition, body.input)
        return {"results": results, "final_output": results[-1]["output"] if results else ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/run")
async def run_saved_pipeline(pipeline_id: str, body: PipelineRunRequest):
    """Execute a saved pipeline by ID."""
    definition = get_pipeline(pipeline_id)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")
    # Allow step overrides from request
    if body.definition is not None:
        definition = body.definition.model_dump()
    try:
        results = await run_pipeline(definition, body.input)
        return {
            "pipeline_id": pipeline_id,
            "results": results,
            "final_output": results[-1]["output"] if results else "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
