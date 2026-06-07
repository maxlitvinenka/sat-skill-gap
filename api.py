"""FastAPI server for live pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lib import (
    K_NEIGHBORS,
    KERNEL_SIGMA,
    PROPAGATION_ALPHA,
    RANDOM_SEED,
    PipelineResult,
    analyze_student,
    build_computation_trace,
    build_dashboard_meta,
    build_students_meta,
    inject_custom_student,
    run_pipeline,
    serialize_student_dashboard,
    skills_catalog,
)

app = FastAPI(title="SAT Skill Gap Live API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class AppState:
    result: PipelineResult | None = None
    seed: int = RANDOM_SEED
    custom_injected: bool = False


state = AppState()


class PipelineRequest(BaseModel):
    seed: int = RANDOM_SEED


class ObservedSkillInput(BaseModel):
    skillId: str
    mastery: float = Field(ge=0, le=100)


class CustomStudentInput(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    observedSkills: list[ObservedSkillInput] = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    studentId: str | None = None
    seed: int | None = None
    sigma: float = Field(default=KERNEL_SIGMA, gt=0)
    alpha: float = Field(default=PROPAGATION_ALPHA, ge=0, le=1)
    k: int = Field(default=K_NEIGHBORS, ge=1, le=50)
    customStudent: CustomStudentInput | None = None


def _ensure_pipeline(seed: int | None = None) -> PipelineResult:
    target_seed = seed if seed is not None else state.seed
    if state.result is None or state.seed != target_seed or state.custom_injected:
        state.result = run_pipeline(seed=target_seed)
        state.seed = target_seed
        state.custom_injected = False
    return state.result


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/skills")
def get_skills() -> list[dict[str, str]]:
    return skills_catalog()


@app.post("/api/pipeline")
def regenerate_pipeline(body: PipelineRequest) -> dict[str, Any]:
    state.result = run_pipeline(seed=body.seed)
    state.seed = body.seed
    state.custom_injected = False
    result = state.result
    return {
        "meta": build_dashboard_meta(result, random_seed=body.seed),
        "students": build_students_meta(result),
        "defaultStudentId": result.student_ids[result.default_student_idx],
    }


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    result = _ensure_pipeline(body.seed)

    observed_work_override = None
    student_idx: int

    if body.customStudent:
        cs = body.customStudent
        skills_payload = [
            {"skill_id": s.skillId, "mastery": s.mastery} for s in cs.observedSkills
        ]
        try:
            result, student_idx, observed_work_override = inject_custom_student(
                result, cs.name.strip(), skills_payload
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state.result = result
        state.custom_injected = True
        target_id = cs.name.strip()
    else:
        target_id = body.studentId or result.student_ids[result.default_student_idx]
        if target_id not in result.student_ids:
            raise HTTPException(status_code=404, detail=f"Unknown student: {target_id}")
        student_idx = result.student_ids.index(target_id)

    analysis = analyze_student(
        result,
        student_idx,
        top_k=body.k,
        sigma=body.sigma,
        alpha=body.alpha,
        observed_work_override=observed_work_override,
    )
    trace = build_computation_trace(
        result,
        student_idx,
        analysis,
        sigma=body.sigma,
        alpha=body.alpha,
        top_k=body.k,
    )
    dashboard = serialize_student_dashboard(
        result,
        student_idx,
        analysis,
        sigma=body.sigma,
        alpha=body.alpha,
        random_seed=state.seed,
        computation_trace=trace,
    )
    return {
        "studentId": target_id,
        "meta": build_dashboard_meta(
            result,
            random_seed=state.seed,
            sigma=body.sigma,
            alpha=body.alpha,
            top_k=body.k,
        ),
        "students": build_students_meta(result),
        "dashboard": dashboard,
    }
