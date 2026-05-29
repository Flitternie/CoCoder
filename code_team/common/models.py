from __future__ import annotations

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    id: str
    stage: str
    task_type: str
    status: str = "pending"
    deps: list[str] = Field(default_factory=list)
    assignee: str | None = None
    payload: dict = Field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 1
    created_at: str
    updated_at: str
    result_ref: str | None = None
    error: str | None = None


class StageResult(BaseModel):
    passed: bool
    steps: int
    feedback: str = ""
    artifacts: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    final_score: int
    feedback: dict[str, str] = Field(default_factory=dict)


class WorkerTaskResult(BaseModel):
    ok: bool
    worker_id: str
    task_id: str
    text: str = ""
    output_path: str | None = None
    error: str | None = None


class RepoBundle(BaseModel):
    dataset: str
    repo: str
    repo_dir: str
    config: dict
    requirement: str
    uml_class: str
    uml_sequence: str
    arch_design: str


class WorkspaceManifest(BaseModel):
    dataset: str
    repo: str
    copied: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    excluded_reference_paths: list[str] = Field(default_factory=list)
