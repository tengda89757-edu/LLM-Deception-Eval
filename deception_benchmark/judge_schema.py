from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    task_name: str
    label: Any
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: list[str]
    short_rationale: str
    json_valid: bool


class FinalJudgeDecision(BaseModel):
    task_name: str
    label: Any
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: list[str]
    short_rationale: str
    final_label_source: str
    judge_disagreement_flag: int
    human_escalation_flag: int
    json_invalid: int
    cascade_level: int
    final_or_not: int = 1

