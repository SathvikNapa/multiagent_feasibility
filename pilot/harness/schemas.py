"""Structured-output and event schemas, per the blueprint's protocol section."""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProbeRecord(BaseModel):
    agent_id: str
    round_idx: int
    phase: Literal["pre_discussion", "post_round", "final"]
    p_feasible: float
    p_team_success: float
    reason: str = ""
    conflict_constraint_ids: list[str] = Field(default_factory=list)
    ts: float = Field(default_factory=time.time)


class FinalDecision(BaseModel):
    decision: Literal["SOLVED", "INFEASIBLE", "UNKNOWN", "REQUEST_RELAXATION"]
    p_feasible: float
    p_team_success: float
    solution_code: Optional[str] = None
    conflict_constraint_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    requested_relaxations: list[str] = Field(default_factory=list)
    reason: str = ""


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str
    ts: float = Field(default_factory=time.time)
    kind: Literal[
        "message", "tool_call", "artifact", "verification_result",
        "budget_consumption", "termination", "probe",
    ]
    agent_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)


class RunConfig(BaseModel):
    run_id: str
    family_id: str
    category: Literal["F1", "F2", "F3"]
    variant_id: str
    model_name: str
    architecture: str  # see harness/protocol.py's ARCHITECTURES dict for valid values
    repeat_idx: int
    max_rounds: int = 2
    max_tool_calls: int = 3
    temperature: float = 1.0
    max_output_tokens_per_call: int = 2200
    total_token_budget: int = 24000


class RunResult(BaseModel):
    run_id: str
    config: RunConfig
    true_feasible: bool
    final_decision: Optional[FinalDecision] = None
    probes: list[ProbeRecord] = Field(default_factory=list)
    verdict: str = "EVALUATION_ERROR"
    functional_pass: bool = False
    constraint_pass: bool = False
    violated_constraint_ids: list[str] = Field(default_factory=list)
    artifact_status: Literal["COMPLETE_SUBMITTED", "NO_SUBMISSION", "RESPONSE_TRUNCATED"] = "NO_SUBMISSION"
    termination: str = "COMPLETED"
    tokens_used_prompt: int = 0
    tokens_used_completion: int = 0
    num_calls: int = 0
    wall_time_s: float = 0.0
    # Separate from the above (which includes these too, since they count
    # toward the same team budget) -- the doc requires the aggregator's
    # usage be reported separately from the participating agents'.
    aggregator_calls: int = 0
    aggregator_tokens_prompt: int = 0
    aggregator_tokens_completion: int = 0
    supported_detection: Optional[bool] = None
    error: Optional[str] = None
    extra: dict = Field(default_factory=dict)
