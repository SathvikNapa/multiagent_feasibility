"""
Architecture orchestration.

- single_agent: one agent, two rounds (attempt + self-review).
- team4_aggregator: LEGACY 4-worker team, fully-connected discussion, a
  DETERMINISTIC (non-LLM) aggregator. Kept as-is for comparability with
  already-collected results; this was the pilot's original design,
  before the user's research doc specified communication topologies and
  an LLM aggregator explicitly.
- comm_independent / comm_chain / comm_star / comm_fully_connected: the
  user doc's four communication protocols, implemented as one
  topology-parameterized function (`run_comm_topology`) that varies only
  WHO SEES WHOM in the single discussion round (see
  `_topology_neighbors`), plus a SEPARATE LLM aggregation call that
  selects the team's final decision from the 4 agents' submissions
  (never writes new code itself, never calls tools) -- its usage is
  tracked in RunResult.aggregator_* fields, separate from the 4
  participating agents' num_calls/tokens_used_*, per the doc's
  requirement that the aggregator's use be "reported separately".

Simplifications vs. the full blueprint/doc protocol (documented, not
hidden):
  - Probe elicitation and the action/claim submission are combined into
    one structured JSON call per round, instead of separate probe +
    submission calls, to control pilot token cost. The probe fields
    (p_feasible, p_team_success) are still private per round and are
    never shown to peers -- only decision/reason/evidence_ids are shared,
    matching the doc's "private feasibility judgments remain hidden
    unless confidence sharing is explicitly being tested."
  - Two rounds total (pre-discussion + one discussion/self-review round),
    not the doc's open-ended synchronous-rounds schedule.
  - Agent *positions* are not rotated across repeats (the doc asks for
    this "to reduce position effects") -- each repeat reuses the same
    agent-index-to-topology-slot assignment.
"""
from __future__ import annotations

import functools
import time
from collections import Counter

from harness.agents import (AGGREGATOR_SYSTEM_PROMPT, build_aggregator_prompt,
                             build_review_prompt, build_system_prompt,
                             build_team_review_prompt, call_for_decision)
from harness.llm_client import LLMClient
from harness.schemas import FinalDecision, ProbeRecord, RunConfig, RunResult
from tasks.registry import TaskVariant

DECISION_PRIORITY = ["SOLVED", "INFEASIBLE", "REQUEST_RELAXATION", "UNKNOWN"]


def _probe_from_decision(agent_id: str, round_idx: int, phase: str, d: FinalDecision) -> ProbeRecord:
    return ProbeRecord(
        agent_id=agent_id,
        round_idx=round_idx,
        phase=phase,
        p_feasible=d.p_feasible,
        p_team_success=d.p_team_success,
        reason=d.reason,
        conflict_constraint_ids=d.conflict_constraint_ids,
    )


def run_single_agent(variant: TaskVariant, client: LLMClient, cfg: RunConfig) -> RunResult:
    t0 = time.time()
    result = RunResult(run_id=cfg.run_id, config=cfg, true_feasible=variant.feasible)
    system_prompt = build_system_prompt()

    try:
        d0, raw0, usage0 = call_for_decision(
            client, system_prompt, variant.prompt, cfg.temperature, cfg.max_output_tokens_per_call,
        )
        result.num_calls += 1
        result.tokens_used_prompt += usage0.get("prompt_tokens", 0)
        result.tokens_used_completion += usage0.get("completion_tokens", 0)

        if d0 is None:
            result.termination = "RESPONSE_TRUNCATED" if usage0.get("finish_reason") == "length" else "UNPARSEABLE_OUTPUT"
            result.artifact_status = "RESPONSE_TRUNCATED" if result.termination == "RESPONSE_TRUNCATED" else "NO_SUBMISSION"
            result.error = f"finish_reason={usage0.get('finish_reason')} raw_tail={raw0[-800:]!r}"
            result.wall_time_s = time.time() - t0
            return result

        result.probes.append(_probe_from_decision("agent_0", 0, "pre_discussion", d0))

        review_prompt = build_review_prompt(variant.prompt, d0.model_dump_json())
        d1, raw1, usage1 = call_for_decision(
            client, system_prompt, review_prompt, cfg.temperature, cfg.max_output_tokens_per_call,
        )
        result.num_calls += 1
        result.tokens_used_prompt += usage1.get("prompt_tokens", 0)
        result.tokens_used_completion += usage1.get("completion_tokens", 0)

        final = d1 if d1 is not None else d0
        result.probes.append(_probe_from_decision("agent_0", 1, "final", final))
        result.final_decision = final
        result.artifact_status = "COMPLETE_SUBMITTED" if final.solution_code else "NO_SUBMISSION"
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        result.termination = "RUNTIME_ERROR"

    result.wall_time_s = time.time() - t0
    return result


def _aggregate_team(finals: list[FinalDecision]) -> FinalDecision:
    votes = Counter(d.decision for d in finals)
    top_count = max(votes.values())
    tied = [dec for dec, c in votes.items() if c == top_count]
    if len(tied) == 1:
        chosen = tied[0]
    else:
        chosen = next(d for d in DECISION_PRIORITY if d in tied)

    contributing = [d for d in finals if d.decision == chosen]
    solution_code = None
    if chosen == "SOLVED":
        with_code = [d for d in contributing if d.solution_code]
        solution_code = with_code[0].solution_code if with_code else None
        if solution_code is None:
            chosen = "UNKNOWN"  # majority said SOLVED but nobody actually produced code

    # str(...) guards against a rare observed crash where a model's JSON put a
    # non-string (e.g. a number) in one of these list[str] fields and it slid
    # through validation, producing a set of mixed types that sorted() can't
    # compare -- "'<' not supported between instances of 'NoneType' and 'int'".
    evidence_ids = sorted({str(e) for d in contributing for e in d.evidence_ids if e is not None})
    conflict_ids = sorted({str(c) for d in contributing for c in d.conflict_constraint_ids if c is not None})
    relaxations = sorted({str(r) for d in contributing for r in d.requested_relaxations if r is not None})
    reason = " | ".join(d.reason for d in contributing if d.reason)[:800]

    return FinalDecision(
        decision=chosen,
        p_feasible=sum(d.p_feasible for d in finals) / len(finals),
        p_team_success=sum(d.p_team_success for d in finals) / len(finals),
        solution_code=solution_code,
        conflict_constraint_ids=conflict_ids,
        evidence_ids=evidence_ids,
        requested_relaxations=relaxations,
        reason=reason,
    )


def run_team4(variant: TaskVariant, client: LLMClient, cfg: RunConfig) -> RunResult:
    t0 = time.time()
    result = RunResult(run_id=cfg.run_id, config=cfg, true_feasible=variant.feasible)
    system_prompt = build_system_prompt()
    n_workers = 4

    try:
        round0: list[FinalDecision] = []
        round0_text: list[str] = []
        for w in range(n_workers):
            d, raw, usage = call_for_decision(
                client, system_prompt, variant.prompt, cfg.temperature, cfg.max_output_tokens_per_call,
            )
            result.num_calls += 1
            result.tokens_used_prompt += usage.get("prompt_tokens", 0)
            result.tokens_used_completion += usage.get("completion_tokens", 0)
            if d is None:
                d = FinalDecision(decision="UNKNOWN", p_feasible=0.5, p_team_success=0.0,
                                   reason="unparseable round-0 output")
                if result.error is None:
                    result.error = f"agent_{w} round0 unparseable: finish_reason={usage.get('finish_reason')} raw_tail={raw[-800:]!r}"
            round0.append(d)
            round0_text.append(d.model_dump_json())
            result.probes.append(_probe_from_decision(f"agent_{w}", 0, "pre_discussion", d))

        round1: list[FinalDecision] = []
        for w in range(n_workers):
            peer_summaries = [
                f"decision={pd.decision}, reason={pd.reason[:200]}, evidence_ids={pd.evidence_ids}"
                for i, pd in enumerate(round0) if i != w
            ]
            review_prompt = build_team_review_prompt(variant.prompt, round0_text[w], peer_summaries)
            d, raw, usage = call_for_decision(
                client, system_prompt, review_prompt, cfg.temperature, cfg.max_output_tokens_per_call,
            )
            result.num_calls += 1
            result.tokens_used_prompt += usage.get("prompt_tokens", 0)
            result.tokens_used_completion += usage.get("completion_tokens", 0)
            if d is None:
                d = round0[w]
            round1.append(d)
            result.probes.append(_probe_from_decision(f"agent_{w}", 1, "final", d))

        final = _aggregate_team(round1)
        result.final_decision = final
        result.artifact_status = "COMPLETE_SUBMITTED" if final.solution_code else "NO_SUBMISSION"
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        result.termination = "RUNTIME_ERROR"

    result.wall_time_s = time.time() - t0
    return result


def _topology_neighbors(topology: str, n: int) -> dict:
    """For each agent index, the OTHER agents' round-0 messages it is
    allowed to see before its round-1 submission -- the four protocols
    from the user's research doc."""
    if topology == "independent_ensemble":
        return {i: [] for i in range(n)}
    if topology == "chain":
        return {i: ([i - 1] if i > 0 else []) for i in range(n)}
    if topology == "star":
        d = {0: list(range(1, n))}
        d.update({i: [0] for i in range(1, n)})
        return d
    if topology == "fully_connected":
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    raise ValueError(f"unknown topology: {topology}")


def _submission_dict(d: FinalDecision) -> dict:
    return {
        "decision": d.decision, "p_feasible": d.p_feasible, "reason": d.reason,
        "evidence_ids": d.evidence_ids, "conflict_constraint_ids": d.conflict_constraint_ids,
        "solution_code": d.solution_code,
    }


def run_comm_topology(variant: TaskVariant, client: LLMClient, cfg: RunConfig, topology: str) -> RunResult:
    """
    4-agent team under a named communication topology (independent_ensemble
    / chain / star / fully_connected), with a SEPARATE LLM aggregation call
    selecting the final decision -- per the user doc's protocol ("a
    separate aggregation call receives these submissions and selects the
    team's final decision and artifact. It cannot conduct additional
    search or call tools... its tokens count toward the team budget, and
    its use is reported separately"). Message delivery is one-directional
    per the adjacency rule above; agents with no allowed neighbors (e.g.
    chain's first agent, independent_ensemble's everyone) get a plain
    self-review round instead of a team-discussion round, rather than
    being shown an empty, confusing "your teammates said: (nothing)".
    """
    t0 = time.time()
    result = RunResult(run_id=cfg.run_id, config=cfg, true_feasible=variant.feasible)
    system_prompt = build_system_prompt()
    n_workers = 4
    neighbors = _topology_neighbors(topology, n_workers)

    try:
        round0: list[FinalDecision] = []
        round0_text: list[str] = []
        for w in range(n_workers):
            d, raw, usage = call_for_decision(
                client, system_prompt, variant.prompt, cfg.temperature, cfg.max_output_tokens_per_call,
            )
            result.num_calls += 1
            result.tokens_used_prompt += usage.get("prompt_tokens", 0)
            result.tokens_used_completion += usage.get("completion_tokens", 0)
            if d is None:
                d = FinalDecision(decision="UNKNOWN", p_feasible=0.5, p_team_success=0.0,
                                   reason="unparseable round-0 output")
            round0.append(d)
            round0_text.append(d.model_dump_json())
            result.probes.append(_probe_from_decision(f"agent_{w}", 0, "pre_discussion", d))

        final_submissions: list[FinalDecision] = []
        for w in range(n_workers):
            allowed = neighbors[w]
            if not allowed:
                review_prompt = build_review_prompt(variant.prompt, round0_text[w])
            else:
                peer_summaries = [
                    f"decision={round0[j].decision}, reason={round0[j].reason[:200]}, evidence_ids={round0[j].evidence_ids}"
                    for j in allowed
                ]
                review_prompt = build_team_review_prompt(variant.prompt, round0_text[w], peer_summaries)
            d, raw, usage = call_for_decision(
                client, system_prompt, review_prompt, cfg.temperature, cfg.max_output_tokens_per_call,
            )
            result.num_calls += 1
            result.tokens_used_prompt += usage.get("prompt_tokens", 0)
            result.tokens_used_completion += usage.get("completion_tokens", 0)
            if d is None:
                d = round0[w]
            final_submissions.append(d)
            result.probes.append(_probe_from_decision(f"agent_{w}", 1, "final", d))

        submissions_payload = [_submission_dict(d) for d in final_submissions]
        agg_prompt = build_aggregator_prompt(variant.prompt, submissions_payload)
        agg_decision, agg_raw, agg_usage = call_for_decision(
            client, AGGREGATOR_SYSTEM_PROMPT, agg_prompt, 1.0, cfg.max_output_tokens_per_call,
        )
        result.aggregator_calls += 1
        result.aggregator_tokens_prompt += agg_usage.get("prompt_tokens", 0)
        result.aggregator_tokens_completion += agg_usage.get("completion_tokens", 0)
        if agg_decision is None:
            # aggregator call itself failed to parse -- fall back to the
            # deterministic majority-vote aggregator rather than losing the run.
            agg_decision = _aggregate_team(final_submissions)

        result.final_decision = agg_decision
        result.artifact_status = "COMPLETE_SUBMITTED" if agg_decision.solution_code else "NO_SUBMISSION"
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        result.termination = "RUNTIME_ERROR"

    result.wall_time_s = time.time() - t0
    return result


ARCHITECTURES = {
    "single_agent": run_single_agent,
    "team4_aggregator": run_team4,  # legacy: deterministic aggregator, fully-connected discussion
    "comm_independent": functools.partial(run_comm_topology, topology="independent_ensemble"),
    "comm_chain": functools.partial(run_comm_topology, topology="chain"),
    "comm_star": functools.partial(run_comm_topology, topology="star"),
    "comm_fully_connected": functools.partial(run_comm_topology, topology="fully_connected"),
}
