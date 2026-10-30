"""
Self-improvement protocol: Frozen / Reflection / Scaffold improvement /
Recursive improvement, per the user's research doc.

Scope note (deliberately small, like every other extension in this
pilot): this runs ONE lineage per condition, on ONE task, for a handful of
generations, with single_agent only (not the 4-worker team, to bound
cost). This is a MINIMAL, ILLUSTRATIVE pass through the protocol's
mechanics, not a powered self-improvement study -- per the blueprint's
own framing of its "Practical first pilot" for this exact extension:
"Three is a debugging minimum, not a power guarantee."

What can change, per condition (matches the user doc's table):
  - frozen: fresh attempts, no memory, no system changes.
  - reflection: carries forward a text summary of prior attempts' outcomes
    (prepended to the TASK prompt), system prompt is unchanged.
  - scaffold: after each generation, asks the model to propose ONE
    additional rule (a sentence) to append to the SYSTEM prompt, using a
    FIXED meta-prompt ("how do you propose a rule from a failure").
  - recursive: same as scaffold, but the meta-prompt itself is also
    revisable -- after each generation the model may ALSO propose a
    revision to the meta-prompt used to derive future rules.

What stays fixed in all conditions (matches the doc's "what remains
fixed"): task requirements/feasibility ground truth, the trusted
evaluator, resource limits. The system cannot grant itself new
information/permissions -- the "patches" here are limited to editable
text (system-prompt rules / the meta-prompt), never code, tool
permissions, or the evaluator.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from harness.agents import build_system_prompt, call_for_decision
from harness.evaluator import finalize_run_result
from harness.llm_client import LLMClient
from harness.schemas import FinalDecision, RunConfig, RunResult
from tasks.registry import TaskVariant

DEFAULT_META_PROMPT = (
    "You just attempted a coding task and failed. Given the task and your "
    "failed attempt's outcome, propose ONE short, general rule (a single "
    "sentence, imperative mood) to add to your own instructions that would "
    "help you avoid this mistake on similar future tasks. Reply with ONLY "
    "that one sentence, no quotes, no preamble."
)

META_META_PROMPT = (
    "You just proposed a rule for yourself after a failure, using the "
    "instructions below. Consider whether THOSE INSTRUCTIONS THEMSELVES "
    "could be improved to produce better future rules (e.g. more specific, "
    "less likely to produce a vague or unhelpful rule). If you see a clear "
    "improvement, reply with the ONE revised instruction paragraph "
    "(replacing the instructions below). If you see no clear improvement, "
    "reply with exactly: NO_CHANGE\n\n"
    "Current instructions:\n{meta_prompt}"
)


@dataclass
class GenerationRecord:
    generation: int
    run_result: RunResult
    system_prompt_used: str
    reflection_text_used: str = ""
    proposed_rule: str = ""
    meta_prompt_used: str = ""


@dataclass
class LineageResult:
    condition: str
    variant_id: str
    model_name: str
    generations: list = field(default_factory=list)  # list[GenerationRecord]
    final_meta_prompt: str = DEFAULT_META_PROMPT


def _reflection_summary(generations: list) -> str:
    if not generations:
        return ""
    lines = []
    for g in generations:
        rr = g.run_result
        dec = rr.final_decision.decision if rr.final_decision else "NONE"
        lines.append(f"Attempt {g.generation}: decision={dec}, verdict={rr.verdict} "
                      f"(reason given: {rr.final_decision.reason[:150] if rr.final_decision else 'n/a'})")
    return (
        "Summary of your previous attempts at this exact task (same task, "
        "fresh code each time -- learn from these outcomes):\n" + "\n".join(lines)
    )


def _run_one_generation(variant: TaskVariant, client: LLMClient, cfg: RunConfig,
                          system_prompt: str, reflection_text: str, architecture: str = "single_agent") -> RunResult:
    from harness.protocol import _probe_from_decision, ARCHITECTURES  # reuse helper

    task_prompt = f"{reflection_text}\n\n---\n{variant.prompt}" if reflection_text else variant.prompt

    if architecture != "single_agent":
        # Each generation's attempt is produced by the full team/comm
        # protocol instead of one agent; the reflection/scaffold text still
        # rides along as a prefix baked into the shared task prompt every
        # team member and the aggregator see.
        team_variant = dataclasses.replace(variant, prompt=task_prompt)
        rr = ARCHITECTURES[architecture](team_variant, client, cfg)
        return finalize_run_result(variant, rr)

    t0 = time.time()
    result = RunResult(run_id=cfg.run_id, config=cfg, true_feasible=variant.feasible)
    try:
        d0, raw0, usage0 = call_for_decision(
            client, system_prompt, task_prompt, cfg.temperature, cfg.max_output_tokens_per_call,
        )
        result.num_calls += 1
        result.tokens_used_prompt += usage0.get("prompt_tokens", 0)
        result.tokens_used_completion += usage0.get("completion_tokens", 0)
        if d0 is None:
            result.termination = "UNPARSEABLE_OUTPUT" if usage0.get("finish_reason") != "length" else "RESPONSE_TRUNCATED"
            result.wall_time_s = time.time() - t0
            return result
        result.probes.append(_probe_from_decision("agent_0", 0, "pre_discussion", d0))
        result.final_decision = d0
        result.artifact_status = "COMPLETE_SUBMITTED" if d0.solution_code else "NO_SUBMISSION"
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        result.termination = "RUNTIME_ERROR"
    result.wall_time_s = time.time() - t0
    return finalize_run_result(variant, result)


def _propose_rule(client: LLMClient, variant: TaskVariant, last_gen: GenerationRecord, meta_prompt: str) -> str:
    rr = last_gen.run_result
    context = (
        f"{meta_prompt}\n\nTask:\n{variant.prompt}\n\nYour failed attempt's decision: "
        f"{rr.final_decision.decision if rr.final_decision else 'NONE'}, verdict: {rr.verdict}, "
        f"your stated reason: {rr.final_decision.reason if rr.final_decision else 'n/a'}"
    )
    text, _usage = client.chat([{"role": "user", "content": context}], temperature=1.0, max_tokens=120)
    return text.strip().strip('"')


def _propose_meta_prompt_revision(client: LLMClient, meta_prompt: str) -> str:
    text, _usage = client.chat(
        [{"role": "user", "content": META_META_PROMPT.format(meta_prompt=meta_prompt)}],
        temperature=1.0, max_tokens=250,
    )
    text = text.strip()
    if text.upper().startswith("NO_CHANGE"):
        return meta_prompt
    return text


def run_lineage(variant: TaskVariant, client: LLMClient, model_name: str, condition: str,
                  n_generations: int = 3, lineage_idx: int = 0, max_output_tokens: int = 2200,
                  architecture: str = "single_agent") -> LineageResult:
    assert condition in ("frozen", "reflection", "scaffold", "recursive")
    base_system_prompt = build_system_prompt()
    system_prompt = base_system_prompt
    meta_prompt = DEFAULT_META_PROMPT
    lineage = LineageResult(condition=condition, variant_id=variant.variant_id, model_name=model_name)

    for g in range(n_generations):
        reflection_text = _reflection_summary(lineage.generations) if condition == "reflection" else ""
        cfg = RunConfig(
            run_id=f"selfimprove_{condition}_{variant.variant_id}_{model_name}_lineage{lineage_idx}_gen{g}",
            family_id=variant.family_id, category=variant.category, variant_id=variant.variant_id,
            model_name=model_name,
            architecture=architecture if architecture != "single_agent" else "self_improve_single_agent",
            repeat_idx=g, max_output_tokens_per_call=max_output_tokens,
        )
        rr = _run_one_generation(variant, client, cfg, system_prompt, reflection_text, architecture=architecture)
        record = GenerationRecord(
            generation=g, run_result=rr, system_prompt_used=system_prompt,
            reflection_text_used=reflection_text, meta_prompt_used=meta_prompt,
        )
        lineage.generations.append(record)

        failed = rr.verdict != "PASS"
        if condition in ("scaffold", "recursive") and failed and g < n_generations - 1:
            try:
                rule = _propose_rule(client, variant, record, meta_prompt)
                record.proposed_rule = rule
                if rule and len(rule) < 400:
                    system_prompt = system_prompt + f"\n- {rule}"
            except Exception as e:  # noqa: BLE001
                record.proposed_rule = f"<proposal failed: {e}>"
            if condition == "recursive":
                try:
                    meta_prompt = _propose_meta_prompt_revision(client, meta_prompt)
                except Exception:  # noqa: BLE001
                    pass

    lineage.final_meta_prompt = meta_prompt
    return lineage
