"""Per-call agent prompting + robust structured-output parsing."""
from __future__ import annotations

import json
import re
from typing import Optional

from harness.llm_client import LLMClient
from harness.schemas import FinalDecision

DECISION_SCHEMA_INSTRUCTIONS = """
You must respond with ONLY a single JSON object (no markdown fences, no
prose outside the JSON) with exactly these fields:

{
  "decision": "SOLVED" | "INFEASIBLE" | "UNKNOWN" | "REQUEST_RELAXATION",
  "p_feasible": <float 0..1, your probability the task as stated is feasible>,
  "p_team_success": <float 0..1, your probability of eventual valid success>,
  "solution_code": <string containing a complete Python function, using
                     EXACTLY the function name/signature the task asks
                     for, if decision is SOLVED, else null>,
  "conflict_constraint_ids": [<short string ids of any constraints you believe
                                conflict, else empty list>],
  "evidence_ids": [<short string labels for the evidence/reasoning points you
                     relied on, else empty list>],
  "requested_relaxations": [<short strings describing a relaxation you would
                              need if decision is REQUEST_RELAXATION, else []>],
  "reason": "<concise explanation, 1-4 sentences, including any specific
              numbers/values/counterexample that support your decision>"
}

Rules:
- Use INFEASIBLE only if you can state a specific counterexample or proof
  that no valid solution can satisfy the stated requirements.
- Use SOLVED only if you provide complete, runnable solution_code.
- Do not claim SOLVED if you have not actually verified your code reasons
  about the budget/constraint explicitly.
""".strip()


def _strip_thinking(text: str) -> str:
    # Reasoning models (e.g. nemotron, qwen3.5 with thinking on) prefix their
    # reply with a <think>...</think> chain-of-thought block that often
    # contains its own draft/example JSON with braces -- naive first-'{'/
    # last-'}' extraction then spans from a brace inside the thinking block
    # all the way to the real answer's closing brace, grabbing prose in
    # between and failing to parse. Drop everything through the last
    # </think> (if any) before extracting JSON.
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx != -1 else text


def _extract_json_block(text: str) -> Optional[str]:
    text = _strip_thinking(text).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return None


def parse_decision(raw_text: str) -> Optional[FinalDecision]:
    candidates = [raw_text.strip()]
    block = _extract_json_block(raw_text)
    if block:
        candidates.append(block)
    for cand in candidates:
        try:
            data = json.loads(cand)
            return FinalDecision(**data)
        except Exception:  # noqa: BLE001
            continue
    return None


def call_for_decision(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[Optional[FinalDecision], str, dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text, usage = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    decision = parse_decision(text)
    if decision is None:
        # one repair attempt
        repair_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                "Your previous reply was not valid JSON matching the required "
                "schema. Reply again with ONLY the corrected JSON object."
            )},
        ]
        text2, usage2 = client.chat(repair_messages, temperature=1.0, max_tokens=max_tokens)
        decision = parse_decision(text2)
        usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
        if decision is not None:
            text = text2
    return decision, text, usage


def build_system_prompt() -> str:
    return (
        "You are a careful coding agent. You will be given a programming "
        "task that may be feasible, feasible-but-constrained, or logically "
        "infeasible on at least one valid input. Think step by step "
        "internally, but your final reply must be ONLY the JSON object "
        "described below.\n\n" + DECISION_SCHEMA_INSTRUCTIONS
    )


def build_review_prompt(variant_prompt: str, own_previous_json: str) -> str:
    return (
        f"{variant_prompt}\n\n---\n"
        "This is a self-review round. Your previous answer was:\n"
        f"{own_previous_json}\n\n"
        "Re-examine it for correctness (functional correctness AND the "
        "stated constraint/budget). If you find an error, correct it. "
        "Reply again with ONLY the JSON object."
    )


def build_team_review_prompt(variant_prompt: str, own_previous_json: str, peer_summaries: list[str]) -> str:
    peers_block = "\n".join(f"- Peer {i+1}: {s}" for i, s in enumerate(peer_summaries))
    return (
        f"{variant_prompt}\n\n---\n"
        "This is a team discussion round. Your previous answer was:\n"
        f"{own_previous_json}\n\n"
        "Here is what your teammates concluded (their decision, reason, and "
        "any evidence ids -- NOT their full code):\n"
        f"{peers_block}\n\n"
        "Reconsider your answer in light of your teammates' reasoning. You "
        "may change your mind, but only if their stated reasoning actually "
        "supports it -- do not switch just because others disagree with you. "
        "Reply again with ONLY the JSON object."
    )


AGGREGATOR_SYSTEM_PROMPT = (
    "You are the final aggregator for a team of agents that each attempted "
    "the same task. You do NOT solve the task yourself, run any tools, or "
    "do additional search. You ONLY review the team's final submissions "
    "and select the team's single final answer.\n\n" + DECISION_SCHEMA_INSTRUCTIONS
)


def build_aggregator_prompt(variant_prompt: str, submissions: list[dict]) -> str:
    lines = []
    for i, s in enumerate(submissions):
        has_code = "yes" if s.get("solution_code") else "no"
        lines.append(
            f"- Agent {i + 1}: decision={s['decision']}, has_code={has_code}, "
            f"p_feasible={s['p_feasible']:.2f}, evidence_ids={s['evidence_ids']}, "
            f"conflict_constraint_ids={s['conflict_constraint_ids']}\n"
            f"  reason: {s['reason'][:300]}"
        )
    submissions_block = "\n".join(lines)
    return (
        f"Original task:\n{variant_prompt}\n\n---\n"
        f"Team submissions:\n{submissions_block}\n\n"
        "Select the team's final decision. If selecting SOLVED, copy the "
        "solution_code EXACTLY from whichever agent's submission you judge "
        "most likely correct (do not write new code yourself, and do not "
        "merge/edit code from multiple agents). Reply with ONLY the JSON "
        "object."
    )
