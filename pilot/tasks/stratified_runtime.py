"""Converts the validated stratified-dataset constructions
(tasks/stratified_dataset_final.jsonl -- 103 LCB/APPS-derived C1-C4
problems, each independently verified in stratified_verify.py) into
TaskVariant objects the pilot harness can actually run: one F1 (feasible),
F2 (feasible-but-constrained), F3 (certified infeasible) variant per
problem, same paired-variant protocol as the hand-built families in
registry.py.

Design choices, made explicit because they're not forced by the data:

- C1/C3/C4 constructions already encode all three tiers (f1/f2/f3 fields,
  or num_registers_f1/f2/f3) -- verified in Stage 2, reused as-is here.
- C2 constructions only define ONE state machine + oracle_code; the
  verifier computed the exact minimum op count (derived_min_ops) via BFS.
  F1/F2/F3 budgets are DERIVED from that single number here, the same way
  adj_swap_sort/cyclic_rotation_budget derive their budgets from a single
  inversion-count/target_idx anchor: F1 = unlimited, F2 = exactly
  derived_min_ops, F3 = derived_min_ops - 1.
- C3's concrete test stream(s) are generated here (not stored in the
  construction): one deterministic pseudo-random stream for F1/F2
  (functional-correctness check only, matching the hand-built families'
  single-instance F1/F2 grading), and the FULL enumerated domain for F3
  (required for the pigeonhole argument's exhaustive check, same as
  bitvector_recall/stream_popcount_recall) -- reusing stratified_verify's
  own enumeration cap (50000).
- The method name on the C3 stream object is kept as `next_bit()` (even
  though the alphabet may be non-binary) purely so the existing
  bitvector_recall_ast_check/bitvector_recall_taint_check in registry.py
  -- which key off that exact method name -- are reusable verbatim rather
  than duplicated.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Optional

from tasks.registry import TaskVariant

ROOT = Path(__file__).parent.parent
TASKS = ROOT / "tasks"

_MAX_C3_ENUM = 50000


def _load_problem_texts() -> dict[tuple[str, int], tuple[str, str]]:
    """(source, source_id) -> (title, problem_text)."""
    out = {}
    lcb = json.load(open(TASKS / "stratified_lcb_full.json"))
    for r in lcb:
        out[("livecodebench", r["question_id"])] = (r["title"], r["content"])
    apps = json.load(open(TASKS / "stratified_apps_full.json"))
    for r in apps:
        out[("apps", r["id"])] = (f"apps_{r['id']}", r["question"])
    return out


_PROBLEM_TEXTS = _load_problem_texts()


def _problem_text(source: str, source_id) -> str:
    key = (source, source_id)
    if key not in _PROBLEM_TEXTS:
        return "(original problem statement unavailable)"
    _, text = _PROBLEM_TEXTS[key]
    return text[:4000]


# ---------------------------------------------------------------------------
# C1 -- output/semantic: base problem + an extra predicate on the output
# ---------------------------------------------------------------------------

def _c1_prompt(problem_text: str, input_val, extra_code: str, feasible: bool) -> str:
    infeasible_note = (
        "" if feasible else
        "\n\nIt is possible that NO output satisfies both ordinary correctness "
        "and this extra predicate on this specific input -- do not force a "
        "guess if you determine that is the case."
    )
    return (
        f"{problem_text}\n\n"
        "Write a Python function `solve(input)` that returns the correct "
        "answer to the problem above for the given `input`.\n\n"
        f"The specific input to solve is:\n{input_val!r}\n\n"
        "In addition to being a correct answer to the original problem, your "
        "returned output must ALSO satisfy this extra Python predicate on "
        "this input (an additional checker beyond ordinary correctness):\n\n"
        f"```python\n{extra_code}\n```"
        f"{infeasible_note}"
    )


def _make_c1_variants(row: dict) -> list[TaskVariant]:
    problem_text = _problem_text(row["source"], row["source_id"])
    c = row["construction"]
    out = []
    for cat in ("F1", "F2", "F3"):
        spec = c[cat.lower()]
        feasible = cat != "F3"
        out.append(TaskVariant(
            family_id="stratified_c1",
            category=cat,
            primary_constraint_type="C1",
            variant_id=f"stratified_c1_{row['source']}_{row['source_id']}_{cat}",
            prompt=_c1_prompt(problem_text, spec["input"], spec["extra_constraint_code"], feasible),
            params={
                "oracle_code": row["oracle_code"],
                "extra_constraint_code": spec["extra_constraint_code"],
                "input": spec["input"],
            },
            feasible=feasible,
            infeasibility_certificate=(
                None if feasible else
                "Independently verified: no output can satisfy both base correctness "
                "and the stated extra predicate on this input (gpt-5.4 construction, "
                "re-executed and checked by stratified_verify.verify_c1)."
            ),
            secondary_constraint_tags=["stratified", row["source"]],
        ))
    return out


# ---------------------------------------------------------------------------
# C2 -- operation budget: generic Workspace(peek/apply_op) + derived budget
# ---------------------------------------------------------------------------

def _c2_prompt(problem_text: str, apply_op_code: str, peek_code: str, goal_code: str,
               op_arity: int, budget: Optional[int]) -> str:
    budget_text = (
        "There is NO budget on the number of apply_op() calls for this variant."
        if budget is None else
        f"Write a function that reaches the goal using AT MOST {budget} calls to workspace.apply_op()."
    )
    return (
        f"{problem_text}\n\n"
        "You must solve this problem through a restricted interactive interface "
        "rather than by returning a value directly. You are given `workspace`, "
        "an object with:\n"
        "  - workspace.peek(i) -> value   (read-only observation of the current state at position i)\n"
        f"  - workspace.apply_op(*args) -> None   (the ONLY state-changing operation; "
        f"takes {op_arity} integer argument(s))\n\n"
        "Here is exactly what apply_op and peek do (read this code -- it IS the "
        "interface, not just a description of it) and what a solved state looks "
        "like:\n\n"
        f"```python\n{peek_code}\n\n{apply_op_code}\n\n{goal_code}\n```\n\n"
        "Write a Python function `solve(workspace)` that calls workspace.apply_op(...) "
        "(and workspace.peek(...) as needed) to drive the state to a point where "
        f"goal_reached(state, input) would be True.\n\n{budget_text}\n\n"
        "Do not use any other method to inspect or mutate state; only peek and "
        "apply_op exist."
    )


def _make_c2_variants(row: dict) -> list[TaskVariant]:
    problem_text = _problem_text(row["source"], row["source_id"])
    c = row["construction"]
    verification = row.get("verification") or {}
    min_ops = verification.get("derived_min_ops")
    out = []
    budgets = {"F1": None, "F2": min_ops, "F3": (min_ops - 1) if min_ops is not None else None}
    for cat in ("F1", "F2", "F3"):
        budget = budgets[cat]
        feasible = cat != "F3"
        out.append(TaskVariant(
            family_id="stratified_c2",
            category=cat,
            primary_constraint_type="C2",
            variant_id=f"stratified_c2_{row['source']}_{row['source_id']}_{cat}",
            prompt=_c2_prompt(problem_text, c["apply_op_code"], c["peek_code"], c["goal_code"], c["op_arity"], budget),
            params={
                "init_state_code": c["init_state_code"],
                "apply_op_code": c["apply_op_code"],
                "peek_code": c["peek_code"],
                "goal_code": c["goal_code"],
                "state_input": c["state_input"],
                "op_arity": c["op_arity"],
                "budget": budget,
                "min_ops": min_ops,
            },
            feasible=feasible,
            infeasibility_certificate=(
                None if feasible else
                f"Independently verified via BFS: the exact minimum number of apply_op "
                f"calls needed to reach any goal state is {min_ops}; the declared budget "
                f"{budget} is one less, so no sequence of <= budget calls can succeed."
            ),
            secondary_constraint_tags=["stratified", row["source"]],
        ))
    return out


# ---------------------------------------------------------------------------
# C3 -- register/state budget: stream.next_bit() + RegisterBank, derived streams
# ---------------------------------------------------------------------------

def _c3_prompt(problem_text: str, stream_to_input_code: str, oracle_code: str,
               stream_length: int, alphabet_size: int, num_registers: int, bits_per_register: int) -> str:
    total_states = 2 ** (num_registers * bits_per_register)
    alphabet_note = (
        "each a bit (0 or 1)" if alphabet_size == 2 else
        f"each an integer in [0, {alphabet_size - 1}]"
    )
    return (
        f"{problem_text}\n\n"
        f"A hidden stream of exactly {stream_length} symbols ({alphabet_note}) will be "
        "revealed one at a time through `stream.next_bit()` (single pass -- once called, "
        "that symbol is gone; it returns None once exhausted). The stream encodes the "
        "input to the problem above via this mapping:\n\n"
        f"```python\n{stream_to_input_code}\n\n{oracle_code}\n```\n\n"
        f"You also have `registers`, a RegisterBank with EXACTLY {num_registers} "
        f"register(s), each able to hold an integer in [0, {2 ** bits_per_register - 1}] via "
        f"registers.set(i, value) / registers.get(i) (0 <= i < {num_registers}). This gives "
        f"you {num_registers * bits_per_register} total bits of usable memory "
        f"({total_states} distinguishable states), and NOTHING else may be retained (no "
        "lists, no extra data structures, no plain variables holding information derived "
        "from the stream) -- registers are your only memory.\n\n"
        "Write a Python function `solve(stream, registers)` that reads all "
        f"{stream_length} symbols from the stream (calling next_bit() exactly "
        f"{stream_length} times, in order), reconstructs the problem's input via the "
        "stream_to_input mapping shown above, and returns `oracle(that_input)` -- i.e. the "
        "correct answer to the original problem above -- using only what you stored in "
        "`registers` to compute it."
    )


def _gen_c3_streams(stream_length: int, alphabet_size: int, seed_key: str):
    rng = random.Random(seed_key)
    f1f2_stream = [rng.randrange(alphabet_size) for _ in range(stream_length)]
    total = alphabet_size ** stream_length
    if total <= _MAX_C3_ENUM:
        f3_domain = [list(c) for c in itertools.product(range(alphabet_size), repeat=stream_length)]
    else:
        f3_domain = [[rng.randrange(alphabet_size) for _ in range(stream_length)] for _ in range(_MAX_C3_ENUM)]
    return f1f2_stream, f3_domain


def _make_c3_variants(row: dict) -> list[TaskVariant]:
    problem_text = _problem_text(row["source"], row["source_id"])
    c = row["construction"]
    stream_length, alphabet_size = c["stream_length"], c["alphabet_size"]
    f1f2_stream, f3_domain = _gen_c3_streams(stream_length, alphabet_size, f"{row['source']}_{row['source_id']}")
    out = []
    for cat in ("F1", "F2", "F3"):
        nreg, bpr = c[f"num_registers_{cat.lower()}"], c[f"bits_per_register_{cat.lower()}"]
        feasible = cat != "F3"
        test_domain = f3_domain if cat == "F3" else [f1f2_stream]
        verification = row.get("verification") or {}
        required = verification.get("derived_required_outputs")
        out.append(TaskVariant(
            family_id="stratified_c3",
            category=cat,
            primary_constraint_type="C3",
            variant_id=f"stratified_c3_{row['source']}_{row['source_id']}_{cat}",
            prompt=_c3_prompt(problem_text, c["stream_to_input_code"], row["oracle_code"],
                               stream_length, alphabet_size, nreg, bpr),
            params={
                "stream_to_input_code": c["stream_to_input_code"],
                "oracle_code": row["oracle_code"],
                "num_registers": nreg,
                "bits_per_register": bpr,
                "test_domain": test_domain,
            },
            feasible=feasible,
            infeasibility_certificate=(
                None if feasible else
                f"Independently verified by brute force: {required} distinct outputs are "
                f"required across the input domain, but only {2 ** (nreg * bpr)} states are "
                "representable with the declared register capacity -- a pigeonhole violation."
            ),
            secondary_constraint_tags=["stratified", row["source"]],
        ))
    return out


# ---------------------------------------------------------------------------
# C4 -- information/access: _PeekView over a visible prefix only
# ---------------------------------------------------------------------------

def _c4_prompt(problem_text: str, visible: list, total_length: int, guarantee_text: Optional[str]) -> str:
    guarantee_block = f"\n\nYou are given this additional guarantee: {guarantee_text}\n" if guarantee_text else ""
    return (
        f"{problem_text}\n\n"
        f"The input to this problem is a list of length {total_length}, but you do NOT "
        "have direct access to it. Instead you are given `view`, an object with:\n"
        f"  - view.peek(i) -> value   (only valid for 0 <= i < {len(visible)}; any other "
        "index raises an error -- there is no way to see the remaining elements)\n"
        f"  - view.visible_count -> int (={len(visible)})\n"
        f"  - view.total_length -> int (={total_length})\n\n"
        f"The visible prefix is: {visible!r}{guarantee_block}\n"
        "Write a Python function `solve(view)` that returns the correct answer to the "
        "problem above for the full (partially hidden) input."
    )


def _make_c4_variants(row: dict) -> list[TaskVariant]:
    problem_text = _problem_text(row["source"], row["source_id"])
    c = row["construction"]
    visible_k = c["visible_k"]
    out = []
    for cat, input_val, guarantee in (
        ("F1", c["f1_input"], None),
        ("F2", c["f2_input"], c.get("f2_guarantee_text")),
        ("F3", c["f3_input_a"], None),
    ):
        feasible = cat != "F3"
        out.append(TaskVariant(
            family_id="stratified_c4",
            category=cat,
            primary_constraint_type="C4",
            variant_id=f"stratified_c4_{row['source']}_{row['source_id']}_{cat}",
            prompt=_c4_prompt(problem_text, input_val[:visible_k], len(input_val), guarantee),
            params={
                "oracle_code": row["oracle_code"],
                "input": input_val,
                "visible_k": visible_k,
            },
            feasible=feasible,
            infeasibility_certificate=(
                None if feasible else
                "Independently verified: an adversarial alternative input sharing the "
                "identical visible prefix yields a different correct answer, so no function "
                "of only the visible prefix can be correct on both (stratified_verify.verify_c4)."
            ),
            secondary_constraint_tags=["stratified", row["source"]],
        ))
    return out


_BUILDERS = {"C1": _make_c1_variants, "C2": _make_c2_variants, "C3": _make_c3_variants, "C4": _make_c4_variants}


def build_stratified_families(limit_per_type: Optional[int] = None) -> dict[str, list[TaskVariant]]:
    """Returns {"stratified_c1": [...], "stratified_c2": [...], ...}, each a
    flat list of F1/F2/F3 TaskVariants for every problem of that type."""
    rows = [json.loads(l) for l in open(TASKS / "stratified_dataset_final.jsonl")]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["constraint_type"], []).append(r)

    families = {}
    for ctype, builder in _BUILDERS.items():
        problem_rows = by_type.get(ctype, [])
        if limit_per_type is not None:
            problem_rows = problem_rows[:limit_per_type]
        variants = []
        for row in problem_rows:
            variants.extend(builder(row))
        families[f"stratified_{ctype.lower()}"] = variants
    return families


if __name__ == "__main__":
    fams = build_stratified_families(limit_per_type=2)
    for k, v in fams.items():
        print(f"{k}: {len(v)} variants")
        print("  sample prompt (truncated):")
        print("  " + v[0].prompt[:300].replace("\n", "\n  "))
        print()
