"""Plumbing smoke test for the generic stratified evaluators (sandbox.py's
evaluate_stratified_c1..c4) -- NOT a model run. Uses a "perfect/cheating"
test agent per type (built with full white-box knowledge of the
construction, exactly like stratified_verify.py's own checks) to confirm:
(a) no unhandled crash anywhere in the dynamic-exec/sandbox wiring,
(b) F1/F2 are actually achievable (PASS) by a correct solver,
(c) F3's infeasibility is actually enforced (a forced "solve it anyway"
    attempt gets WRONG_ANSWER/CONSTRAINT_VIOLATION, never PASS).
Run from pilot/: python3 -m tasks.stratified_runtime_selftest
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import evaluate  # noqa: E402
from tasks.stratified_runtime import build_stratified_families  # noqa: E402

N_PER_TYPE = 3


# ---- C1: hardcode the known-correct oracle output ----

def agent_c1(variant) -> str:
    p = variant.params
    ns: dict = {}
    exec(compile(p["oracle_code"], "<ref>", "exec"), ns)
    expected = ns["oracle"](p["input"])
    return f"def solve(input):\n    return {expected!r}\n"


# ---- C2: BFS offline (full, unrestricted) to find the real op path ----

def _bfs_path(init_code, apply_code, goal_code, state_input, op_arity):
    ns: dict = {}
    for code in (init_code, apply_code, goal_code):
        exec(compile(code, "<ref>", "exec"), ns)
    start = ns["init_state"](state_input)

    def key(s):
        return json.dumps(s) if not isinstance(s, (int, float, str, bool)) else s

    def arg_ranges(s):
        n = len(s) if hasattr(s, "__len__") else 64
        return itertools.product(range(max(n, 1)), repeat=op_arity)

    seen = {key(start)}
    frontier = deque([(start, [])])
    for _ in range(200000):
        if not frontier:
            return None
        state, path = frontier.popleft()
        if ns["goal_reached"](state, state_input):
            return path
        for args in arg_ranges(state):
            nxt = json.loads(json.dumps(state))
            try:
                ns["apply_op"](nxt, *args)
            except Exception:
                continue
            k = key(nxt)
            if k not in seen:
                seen.add(k)
                frontier.append((nxt, path + [args]))
    return None


def agent_c2(variant, path) -> str:
    args_repr = repr(path)
    return (
        f"def solve(workspace):\n"
        f"    for a in {args_repr}:\n"
        f"        workspace.apply_op(*a)\n"
    )


# ---- C4: hardcode the known-correct oracle output ----

def agent_c4(variant) -> str:
    p = variant.params
    ns: dict = {}
    exec(compile(p["oracle_code"], "<ref>", "exec"), ns)
    expected = ns["oracle"](p["input"])
    return f"def solve(view):\n    return {expected!r}\n"


def main():
    families = build_stratified_families(limit_per_type=N_PER_TYPE)
    summary = {"C1": [], "C2": [], "C3": [], "C4": []}

    for variant in families["stratified_c1"]:
        code = agent_c1(variant)
        res = evaluate(variant, code)
        summary["C1"].append((variant.variant_id, variant.category, variant.feasible, res.verdict))

    for variant in families["stratified_c2"]:
        p = variant.params
        path = _bfs_path(p["init_state_code"], p["apply_op_code"], p["goal_code"], p["state_input"], p["op_arity"])
        if path is None:
            summary["C2"].append((variant.variant_id, variant.category, variant.feasible, "NO_PATH_FOUND"))
            continue
        code = agent_c2(variant, path)
        res = evaluate(variant, code)
        summary["C2"].append((variant.variant_id, variant.category, variant.feasible, res.verdict, f"path_len={len(path)} budget={p['budget']}"))

    for variant in families["stratified_c3"]:
        # No generic register-packing agent (see module docstring discussion);
        # just confirm no crash with an empty/no-op submission that still
        # exercises the AST/taint checks and the subprocess wiring.
        res = evaluate(variant, "def solve(stream, registers):\n    for _ in range(10000):\n        pass\n    return None\n")
        summary["C3"].append((variant.variant_id, variant.category, variant.feasible, res.verdict, "(no-op agent, crash check only)"))

    for variant in families["stratified_c4"]:
        code = agent_c4(variant)
        res = evaluate(variant, code)
        summary["C4"].append((variant.variant_id, variant.category, variant.feasible, res.verdict))

    for ctype, rows in summary.items():
        print(f"=== {ctype} ===")
        for row in rows:
            print(" ", row)


if __name__ == "__main__":
    main()
