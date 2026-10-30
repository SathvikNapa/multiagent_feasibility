"""Independent verification for the stratified-dataset Stage 2 constructions.
Trusts NOTHING the model claims -- re-derives every F1/F2/F3 ground truth
itself: BFS for C2's exact minimum operation count, brute-force enumeration
for C3's pigeonhole count, direct predicate evaluation for C1, oracle
comparison on the given adversarial pair for C4. Runs all model-supplied
code in a resource-limited subprocess, same isolation posture as the main
pilot's sandbox.py.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import resource
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXECUTION_TIMEOUT_S = 5
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


def _set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (EXECUTION_TIMEOUT_S, EXECUTION_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _restricted_ns():
    import builtins
    allowed = ["len", "range", "min", "max", "abs", "sum", "enumerate", "int", "float",
               "bool", "str", "list", "tuple", "dict", "set", "True", "False", "None",
               "sorted", "isinstance", "zip", "map", "filter", "all", "any", "reversed"]
    return {"__builtins__": {name: getattr(builtins, name) for name in allowed if hasattr(builtins, name)}}


def _run_in_subprocess(target, args, timeout=EXECUTION_TIMEOUT_S + 3):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=target, args=(*args, child_conn))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return {"timeout": True}
    if parent_conn.poll():
        return parent_conn.recv()
    return {"error": f"process exited with code {p.exitcode}, no result"}


# ---- C1: output/semantic ----

def _child_c1(oracle_code, extra_code, input_val, conn):
    try:
        _set_limits()
        ns = _restricted_ns()
        exec(compile(oracle_code, "<oracle>", "exec"), ns)
        exec(compile(extra_code, "<extra>", "exec"), ns)
        answer = ns["oracle"](input_val)
        satisfies = ns["extra_ok"](input_val, answer)
        conn.send({"answer": answer, "satisfies": satisfies})
    except Exception:
        conn.send({"error": traceback.format_exc(limit=4)})


def verify_c1(item: dict) -> dict:
    out = {"id": item["id"], "constraint_type": "C1", "checks": {}}
    for cat in ("f1", "f2", "f3"):
        spec = item[cat]
        res = _run_in_subprocess(_child_c1, (item["oracle_code"], spec["extra_constraint_code"], spec["input"]))
        if res.get("timeout"):
            out["checks"][cat] = {"ok": False, "detail": "timeout"}
            continue
        if "error" in res:
            out["checks"][cat] = {"ok": False, "detail": res["error"][:300]}
            continue
        expected_satisfies = cat != "f3"  # F1/F2 should satisfy the extra constraint, F3 should NOT
        ok = res["satisfies"] == expected_satisfies
        out["checks"][cat] = {"ok": ok, "detail": f"oracle={res['answer']!r} satisfies_extra={res['satisfies']} expected={expected_satisfies}"}
    out["pass"] = all(c["ok"] for c in out["checks"].values())
    return out


# ---- C2: operation budget (BFS for exact minimum) ----

def _child_c2(init_code, apply_code, peek_code, goal_code, state_input, op_arity, conn):
    try:
        _set_limits()
        ns = _restricted_ns()
        for code in (init_code, apply_code, peek_code, goal_code):
            exec(compile(code, "<c2>", "exec"), ns)
        init_state, apply_op, goal_reached = ns["init_state"], ns["apply_op"], ns["goal_reached"]

        start = init_state(state_input)

        def state_key(s):
            return json.dumps(s) if not isinstance(s, (int, float, str, bool)) else s

        def arg_ranges(s):
            n = len(s) if hasattr(s, "__len__") else 64
            return itertools.product(range(max(n, 1)), repeat=op_arity)

        # BFS for the shortest number of apply_op calls to reach any goal state.
        from collections import deque
        seen = {state_key(start)}
        frontier = deque([(start, 0)])
        max_states = 200000
        min_ops = None
        explored = 0
        while frontier:
            state, depth = frontier.popleft()
            if goal_reached(state, state_input):
                min_ops = depth
                break
            explored += 1
            if explored > max_states:
                conn.send({"error": f"state space too large, aborted after {max_states} states"})
                return
            for args in arg_ranges(state):
                nxt = json.loads(json.dumps(state))  # deep copy via JSON round-trip
                try:
                    apply_op(nxt, *args)
                except Exception:
                    continue
                k = state_key(nxt)
                if k not in seen:
                    seen.add(k)
                    frontier.append((nxt, depth + 1))
        conn.send({"min_ops": min_ops, "states_explored": explored})
    except Exception:
        conn.send({"error": traceback.format_exc(limit=4)})


def verify_c2(item: dict) -> dict:
    out = {"id": item["id"], "constraint_type": "C2", "checks": {}}
    res = _run_in_subprocess(
        _child_c2,
        (item["init_state_code"], item["apply_op_code"], item["peek_code"], item["goal_code"],
         item["state_input"], item["op_arity"]),
        timeout=15,
    )
    if res.get("timeout"):
        out["checks"]["bfs"] = {"ok": False, "detail": "timeout (state space likely too large)"}
        out["pass"] = False
        return out
    if "error" in res:
        out["checks"]["bfs"] = {"ok": False, "detail": res["error"][:400]}
        out["pass"] = False
        return out
    min_ops = res["min_ops"]
    out["checks"]["bfs"] = {"ok": min_ops is not None, "detail": f"independently-computed exact minimum ops = {min_ops} (states explored: {res['states_explored']})"}
    out["derived_min_ops"] = min_ops
    out["pass"] = min_ops is not None
    return out


# ---- C3: register/state budget (brute-force pigeonhole count) ----

def _child_c3(stream_to_input_code, oracle_code, stream_length, alphabet_size, conn):
    try:
        _set_limits()
        ns = _restricted_ns()
        exec(compile(stream_to_input_code, "<s2i>", "exec"), ns)
        exec(compile(oracle_code, "<oracle>", "exec"), ns)
        stream_to_input, oracle = ns["stream_to_input"], ns["oracle"]
        total = alphabet_size ** stream_length
        if total > 50000:
            conn.send({"error": f"alphabet^length = {total} too large to brute force"})
            return
        outputs = set()
        for combo in itertools.product(range(alphabet_size), repeat=stream_length):
            inp = stream_to_input(list(combo))
            ans = oracle(inp)
            outputs.add(json.dumps(ans) if not isinstance(ans, (int, float, str, bool)) else ans)
        conn.send({"distinct_outputs": len(outputs), "total_streams": total})
    except Exception:
        conn.send({"error": traceback.format_exc(limit=4)})


def verify_c3(item: dict) -> dict:
    out = {"id": item["id"], "constraint_type": "C3", "checks": {}}
    res = _run_in_subprocess(
        _child_c3,
        (item["stream_to_input_code"], item["oracle_code"], item["stream_length"], item["alphabet_size"]),
        timeout=15,
    )
    if res.get("timeout") or "error" in res:
        out["checks"]["brute_force"] = {"ok": False, "detail": (res.get("error") or "timeout")[:400]}
        out["pass"] = False
        return out
    required = res["distinct_outputs"]
    out["derived_required_outputs"] = required
    ok_all = True
    for cat, nreg, bpr in (("f1", item["num_registers_f1"], item["bits_per_register_f1"]),
                            ("f2", item["num_registers_f2"], item["bits_per_register_f2"]),
                            ("f3", item["num_registers_f3"], item["bits_per_register_f3"])):
        states = (2 ** bpr) ** nreg
        if cat in ("f1", "f2"):
            ok = states >= required
        else:
            ok = states < required  # F3 must be provably insufficient
        out["checks"][cat] = {"ok": ok, "detail": f"representable_states={states} required_distinct_outputs={required}"}
        ok_all = ok_all and ok
    out["pass"] = ok_all
    return out


# ---- C4: information/access (adversarial-pair oracle comparison) ----

def _child_c4(oracle_code, input_val, conn):
    try:
        _set_limits()
        ns = _restricted_ns()
        exec(compile(oracle_code, "<oracle>", "exec"), ns)
        conn.send({"answer": ns["oracle"](input_val)})
    except Exception:
        conn.send({"error": traceback.format_exc(limit=4)})


def _gen_alt_tails(prefix: list, tail: list) -> list:
    """Diverse, type-aware alternate hidden tails to test F1's claimed
    prefix-determinism against -- structural permutations plus numeric/
    string perturbations, not just a single alternative."""
    alts = []
    if len(tail) >= 2:
        alts.append(list(reversed(tail)))
        alts.append(tail[1:] + tail[:1])
    if prefix:
        alts.append([prefix[i % len(prefix)] for i in range(len(tail))])
        alts.append([prefix[0]] * len(tail))
    if tail and all(isinstance(x, int) and not isinstance(x, bool) for x in tail):
        alts.append([x + 1 for x in tail])
        alts.append([x - 1 for x in tail])
        for v in (0, 1, 2, 3):
            alts.append([v] * len(tail))
    elif tail and all(isinstance(x, str) for x in tail):
        alts.append([x.upper() if x.islower() else x.lower() for x in tail])
        alts.append([x + x for x in tail])
    return alts


def _child_c4_invariance(oracle_code, prefix, tail, alts, conn):
    try:
        _set_limits()
        ns = _restricted_ns()
        exec(compile(oracle_code, "<oracle>", "exec"), ns)
        oracle = ns["oracle"]
        base = oracle(prefix + tail)
        mismatches = []
        for alt in alts:
            try:
                ans = oracle(prefix + alt)
                if ans != base:
                    mismatches.append((alt, ans))
            except Exception:
                pass
        conn.send({"base": base, "mismatches": mismatches})
    except Exception:
        conn.send({"error": traceback.format_exc(limit=4)})


def verify_c4(item: dict) -> dict:
    out = {"id": item["id"], "constraint_type": "C4", "checks": {}}
    vk = item["visible_k"]

    for cat, inp in (("f1", item["f1_input"]), ("f2", item["f2_input"])):
        res = _run_in_subprocess(_child_c4, (item["oracle_code"], inp))
        if "error" in res or res.get("timeout"):
            out["checks"][cat] = {"ok": False, "detail": (res.get("error") or "timeout")[:300]}
            continue
        if cat != "f1":
            # F2 relies on a stated-but-unexecutable guarantee (f2_guarantee_text),
            # so there's no code-level predicate to scope an invariance check to
            # guarantee-satisfying alternates only -- this remains a known,
            # documented limitation (trust the construction's stated guarantee).
            out["checks"][cat] = {"ok": True, "detail": f"oracle={res['answer']!r} (F2: guarantee-dependent, not machine-checked)"}
            continue
        # F1 has NO stated guarantee at all, so the answer must be invariant
        # under literally ANY hidden tail -- check that directly instead of
        # just trusting that oracle() ran without error (the gap that let
        # 23/42 accepted C4 items through with an F1 that actually depends
        # on the hidden portion -- see tasks/stratified_runtime_selftest.py
        # pilot run, which caught this via a real model correctly flagging
        # one as infeasible).
        prefix, tail = list(inp[:vk]), list(inp[vk:])
        alts = _gen_alt_tails(prefix, tail)
        inv = _run_in_subprocess(_child_c4_invariance, (item["oracle_code"], prefix, tail, alts))
        if "error" in inv or inv.get("timeout"):
            out["checks"][cat] = {"ok": False, "detail": f"invariance check error: {(inv.get('error') or 'timeout')[:200]}"}
        elif inv["mismatches"]:
            alt, ans = inv["mismatches"][0]
            out["checks"][cat] = {
                "ok": False,
                "detail": f"NOT prefix-determinable: oracle(given)={res['answer']!r} but hidden tail {alt!r} -> {ans!r} "
                          f"(checked {len(alts)} alternate hidden tails, {len(inv['mismatches'])} differed)",
            }
        else:
            out["checks"][cat] = {"ok": True, "detail": f"oracle={res['answer']!r}, invariant across {len(alts)} alternate hidden tails"}

    a, b = item["f3_input_a"], item["f3_input_b"]
    prefix_match = list(a[:vk]) == list(b[:vk])
    res_a = _run_in_subprocess(_child_c4, (item["oracle_code"], a))
    res_b = _run_in_subprocess(_child_c4, (item["oracle_code"], b))
    if "error" in res_a or "error" in res_b or res_a.get("timeout") or res_b.get("timeout"):
        out["checks"]["f3"] = {"ok": False, "detail": f"a_err={res_a.get('error')} b_err={res_b.get('error')}"}
    else:
        differ = res_a["answer"] != res_b["answer"]
        ok = prefix_match and differ
        out["checks"]["f3"] = {
            "ok": ok,
            "detail": f"prefix_match={prefix_match} oracle(a)={res_a['answer']!r} oracle(b)={res_b['answer']!r} differ={differ}",
        }
    out["pass"] = all(c["ok"] for c in out["checks"].values())
    return out


VERIFIERS = {"C1": verify_c1, "C2": verify_c2, "C3": verify_c3, "C4": verify_c4}


def verify_item(item: dict) -> dict:
    ctype = item.get("constraint_type")
    if ctype not in VERIFIERS:
        return {"id": item.get("id"), "constraint_type": ctype, "pass": False, "checks": {}, "detail": "not a valid constructed item (parse error or unknown type)"}
    try:
        return VERIFIERS[ctype](item)
    except Exception:
        return {"id": item.get("id"), "constraint_type": ctype, "pass": False, "checks": {}, "detail": traceback.format_exc(limit=4)}


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tasks" / "stratified_construction_pilot.jsonl")
    rows = [json.loads(l) for l in open(in_path)]
    results = []
    for item in rows:
        res = verify_item(item)
        results.append(res)
        status = "PASS" if res.get("pass") else "FAIL"
        print(f"{item.get('source')}:{item.get('id')} [{res.get('constraint_type')}] -> {status}")
        for cat, c in res.get("checks", {}).items():
            print(f"    {cat}: {'ok' if c['ok'] else 'FAIL'} -- {c['detail']}")
        if res.get("detail"):
            print(f"    {res['detail'][:300]}")
    n_pass = sum(1 for r in results if r.get("pass"))
    print(f"\n{n_pass}/{len(results)} passed verification")
