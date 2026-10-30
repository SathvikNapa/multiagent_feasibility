"""One-off remediation for the C4 F1 prefix-determinism bug found during
the harness-validation pilot: re-verify every existing C4 construction with
the FIXED verify_c4 (which now actually checks F1's answer is invariant
under the hidden tail, not just that oracle() runs), then send everything
that fails through one retry-construction round via gpt-5.4 + Batch API
with the SPECIFIC counterexample as feedback, re-verify, and write the
final accepted/rejected sets.
"""
from __future__ import annotations

import json
from pathlib import Path

from openai_batch import run_batch
from stratified_construct import build_messages, _extract_json
from stratified_verify import verify_item

ROOT = Path(__file__).parent.parent
TASKS = ROOT / "tasks"
MODEL = "gpt-5.4"
MAX_TOKENS = 10000

LCB_FULL = {r["question_id"]: r for r in json.load(open(TASKS / "stratified_lcb_full.json"))}
APPS_FULL = {r["id"]: r for r in json.load(open(TASKS / "stratified_apps_full.json"))}


def problem_text(source: str, source_id) -> tuple[str, str]:
    if source == "livecodebench":
        r = LCB_FULL[source_id]
        return r["title"], r["content"]
    r = APPS_FULL[source_id]
    return f"apps_{source_id}", r["question"]


def load_all_c4() -> list[dict]:
    rows = []
    for fn in ("stratified_final_C4.jsonl", "stratified_final_C4_apps_new.jsonl"):
        path = TASKS / fn
        if path.exists():
            rows.extend(json.loads(l) for l in open(path))
    return rows


def cid(r: dict) -> str:
    return f"{r['source']}:{r['id']}"


def main():
    rows = load_all_c4()
    print(f"[C4-fix] {len(rows)} existing C4 constructions loaded")

    verified = {cid(r): verify_item(r) for r in rows}
    passed = {k for k, v in verified.items() if v.get("pass")}
    failed = [r for r in rows if cid(r) not in passed]
    print(f"[C4-fix] under FIXED verifier: {len(passed)}/{len(rows)} still pass, {len(failed)} fail -> retrying")

    by_id = {cid(r): r for r in rows}
    retry_requests = []
    for r in failed:
        k = cid(r)
        f1_check = verified[k]["checks"].get("f1", {})
        feedback = (
            f"Your previous F1 construction was rejected: {f1_check.get('detail', 'not prefix-determinable')}. "
            "The F1 answer must be determinable from ONLY the visible prefix (the first visible_k elements) -- "
            "it must be the SAME regardless of what the hidden portion (everything after index visible_k-1) "
            "actually is. Pick a DIFFERENT f1_input (and/or a different/larger visible_k) where this genuinely "
            "holds -- e.g. an input where the answer only depends on properties of the visible prefix itself "
            "(its own elements, length, or a property like 'first element equals a fixed target'), not on "
            "anything about the rest of the array. Before answering, mentally substitute a few different values "
            "for the hidden portion while keeping the visible prefix fixed, and confirm the oracle's answer "
            "truly does not change. F2 and F3 may be reused or revised as needed to stay consistent."
        )
        title, text = problem_text(r["source"], r["id"])
        retry_requests.append({
            "custom_id": k, "model": MODEL, "max_completion_tokens": MAX_TOKENS,
            "messages": build_messages("C4", title, text, feedback=feedback),
        })

    retry_out = run_batch(retry_requests, description=f"C4 F1 prefix-determinism fix retry (n={len(retry_requests)})")

    constructed = dict(by_id)
    for r in failed:
        k = cid(r)
        out = retry_out[k]
        if out["error"]:
            constructed[k] = {**r, "constraint_type": "API_ERROR", "_err": out["error"]}
            continue
        try:
            parsed = _extract_json(out["content"])
        except (ValueError, json.JSONDecodeError) as e:
            constructed[k] = {**r, "constraint_type": "PARSE_ERROR", "_raw": out["content"][:1000], "_err": str(e)}
            continue
        constructed[k] = {"source": r["source"], "id": r["id"], "title": r["title"], **parsed}
        verified[k] = verify_item(constructed[k])

    final_passed = {k for k, v in verified.items() if v.get("pass")}
    print(f"[C4-fix] after retry: {len(final_passed)}/{len(rows)} pass ({len(final_passed) - len(passed)} newly fixed of {len(failed)} retried)")

    final_path = TASKS / "stratified_final_C4.jsonl"
    rejects_path = TASKS / "stratified_rejects_C4_f1fix.jsonl"
    with open(final_path, "w") as fout, open(rejects_path, "w") as rout:
        for k, c in constructed.items():
            record = {**c, "_verification": verified[k]}
            if k in final_passed:
                fout.write(json.dumps(record) + "\n")
            else:
                rout.write(json.dumps(record) + "\n")

    # The old _apps_new file's accepted rows are now folded into the
    # rewritten stratified_final_C4.jsonl above; remove it so
    # consolidate_stratified.py doesn't double-count.
    old_apps_new = TASKS / "stratified_final_C4_apps_new.jsonl"
    if old_apps_new.exists():
        old_apps_new.unlink()

    print(f"[C4-fix] final: {len(final_passed)} accepted -> {final_path}")
    print(f"[C4-fix] rejected: {len(rows) - len(final_passed)} -> {rejects_path}")
    for k in sorted(constructed):
        print(f"  {'PASS' if k in final_passed else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
