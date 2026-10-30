"""APPS-only extension of the Stage 2 construction pipeline: LiveCodeBench
and ImpossibleBench are frozen; this constructs the "constrained"/"impossible"
(F2/F3) tiers -- via the same gpt-5.4 + Batch API + independent-verification
pipeline as stratified_pipeline.py -- for every classified APPS problem NOT
already attempted in the earlier mixed LCB+APPS run. The "canonical" (F1)
tier for APPS needs no model call: it's the problem's own given solution,
attached directly from the cached APPS data.
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


def already_attempted(ctype: str) -> set[str]:
    ids = set()
    for fn in (f"stratified_final_{ctype}.jsonl", f"stratified_rejects_{ctype}.jsonl"):
        path = TASKS / fn
        if not path.exists():
            continue
        for l in open(path):
            r = json.loads(l)
            if r.get("source") == "apps":
                ids.add(r["id"])
    return ids


def load_new_apps_candidates(ctype: str) -> list[dict]:
    rows = [json.loads(l) for l in open(TASKS / "stratified_classification.jsonl")]
    apps = {r["id"]: r for r in json.load(open(TASKS / "stratified_apps_full.json"))}
    done = already_attempted(ctype)
    out = []
    for r in rows:
        if r["source"] != "apps" or r["constraint_type"] != ctype or r["id"] in done:
            continue
        src = apps[r["id"]]
        out.append({"source": "apps", "id": r["id"], "title": f"apps_{r['id']}", "text": src["question"],
                     "difficulty": src.get("difficulty")})
    return out


def cid(item: dict) -> str:
    return f"{item['source']}:{item['id']}"


def parse_result(item: dict, raw_content: str, err: str | None) -> dict:
    if err:
        return {"source": item["source"], "id": item["id"], "title": item["title"], "difficulty": item["difficulty"],
                "constraint_type": "API_ERROR", "_err": err}
    try:
        parsed = _extract_json(raw_content)
    except (ValueError, json.JSONDecodeError) as e:
        return {"source": item["source"], "id": item["id"], "title": item["title"], "difficulty": item["difficulty"],
                "constraint_type": "PARSE_ERROR", "_raw": raw_content[:1000], "_err": str(e)}
    return {"source": item["source"], "id": item["id"], "title": item["title"], "difficulty": item["difficulty"], **parsed}


def verifier_feedback(verify_res: dict) -> str:
    parts = []
    for cat, c in (verify_res.get("checks") or {}).items():
        if not c.get("ok"):
            parts.append(f"{cat}: {c.get('detail', '')[:300]}")
    if verify_res.get("detail"):
        parts.append(verify_res["detail"][:300])
    return "; ".join(parts) or "unknown verification failure"


def run_for_type(ctype: str):
    candidates = load_new_apps_candidates(ctype)
    print(f"[{ctype}] {len(candidates)} NEW apps candidates")
    if not candidates:
        print(f"[{ctype}] nothing new, skipping")
        return

    requests = [{
        "custom_id": cid(it), "model": MODEL, "max_completion_tokens": MAX_TOKENS,
        "messages": build_messages(ctype, it["title"], it["text"]),
    } for it in candidates]
    batch_out = run_batch(requests, description=f"apps-canonical construct {ctype} (round 1, n={len(requests)}, model={MODEL})")

    constructed = {}
    for it in candidates:
        r = batch_out[cid(it)]
        constructed[cid(it)] = parse_result(it, r["content"], r["error"])

    verified = {k: verify_item(v) for k, v in constructed.items()}
    passed = {k for k, v in verified.items() if v.get("pass")}
    failed = {k for k in constructed if k not in passed}
    print(f"[{ctype}] round 1: {len(passed)}/{len(candidates)} passed")

    if failed:
        by_id = {cid(it): it for it in candidates}
        retry_requests = []
        for k in failed:
            it = by_id[k]
            fb = verifier_feedback(verified[k])
            retry_requests.append({
                "custom_id": k, "model": MODEL, "max_completion_tokens": MAX_TOKENS,
                "messages": build_messages(ctype, it["title"], it["text"], feedback=fb),
            })
        retry_out = run_batch(retry_requests, description=f"apps-canonical construct {ctype} (retry, n={len(retry_requests)}, model={MODEL})")
        for k in failed:
            it = by_id[k]
            r = retry_out[k]
            constructed[k] = parse_result(it, r["content"], r["error"])
            verified[k] = verify_item(constructed[k])
        retry_passed = {k for k in failed if verified[k].get("pass")}
        passed |= retry_passed
        print(f"[{ctype}] after retry: {len(passed)}/{len(candidates)} passed ({len(retry_passed)}/{len(failed)} retries succeeded)")

    final_path = TASKS / f"stratified_final_{ctype}_apps_new.jsonl"
    rejects_path = TASKS / f"stratified_rejects_{ctype}_apps_new.jsonl"
    with open(final_path, "w") as fout, open(rejects_path, "w") as rout:
        for k, c in constructed.items():
            record = {**c, "_verification": verified[k]}
            if k in passed:
                fout.write(json.dumps(record) + "\n")
            else:
                rout.write(json.dumps(record) + "\n")

    print(f"[{ctype}] final: {len(passed)} accepted -> {final_path}")
    print(f"[{ctype}] rejected: {len(constructed) - len(passed)} -> {rejects_path}")
    for k in sorted(constructed):
        status = "PASS" if k in passed else "FAIL"
        print(f"  {status}  {k}  ({constructed[k].get('constraint_type')})")


if __name__ == "__main__":
    import sys
    types = sys.argv[1:] if len(sys.argv) > 1 else ["C1", "C2", "C3", "C4"]
    for ctype in types:
        run_for_type(ctype)
