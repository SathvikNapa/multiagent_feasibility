"""Full Stage 2 pipeline for one constraint type at a time: batch-construct
the top-N classified candidates, independently verify each, batch-retry any
failures ONCE with specific verifier feedback, re-verify, and save only the
items that pass to the validated dataset. Run type by type (per the agreed
plan) -- finish and review one type fully before starting the next.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai_batch import run_batch
from stratified_construct import SYSTEM_PROMPT_TEMPLATE, SCHEMAS, build_messages, _extract_json
from stratified_verify import verify_item

ROOT = Path(__file__).parent.parent
MODEL = "gpt-5.4-mini"  # overridden per-run via --model
MAX_TOKENS = 10000


def load_candidates(ctype: str, n: int) -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / "tasks" / "stratified_classification.jsonl")]
    lcb = {r["question_id"]: r for r in json.load(open(ROOT / "tasks" / "stratified_lcb_full.json"))}
    apps = {r["id"]: r for r in json.load(open(ROOT / "tasks" / "stratified_apps_full.json"))}
    cands = [r for r in rows if r["constraint_type"] == ctype]
    cands = sorted(cands, key=lambda r: -r["confidence"])[:n]
    out = []
    for item in cands:
        if item["source"] == "livecodebench":
            src = lcb[item["id"]]
            title, text = src["title"], src["content"]
        else:
            src = apps[item["id"]]
            title, text = f"apps_{item['id']}", src["question"]
        out.append({"source": item["source"], "id": item["id"], "title": title, "text": text})
    return out


def cid(item: dict) -> str:
    return f"{item['source']}:{item['id']}"


def parse_result(item: dict, raw_content: str, err: str | None) -> dict:
    if err:
        return {"source": item["source"], "id": item["id"], "title": item["title"], "constraint_type": "API_ERROR", "_err": err}
    try:
        parsed = _extract_json(raw_content)
    except (ValueError, json.JSONDecodeError) as e:
        return {"source": item["source"], "id": item["id"], "title": item["title"], "constraint_type": "PARSE_ERROR", "_raw": raw_content[:1000], "_err": str(e)}
    return {"source": item["source"], "id": item["id"], "title": item["title"], **parsed}


def verifier_feedback(verify_res: dict) -> str:
    parts = []
    for cat, c in (verify_res.get("checks") or {}).items():
        if not c.get("ok"):
            parts.append(f"{cat}: {c.get('detail', '')[:300]}")
    if verify_res.get("detail"):
        parts.append(verify_res["detail"][:300])
    return "; ".join(parts) or "unknown verification failure"


def run_for_type(ctype: str, n: int, model: str = MODEL, out_suffix: str = ""):
    candidates = load_candidates(ctype, n)
    print(f"[{ctype}] {len(candidates)} candidates loaded")

    requests = [{
        "custom_id": cid(it), "model": model, "max_completion_tokens": MAX_TOKENS,
        "messages": build_messages(ctype, it["title"], it["text"]),
    } for it in candidates]
    batch_out = run_batch(requests, description=f"stratified construct {ctype} (round 1, n={len(requests)}, model={model})")

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
                "custom_id": k, "model": model, "max_completion_tokens": MAX_TOKENS,
                "messages": build_messages(ctype, it["title"], it["text"], feedback=fb),
            })
        retry_out = run_batch(retry_requests, description=f"stratified construct {ctype} (retry, n={len(retry_requests)}, model={model})")
        for k in failed:
            it = by_id[k]
            r = retry_out[k]
            constructed[k] = parse_result(it, r["content"], r["error"])
            verified[k] = verify_item(constructed[k])
        retry_passed = {k for k in failed if verified[k].get("pass")}
        passed |= retry_passed
        print(f"[{ctype}] after retry: {len(passed)}/{len(candidates)} passed ({len(retry_passed)}/{len(failed)} retries succeeded)")

    out_dir = ROOT / "tasks"
    final_path = out_dir / f"stratified_final_{ctype}{out_suffix}.jsonl"
    rejects_path = out_dir / f"stratified_rejects_{ctype}{out_suffix}.jsonl"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["C1", "C2", "C3", "C4"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-suffix", default="", help="appended to output filenames, e.g. '_gpt54' to avoid overwriting a prior run")
    args = ap.parse_args()
    run_for_type(args.type, args.n, model=args.model, out_suffix=args.out_suffix)
