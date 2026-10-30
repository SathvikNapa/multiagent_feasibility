"""Consolidate the APPS-only feasibility dataset: every accepted C1-C4
construction sourced from APPS (both the original mixed LCB+APPS run and
the apps_canonical_pipeline.py extension), each tagged with its APPS-native
difficulty tier and its own canonical (reference) solution -- "canonical"
(correct, unconstrained) + "constrained"/"impossible" (the constructed F2/F3
tiers, independently verified) rather than relying on a model-drafted F1.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TASKS = ROOT / "tasks"

apps_full = {r["id"]: r for r in json.load(open(TASKS / "stratified_apps_full.json"))}


def canonical_solution(apps_id: int) -> str | None:
    sols = apps_full[apps_id].get("solutions")
    if not sols:
        return None
    try:
        parsed = json.loads(sols) if isinstance(sols, str) else sols
    except (ValueError, json.JSONDecodeError):
        return sols if isinstance(sols, str) else None
    if isinstance(parsed, list) and parsed:
        return parsed[0]
    return None


rows = []
seen_ids = set()
for ctype in ("C1", "C2", "C3", "C4"):
    sources = [TASKS / f"stratified_final_{ctype}.jsonl"]
    extra = TASKS / f"stratified_final_{ctype}_apps_new.jsonl"
    if extra.exists():
        sources.append(extra)

    for path in sources:
        for l in open(path):
            r = json.loads(l)
            if r.get("source") != "apps":
                continue
            key = (ctype, r["id"])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            rows.append({
                "dataset": "apps_canonical",
                "constraint_type": ctype,
                "apps_id": r["id"],
                "title": r.get("title"),
                "difficulty": r.get("difficulty") or apps_full[r["id"]].get("difficulty"),
                "canonical_solution": canonical_solution(r["id"]),
                "oracle_code": r.get("oracle_code"),
                "construction": {k: v for k, v in r.items()
                                  if k not in ("source", "id", "title", "oracle_code", "constraint_type",
                                               "difficulty", "_verification")},
                "verification": r.get("_verification"),
            })

out_path = TASKS / "apps_canonical_dataset.jsonl"
with open(out_path, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

from collections import Counter
by_type = Counter(r["constraint_type"] for r in rows)
by_diff = Counter(r["difficulty"] for r in rows)
by_type_diff = Counter((r["constraint_type"], r["difficulty"]) for r in rows)

print(f"wrote {out_path}: {len(rows)} total APPS-sourced validated constructions")
print(f"by type: {dict(by_type)}")
print(f"by difficulty: {dict(by_diff)}")
print("by type x difficulty:")
for ctype in ("C1", "C2", "C3", "C4"):
    for diff in ("introductory", "interview", "competition"):
        n = by_type_diff.get((ctype, diff), 0)
        print(f"  {ctype} / {diff}: {n}")
