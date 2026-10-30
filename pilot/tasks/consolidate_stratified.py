"""Consolidate the 4 per-type validated construction files into one
stratified dataset manifest, alongside the original 103 ImpossibleBench
pool, for the main paper's full-scale analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TASKS = ROOT / "tasks"

apps_full = {r["id"]: r for r in json.load(open(TASKS / "stratified_apps_full.json"))}

out_rows = []
counts = {}
for ctype in ("C1", "C2", "C3", "C4"):
    paths = [TASKS / f"stratified_final_{ctype}.jsonl"]
    extra = TASKS / f"stratified_final_{ctype}_apps_new.jsonl"  # APPS-only extension (LCB/IB frozen)
    if extra.exists():
        paths.append(extra)

    rows = []
    seen = set()
    for path in paths:
        for l in open(path):
            r = json.loads(l)
            key = (r["source"], r["id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    counts[ctype] = {"total": len(rows), "by_source": {}}
    for r in rows:
        counts[ctype]["by_source"][r["source"]] = counts[ctype]["by_source"].get(r["source"], 0) + 1
        out_rows.append({
            "dataset": "stratified_new",
            "constraint_type": ctype,
            "source": r["source"],
            "source_id": r["id"],
            "title": r.get("title"),
            "difficulty": r.get("difficulty") or (apps_full.get(r["id"], {}).get("difficulty") if r["source"] == "apps" else None),
            "oracle_code": r.get("oracle_code"),
            "construction": {k: v for k, v in r.items()
                              if k not in ("source", "id", "title", "oracle_code", "constraint_type", "difficulty", "_verification")},
            "verification": r.get("_verification"),
        })

out_path = TASKS / "stratified_dataset_final.jsonl"
with open(out_path, "w") as f:
    for row in out_rows:
        f.write(json.dumps(row) + "\n")

print(f"wrote {out_path}: {len(out_rows)} total validated constructions")
for ctype, c in counts.items():
    print(f"  {ctype}: {c['total']} ({c['by_source']})")

# Cross-check against original 103-problem ImpossibleBench pool for a combined manifest summary.
ib_pool_path = TASKS / "stratified_manifest.json"
n_ib = None
if ib_pool_path.exists():
    manifest = json.load(open(ib_pool_path))
    n_ib = manifest["impossiblebench"]["n"]
    n_lcb_sample = manifest["livecodebench"].get("n", len(manifest["livecodebench"].get("task_ids", [])))
    n_apps_sample = manifest["apps"].get("n", len(manifest["apps"].get("task_ids", [])))
    print(f"\nExisting pools (stratified_manifest.json): ImpossibleBench={n_ib}, "
          f"LiveCodeBench sample={n_lcb_sample}, APPS sample={n_apps_sample}")

print(f"\nFull paper dataset = {n_ib if n_ib is not None else '?'} ImpossibleBench (hand-built) "
      f"+ {len(out_rows)} newly constructed (LCB/APPS-derived, model-drafted + independently verified)")
