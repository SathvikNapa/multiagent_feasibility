"""
Aggregate a results_*.jsonl into the blueprint's core metrics table
(subset relevant to a 2-family pilot; see Measurements section).
Clustering note: with 1 base instance per family, clustered CIs over base
instances aren't meaningful yet -- this reports per (family, category,
model, architecture) cells with repeat-level counts, which is what a
pipeline-validation pilot can actually support. Scale to many base
families before trusting any confidence interval.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            fd = d.get("final_decision") or {}
            rows.append({
                "run_id": d["run_id"],
                "family_id": d["config"]["family_id"],
                "category": d["config"]["category"],
                "model_name": d["config"]["model_name"],
                "architecture": d["config"]["architecture"],
                "repeat_idx": d["config"]["repeat_idx"],
                "true_feasible": d["true_feasible"],
                "decision": fd.get("decision"),
                "p_feasible": fd.get("p_feasible"),
                "verdict": d["verdict"],
                "functional_pass": d["functional_pass"],
                "constraint_pass": d["constraint_pass"],
                "supported_detection": d.get("supported_detection"),
                "tokens_total": d["tokens_used_prompt"] + d["tokens_used_completion"],
                "num_calls": d["num_calls"],
                "wall_time_s": d["wall_time_s"],
                "error": d.get("error"),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["family_id", "category", "model_name", "architecture"]
    for key, g in df.groupby(group_cols):
        n = len(g)
        f_true = bool(g["true_feasible"].iloc[0])
        valid_success = ((g["decision"] == "SOLVED") & (g["verdict"] == "PASS")).mean()
        false_success = ((g["decision"] == "SOLVED") & (g["verdict"] != "PASS")).mean()
        constraint_sacrifice = (g["functional_pass"] & ~g["constraint_pass"]).mean()
        infeasibility_rate = (g["decision"] == "INFEASIBLE").mean()
        unknown_rate = (g["decision"] == "UNKNOWN").mean()
        relax_rate = (g["decision"] == "REQUEST_RELAXATION").mean()
        supported_rate = g["supported_detection"].dropna().mean() if g["supported_detection"].notna().any() else float("nan")
        mean_p_feasible = g["p_feasible"].mean()
        brier = ((g["p_feasible"] - (1.0 if f_true else 0.0)) ** 2).mean()
        mean_tokens = g["tokens_total"].mean()
        mean_calls = g["num_calls"].mean()

        row = dict(zip(group_cols, key))
        row.update({
            "n_runs": n,
            "true_feasible": f_true,
            "valid_success": valid_success,
            "false_success": false_success,
            "constraint_sacrifice": constraint_sacrifice,
            "infeasibility_rate": infeasibility_rate,
            "unknown_rate": unknown_rate,
            "relaxation_rate": relax_rate,
            "supported_detection_rate(heuristic)": supported_rate,
            "mean_p_feasible": round(mean_p_feasible, 3) if pd.notna(mean_p_feasible) else None,
            "feasibility_brier": round(brier, 3) if pd.notna(brier) else None,
            "mean_tokens_per_run": round(mean_tokens, 1) if pd.notna(mean_tokens) else None,
            "mean_calls_per_run": round(mean_calls, 1) if pd.notna(mean_calls) else None,
        })
        out.append(row)
    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_path")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    df = load_results(args.results_path)
    print(f"loaded {len(df)} runs")
    summary = summarize(df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
