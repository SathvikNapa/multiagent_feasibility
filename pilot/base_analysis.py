"""
Base analysis: F1/F2/F3 outcomes crossed with constraint type (C1-C4), per
the user's request. Each family maps to exactly one primary constraint
type (adj_swap_sort=C2, ordered_sort_pairs=C1, bitvector_recall=C3,
prefix_restricted_max=C4), so "by constraint type" here is equivalent to
"by family" -- there is exactly 1 base family per type, so a type-level
number IS a single-family number (confounded, per the blueprint's own
warning: "stratified associations... not a causal effect of type" until
multiple families per type exist).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_results, summarize
from tasks.registry import build_pilot_families

FAMILY_TO_TYPE = {fam_id: variants[0].primary_constraint_type
                  for fam_id, variants in build_pilot_families().items()}


def main():
    df = load_results(sys.argv[1] if len(sys.argv) > 1 else "results/results_final2.jsonl")
    df["constraint_type"] = df["family_id"].map(FAMILY_TO_TYPE)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    print("=" * 100)
    print("BASE ANALYSIS: outcome by constraint type x category, collapsed across models/architectures/repeats")
    print("=" * 100)
    type_cat = []
    for (ctype, cat), g in df.groupby(["constraint_type", "category"]):
        n = len(g)
        f_true = bool(g["true_feasible"].iloc[0])
        type_cat.append({
            "constraint_type": ctype,
            "category": cat,
            "family": g["family_id"].iloc[0],
            "n_runs": n,
            "true_feasible": f_true,
            "valid_success": round(((g.decision == "SOLVED") & (g.verdict == "PASS")).mean(), 2),
            "false_success": round(((g.decision == "SOLVED") & (g.verdict != "PASS")).mean(), 2),
            "infeasibility_claim_rate": round((g.decision == "INFEASIBLE").mean(), 2),
            "unknown_rate": round((g.decision == "UNKNOWN").mean(), 2),
            "constraint_sacrifice": round((g.functional_pass & ~g.constraint_pass).mean(), 2),
            "mean_p_feasible": round(g["p_feasible"].mean(), 2) if g["p_feasible"].notna().any() else None,
            "feasibility_brier": round(((g["p_feasible"] - (1.0 if f_true else 0.0)) ** 2).mean(), 3) if g["p_feasible"].notna().any() else None,
        })
    type_cat_df = pd.DataFrame(type_cat).sort_values(["constraint_type", "category"])
    print(type_cat_df.to_string(index=False))

    print()
    print("=" * 100)
    print("BASE ANALYSIS: by constraint type x category x model (collapsed across architecture/repeats)")
    print("=" * 100)
    rows = []
    for (ctype, cat, model), g in df.groupby(["constraint_type", "category", "model_name"]):
        f_true = bool(g["true_feasible"].iloc[0])
        rows.append({
            "constraint_type": ctype, "category": cat, "model": model, "n": len(g),
            "valid_success": round(((g.decision == "SOLVED") & (g.verdict == "PASS")).mean(), 2),
            "false_success": round(((g.decision == "SOLVED") & (g.verdict != "PASS")).mean(), 2),
            "infeasible_claim": round((g.decision == "INFEASIBLE").mean(), 2),
            "unknown": round((g.decision == "UNKNOWN").mean(), 2),
        })
    per_model_df = pd.DataFrame(rows).sort_values(["constraint_type", "category", "model"])
    print(per_model_df.to_string(index=False))

    print()
    print("=" * 100)
    print("BASE ANALYSIS: architecture effect within each constraint type (single_agent vs team4_aggregator)")
    print("=" * 100)
    rows = []
    for (ctype, arch), g in df.groupby(["constraint_type", "architecture"]):
        rows.append({
            "constraint_type": ctype, "architecture": arch, "n": len(g),
            "valid_success": round(((g.decision == "SOLVED") & (g.verdict == "PASS")).mean(), 2),
            "false_success": round(((g.decision == "SOLVED") & (g.verdict != "PASS")).mean(), 2),
            "infeasible_claim_on_F3": round(
                ((g.category == "F3") & (g.decision == "INFEASIBLE")).sum()
                / max((g.category == "F3").sum(), 1), 2),
        })
    arch_df = pd.DataFrame(rows).sort_values(["constraint_type", "architecture"])
    print(arch_df.to_string(index=False))

    type_cat_df.to_csv("results/base_analysis_by_type.csv", index=False)
    per_model_df.to_csv("results/base_analysis_by_type_model.csv", index=False)
    arch_df.to_csv("results/base_analysis_by_type_architecture.csv", index=False)
    print("\nwrote results/base_analysis_by_type*.csv")


if __name__ == "__main__":
    main()
