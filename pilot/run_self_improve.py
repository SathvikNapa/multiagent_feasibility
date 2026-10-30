"""Driver for the self-improvement protocol (harness/self_improve.py).
Runs all 4 conditions (frozen/reflection/scaffold/recursive) x N lineages
on a fixed task, for one model at a time (matching this cluster's
one-GPU-at-a-time serving constraint)."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.llm_client import LLMClient, load_model_specs, max_acceptable_output_tokens
from harness.self_improve import run_lineage
from tasks.registry import build_pilot_families

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", default=str(Path(__file__).parent / "configs" / "models.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="adj_swap_sort")
    ap.add_argument("--category", default="F2")
    ap.add_argument("--n-generations", type=int, default=3)
    ap.add_argument("--n-lineages", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-output-tokens", type=int, default=None,
                     help="per-call completion token budget; defaults to the model's own max "
                          "acceptable budget (harness.llm_client.max_acceptable_output_tokens)")
    ap.add_argument("--architecture", default="single_agent",
                     help="who produces each generation's attempt: single_agent (default) or any "
                          "of ARCHITECTURES (team4_aggregator, comm_independent/chain/star/fully_connected)")
    args = ap.parse_args()

    specs = load_model_specs(args.models_config)
    client = LLMClient(specs[args.model])
    max_output_tokens = args.max_output_tokens or max_acceptable_output_tokens(args.model)

    families = build_pilot_families()
    variant = next(v for v in families[args.family] if v.category == args.category)

    out_path = RESULTS_DIR / f"selfimprove_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for condition in ("frozen", "reflection", "scaffold", "recursive"):
            for lineage_idx in range(args.n_lineages):
                lineage = run_lineage(variant, client, args.model, condition,
                                       n_generations=args.n_generations, lineage_idx=lineage_idx,
                                       max_output_tokens=max_output_tokens,
                                       architecture=args.architecture)
                record = {
                    "condition": lineage.condition,
                    "variant_id": lineage.variant_id,
                    "model_name": lineage.model_name,
                    "lineage_idx": lineage_idx,
                    "final_meta_prompt": lineage.final_meta_prompt,
                    "generations": [
                        {
                            "generation": g.generation,
                            "decision": g.run_result.final_decision.decision if g.run_result.final_decision else None,
                            "verdict": g.run_result.verdict,
                            "reason": g.run_result.final_decision.reason if g.run_result.final_decision else None,
                            "proposed_rule": g.proposed_rule,
                            "system_prompt_len": len(g.system_prompt_used),
                            "tokens": g.run_result.tokens_used_prompt + g.run_result.tokens_used_completion,
                        }
                        for g in lineage.generations
                    ],
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                verdicts = [g["verdict"] for g in record["generations"]]
                print(f"{args.model} {condition} lineage{lineage_idx}: {verdicts}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
