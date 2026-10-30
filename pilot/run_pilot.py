"""
Pilot driver: iterates family x category x model x architecture x repeat,
runs the protocol, evaluates, and writes JSONL events + a results table.

This runs the PIPELINE-VALIDATION pilot (2 hand-built families, see
tasks/registry.py docstring), not the blueprint's full 288-run
LiveCodeBench/APPS baseline -- that requires the curation audit described
in the blueprint's "First curation pilot" section, which is separate,
slower work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.evaluator import finalize_run_result
from harness.llm_client import LLMClient, load_model_specs, max_acceptable_output_tokens
from harness.protocol import ARCHITECTURES
from harness.schemas import RunConfig
from tasks.impossiblebench import build_ib_variants, expanded_task_ids, wrap_ib_as_taskvariant
from tasks.registry import build_pilot_families
from tasks.stratified_runtime import build_stratified_families


def build_all_families(ib_task_count: int = 3, stratified_limit_per_type: int = None) -> dict:
    families = build_pilot_families()
    ib_task_ids = expanded_task_ids(ib_task_count) if ib_task_count != 3 else None
    families["impossiblebench"] = [wrap_ib_as_taskvariant(v) for v in build_ib_variants(ib_task_ids)]
    families.update(build_stratified_families(limit_per_type=stratified_limit_per_type))
    return families

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", default=str(Path(__file__).parent / "configs" / "models.yaml"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--architectures", nargs="+", default=["single_agent", "team4_aggregator"])
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names from models.yaml")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--limit-variants", type=int, default=None, help="cap number of variants for a smoke test")
    ap.add_argument("--families", nargs="+", default=None, help="subset of family_ids from tasks/registry.py")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                     help="per-call completion token budget; defaults to each model's own max "
                          "acceptable budget (harness.llm_client.max_acceptable_output_tokens) "
                          "rather than one number for every model")
    ap.add_argument("--ib-task-count", type=int, default=3,
                     help="number of ImpossibleBench LiveCodeBench-hard problems to use "
                          "(x3 splits each); 3 is the original pilot subset, up to 103 available")
    ap.add_argument("--stratified-limit-per-type", type=int, default=None,
                     help="cap the number of LCB/APPS-derived problems per C1-C4 type from "
                          "tasks/stratified_dataset_final.jsonl (x3 splits each); default is all "
                          "(19/20/22/42); use a small number (e.g. 2-3) for the harness-validation pilot")
    ap.add_argument("--resume", action="store_true",
                     help="append to an existing results_<tag>.jsonl instead of overwriting it, "
                          "skipping any run_id already present -- for continuing after a GPU job "
                          "wall-clock expiry or other interruption without redoing finished work")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    events_path = RESULTS_DIR / f"events_{args.tag}.jsonl"
    results_path = RESULTS_DIR / f"results_{args.tag}.jsonl"

    done_run_ids = set()
    if args.resume and results_path.exists():
        with open(results_path) as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_run_ids.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[pilot] --resume: {len(done_run_ids)} run_ids already in {results_path}, will be skipped")

    specs = load_model_specs(args.models_config)
    model_names = args.models or list(specs.keys())

    families = build_all_families(args.ib_task_count, args.stratified_limit_per_type)
    if args.families:
        families = {k: v for k, v in families.items() if k in args.families}
    all_variants = [v for fam in families.values() for v in fam]
    if args.limit_variants:
        all_variants = all_variants[: args.limit_variants]

    print(f"[pilot] {len(all_variants)} variants x {len(model_names)} models x "
          f"{len(args.architectures)} architectures x {args.repeats} repeats = "
          f"{len(all_variants) * len(model_names) * len(args.architectures) * args.repeats} runs")

    n_done = 0
    t_start = time.time()
    results_mode = "a" if args.resume else "w"
    events_mode = "a" if args.resume else "w"
    with open(results_path, results_mode) as rf, open(events_path, events_mode) as ef:
        for model_name in model_names:
            client = LLMClient(specs[model_name])
            model_max_tokens = args.max_output_tokens or max_acceptable_output_tokens(model_name)
            for arch in args.architectures:
                run_fn = ARCHITECTURES[arch]
                for variant in all_variants:
                    for rep in range(args.repeats):
                        run_id = f"{variant.variant_id}__{model_name}__{arch}__r{rep}"
                        if run_id in done_run_ids:
                            continue
                        cfg = RunConfig(
                            run_id=run_id,
                            family_id=variant.family_id,
                            category=variant.category,
                            variant_id=variant.variant_id,
                            model_name=model_name,
                            architecture=arch,
                            repeat_idx=rep,
                            max_output_tokens_per_call=model_max_tokens,
                        )
                        t0 = time.time()
                        try:
                            result = run_fn(variant, client, cfg)
                            result = finalize_run_result(variant, result)
                        except Exception as e:  # noqa: BLE001
                            print(f"[pilot] RUN FAILED {run_id}: {e}")
                            n_done += 1
                            continue

                        rf.write(result.model_dump_json() + "\n")
                        rf.flush()
                        ef.write(json.dumps({
                            "run_id": run_id, "kind": "termination",
                            "payload": {"verdict": result.verdict, "decision": result.final_decision.decision if result.final_decision else None},
                        }) + "\n")
                        ef.flush()

                        n_done += 1
                        dt = time.time() - t0
                        dec = result.final_decision.decision if result.final_decision else "NONE"
                        print(f"[{n_done}] {run_id} -> verdict={result.verdict} decision={dec} "
                              f"calls={result.num_calls} tok={result.tokens_used_prompt + result.tokens_used_completion} "
                              f"({dt:.1f}s)")

    print(f"[pilot] done: {n_done} runs in {time.time() - t_start:.1f}s -> {results_path}")


if __name__ == "__main__":
    main()
