"""Dry-run the full harness (protocol + evaluator + metrics) with a mock
LLM client, no GPU required. Catches wiring bugs before spending real
model-serving budget."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.evaluator import finalize_run_result
from harness.mock_client import MockLLMClient
from harness.protocol import ARCHITECTURES
from harness.schemas import RunConfig
from tasks.impossiblebench import build_ib_variants, wrap_ib_as_taskvariant
from tasks.registry import build_pilot_families

families = build_pilot_families()
all_variants = [v for fam in families.values() for v in fam]
all_variants += [wrap_ib_as_taskvariant(v) for v in build_ib_variants()]

out_path = Path(__file__).parent / "results" / "results_dryrun.jsonl"
out_path.parent.mkdir(exist_ok=True)

n = 0
with open(out_path, "w") as f:
    for arch_name, run_fn in ARCHITECTURES.items():
        client = MockLLMClient(seed=hash(arch_name) % 1000)
        for variant in all_variants:
            for rep in range(2):
                cfg = RunConfig(
                    run_id=f"{variant.variant_id}__mock__{arch_name}__r{rep}",
                    family_id=variant.family_id, category=variant.category,
                    variant_id=variant.variant_id, model_name="mock",
                    architecture=arch_name, repeat_idx=rep,
                )
                result = run_fn(variant, client, cfg)
                result = finalize_run_result(variant, result)
                f.write(result.model_dump_json() + "\n")
                n += 1
                dec = result.final_decision.decision if result.final_decision else "NONE"
                print(f"{cfg.run_id} -> verdict={result.verdict} decision={dec} calls={result.num_calls} err={result.error}")

print(f"\n{n} dry-run results written to {out_path}")
