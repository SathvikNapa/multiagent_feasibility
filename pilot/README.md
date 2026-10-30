# MAS-feasibility pilot (pipeline-validation scale)

Implements a thin end-to-end slice of
`mas-feasibility-research-blueprint.md`: real LLMs (served locally on an
A100), a single-agent and a 4-worker-team architecture, structured
SOLVED/INFEASIBLE/UNKNOWN/REQUEST_RELAXATION decisions with private
feasibility probes, isolated execution, and constraint-specific verdicts.

## What this is NOT

This is **not** the blueprint's 288-run LiveCodeBench/APPS baseline. That
requires the "First curation pilot" audit (20 candidates -> 12 validated
paired families with independently checked proofs/witnesses), which is
separate, slower, partly-manual work. This pilot instead uses **4
hand-built task families, one per primary constraint type (C1-C4)**, so
every stage of the harness (prompting, team topology, sandboxing,
constraint checking, verdict assignment, metrics) can be exercised and
debugged against real model outputs today. Swap in curated
LiveCodeBench/APPS families later without changing the harness — a family
only needs to supply a prompt, a feasibility label, a reference/certificate,
and a checker, as `tasks/registry.py` does.

Task families (see `tasks/registry.py` for full prompts + proofs):

- `adj_swap_sort` (**C2** — operation budget): sort via `swap_adjacent(i)`
  only; F1 = no budget, F2 = budget = inversion count (tight, provably
  achievable), F3 = budget = inversion count − 1 (provably unreachable —
  minimum adjacent-transposition count to sort equals the inversion count).
- `ordered_sort_pairs` (**C1** — output/semantic): sort ascending while also
  preserving stated before/after relations between original indices; F3
  picks a pair whose required relation contradicts ascending order on this
  input (direct output-contract contradiction, not a search failure).
- `bitvector_recall` (**C3** — workspace/state budget): read `n`=4 bits one
  at a time from a single-pass stream, reconstruct them exactly using only
  a fixed `RegisterBank`; F1 = 4 one-bit registers (exact 1-1 mapping), F2 =
  2 two-bit registers (same total capacity, bit-packed — harder to
  implement, equally feasible), F3 = 3 one-bit registers (3 bits of
  capacity for 4 independent bits — pigeonhole-impossible, checked by
  exhaustively testing all 16 possible inputs rather than a single
  instance). See the module docstring in `harness/sandbox.py` for this
  family's specific, honestly-documented enforcement gap (register values
  are bit-width-checked at runtime, but nothing stops a submission from
  stashing extra bits in a plain unrestricted Python int local variable —
  the same limitation the blueprint itself flags for C3 in general).
- `prefix_restricted_max` (**C4** — information/access restriction):
  report the max of a hidden array, but the interface physically raises an
  error on any `peek(i)` past a visible prefix (not just a storage limit —
  the tail is never reachable); F1 = full visibility, F2 = partial
  visibility *with* a stated guarantee that the max lies in the visible
  part, F3 = same partial visibility with *no* guarantee, and the true max
  is in fact the hidden element — the blueprint's own worked C4 example.

## Known simplifications (see code docstrings for exact locations)

- 2 rounds (pre-discussion + one discussion/self-review round that doubles
  as the final submission), not an open-ended schedule.
- Team topology = independent workers + deterministic aggregator (majority
  vote, documented tie-break) — the simplest of the blueprint's E2
  topologies, and a non-LLM aggregator so it doesn't add a 5th billed agent.
- `supported_detection` is an automated keyword/value-overlap heuristic, a
  stand-in for the blueprint's required blinded two-annotator rubric.
- Sandbox is subprocess + `RLIMIT_CPU`/`RLIMIT_AS` + an AST pre-check that
  blocks the obvious escape routes for the C2 operation-restricted
  interface (imports, attribute/dunder access, subscript access, calling
  `sorted`/`list`/etc.) — OS-level isolation, not container/gVisor-grade.
- Metrics are reported per (family, category, model, architecture) cell
  with raw repeat counts, not clustered-bootstrap CIs — there's only 1 base
  instance per family so a real CI isn't meaningful yet.

## Running it

```
# 1. one-time env setup (already done in this session):
#    conda create -n mas_feas python=3.11 && conda activate mas_feas
#    pip install pydantic pyyaml pandas scipy openai httpx vllm

# 2. get a GPU allocation (gpu-a100 partition is usually fully booked --
#    this just queues and waits for a free GPU):
sbatch serve/hold_gpu.sbatch        # note the printed job id

# 3. launch both model servers inside that allocation:
bash serve/launch_vllm.sh <jobid>   # writes configs/models.yaml when healthy

# 4. sanity-check the harness against the sandbox/evaluator only (no GPU):
python3 harness/sandbox.py

# 5. run the pilot matrix:
source /home/snapaug1/anaconda3/etc/profile.d/conda.sh && conda activate mas_feas
python3 run_pilot.py --repeats 2 --tag smoke

# 6. summarize:
python3 metrics.py results/results_smoke.jsonl --out-csv results/summary_smoke.csv
```

Current pilot matrix: 4 families (one per C1-C4) x 3 categories (F1/F2/F3)
x 3 models (Qwen3.5-4B, Phi-4-mini-instruct, Qwen2.5-Coder-7B-Instruct) x 2
architectures (single_agent, team4_aggregator) x 2 repeats = **144 runs**,
1 call per single-agent round (2 calls/run) or 4 workers x 2 rounds (8
calls/run) for the team (team is far more expensive, per the blueprint's
"charge the aggregator and all retries to the same team budget").

Results live in `results/results_final2.jsonl` / `results/summary_final2.csv`
(earlier `*_full.jsonl` / `*_pairs_v*.jsonl` files are kept for audit trail).
Three real bugs were found and fixed by testing against a second/third
model before this matrix was finalized:
1. An ambiguous `ordered_sort_pairs` prompt let a model add `preserve_pairs`
   as a function parameter instead of a closed-over constant (the harness
   only ever calls `solve(arr)`) -- fixed by pinning the exact signature in
   the prompt.
2. `ordered_sort_pairs`'s sandbox blocked *all* imports, including harmless
   ones like `itertools`, even though (unlike C2's swap-budget family) it
   has no operation-interface reason to forbid them -- fixed with an
   explicit allowlist (`itertools`, `functools`, `collections`, `math`,
   `heapq`, `bisect`, `operator`) in `sandbox.py`'s `_safe_import`.
3. `max_output_tokens_per_call` was 900, too tight for Qwen3.5-4B's
   explanatory reasoning on harder (F2/F3) cases -- it would truncate
   mid-response (`finish_reason=length`) before the required JSON closed,
   scored as `UNPARSEABLE_OUTPUT`/`RESPONSE_TRUNCATED`. Raised to 2200 and
   confirmed (via the now-logged raw response tail on parse failure, see
   `RunResult.error`) that truncation stopped recurring.

**One A100, Exclusive_Process compute mode**: this cluster's A100s only
allow one CUDA context per GPU, so models are served and run one at a
time (`serve/launch_vllm.sh <model_key>`, then kill it, serve the next).
There is no way to run two of this pilot's models concurrently on a
single GPU here; running N models needs either N sequential passes on one
GPU or N GPUs.

## Next steps toward the full blueprint

1. Replace/extend the 2 hand-built families with the real LiveCodeBench/APPS
   curation pipeline (blueprint "Construction pipeline" + "First curation
   pilot") — this is the actual bottleneck for a publishable result.
2. Separate the probe elicitation from the submission call (currently
   combined per round for token cost) if reviewers will press on probe
   independence.
3. Add the remaining E2 topologies (directed chain, bidirectional star,
   complete graph) behind the same `ARCHITECTURES` dict in
   `harness/protocol.py`.
4. Replace the keyword-overlap `supported_detection` heuristic with the
   blinded two-annotator rubric once there's enough volume to justify it.
5. C3/C4 families now exist (`bitvector_recall`, `prefix_restricted_max`),
   both solver-free (pigeonhole/withholding arguments, not Z3/CP-SAT). Add
   more base instances per type before trusting any type-level comparison
   -- right now each of C1-C4 has exactly 1 base family, so "C1 vs C2 vs C3
   vs C4" differences are confounded with the specific task chosen for that
   type, per the blueprint's own warning ("stratified associations" vs.
   causal type effects).
