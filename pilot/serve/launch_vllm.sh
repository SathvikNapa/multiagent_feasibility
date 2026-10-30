#!/bin/bash
# Runs ON the allocated GPU node (invoked via `srun --overlap`), launches ONE
# pilot model as an OpenAI-compatible vLLM server, and blocks on `wait` so
# this remains the step's foreground process.
#
# IMPORTANT #1: do not background-and-exit here. A transient `srun --overlap`
# step that returns immediately gets its cgroup torn down by slurmstepd,
# which kills any nohup'd/disowned children too (nohup survives SIGHUP, not
# cgroup-based kill). Keeping this script blocked on `wait` keeps the step
# (and therefore the vLLM child process) alive for as long as the
# surrounding `srun` process is itself left running.
#
# IMPORTANT #2: this cluster's A100s are in `Exclusive_Process` compute mode
# (`nvidia-smi --query-gpu=compute_mode` -> Exclusive_Process), so only ONE
# CUDA context may exist on a given GPU at a time. Two vLLM servers cannot
# share one GPU here, even at low --gpu-memory-utilization -- the second one
# fails with "CUDA error: CUDA-capable device(s) is/are busy or unavailable"
# as soon as it tries to open the device, regardless of launch timing. This
# script therefore serves exactly one model; run the pilot's models
# sequentially (serve A, run A's share, kill A, serve B, run B's share).
set -uo pipefail

MODEL_KEY="${1:?usage: launch_vllm.sh <model_key: qwen3p5-4b|phi4-mini>}"
ROOT=/home/snapaug1/multiagent_feasibility/pilot
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

NODE=$(hostname)
echo "running on node $NODE, serving $MODEL_KEY"

source /home/snapaug1/anaconda3/etc/profile.d/conda.sh
conda activate mas_feas

API_KEY="pilot-local-key"
export VLLM_API_KEY="$API_KEY"
# FlashInfer's JIT-compiled sampler needs nvcc/CUDA_HOME, which this node's
# env doesn't have (pip-only CUDA runtime, no toolkit) -- fall back to the
# non-FlashInfer sampler path instead of failing on first sample.
export VLLM_USE_FLASHINFER_SAMPLER=0

case "$MODEL_KEY" in
  qwen3p5-4b)
    HF_MODEL="Qwen/Qwen3.5-4B"
    PORT=8801
    ;;
  phi4-mini)
    HF_MODEL="microsoft/Phi-4-mini-instruct"
    PORT=8802
    ;;
  qwen25coder-7b)
    HF_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
    PORT=8803
    ;;
  qwen3p5-9b)
    HF_MODEL="Qwen/Qwen3.5-9B"
    PORT=8804
    ;;
  mistral-7b)
    HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
    PORT=8805
    ;;
  granite-8b)
    HF_MODEL="ibm-granite/granite-3.1-8b-instruct"
    PORT=8806
    ;;
  qwen38-27b)
    HF_MODEL="Qwen/Qwen3.8-27B"
    PORT=8807
    ;;
  gemma4-31b)
    HF_MODEL="google/gemma-4-31B-it"
    PORT=8808
    ;;
  nemotron35-30b)
    HF_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    PORT=8809
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 1
    ;;
esac

# 27-31B dense/MoE models need ~54-62GB just for bf16 weights on one 80GB
# A100 -- leave less headroom unused (higher utilization) and shrink the KV
# cache's context budget (our prompts are all well under 2k tokens anyway).
case "$MODEL_KEY" in
  nemotron35-30b|qwen38-27b|gemma4-31b)
    # 16k context: nemotron needs the room for its <think> preamble: the
    # C1-C4 pilot prompts are short enough that 4096 worked, but
    # ImpossibleBench's LiveCodeBench-hard problem statements alone can
    # exceed 4096 input tokens before any output -- seen failing outright
    # (gemma4-31b: "maximum context length is 4096 tokens ... prompt
    # contains at least 4097 input tokens") on 3/9 IB runs at MAX_LEN=4096.
    GPU_UTIL=0.93
    MAX_LEN=16384
    ;;
  *)
    GPU_UTIL=0.85
    MAX_LEN=8192
    ;;
esac

echo "launching $MODEL_KEY ($HF_MODEL) on port $PORT (gpu_util=$GPU_UTIL, max_len=$MAX_LEN) ..."
vllm serve "$HF_MODEL" \
  --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --dtype bfloat16 \
  --served-model-name "$MODEL_KEY" \
  --api-key "$API_KEY" \
  > "$LOGDIR/vllm_${MODEL_KEY}.log" 2>&1 &
PID=$!

# Large models pull 50-60GB of bf16 weights over NFS before the engine can
# even start loading them into GPU memory, which alone can take several
# minutes beyond vLLM's own init/compile time -- give those more polls.
case "$MODEL_KEY" in
  qwen38-27b|gemma4-31b|nemotron35-30b)
    MAX_POLLS=180   # 180 * 10s = 30 min
    ;;
  *)
    MAX_POLLS=90    # 90 * 10s = 15 min
    ;;
esac

echo "$MODEL_KEY pid=$PID, waiting for health (max ${MAX_POLLS} polls)..."
HEALTHY=0
for i in $(seq 1 "$MAX_POLLS"); do
  ok=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" || true)
  echo "poll $i: $MODEL_KEY=$ok"
  if [ "$ok" = "200" ]; then
    echo "$MODEL_KEY healthy"
    HEALTHY=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "$MODEL_KEY process died, see $LOGDIR/vllm_${MODEL_KEY}.log"
    exit 1
  fi
  sleep 10
done

# Only publish models.yaml once the server has actually answered healthy --
# writing it unconditionally here would let run_pilot.py start hammering an
# endpoint that never came up (seen with gemma4-31b: wrote the file after
# exhausting all polls while weights were still only 10% prefetched).
if [ "$HEALTHY" -ne 1 ]; then
  echo "$MODEL_KEY never became healthy after $MAX_POLLS polls, see $LOGDIR/vllm_${MODEL_KEY}.log" >&2
  exit 1
fi

# Merge this model's entry into models.yaml instead of overwriting the
# whole file -- multiple servers can now run concurrently (one per GPU)
# and each needs to add itself without clobbering the others' entries.
# flock serializes the read-modify-write against a concurrent writer.
(
  flock -x 200
  python3 - "$ROOT/configs/models.yaml" "$MODEL_KEY" "$NODE" "$PORT" "$API_KEY" << 'PYEOF'
import sys, yaml
path, model_key, node, port, api_key = sys.argv[1:6]
try:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
except FileNotFoundError:
    doc = {}
doc.setdefault("models", {})[model_key] = {
    "base_url": f"http://{node}:{port}/v1",
    "api_key": api_key,
    "served_model_name": model_key,
    "thinking_mode": False,
}
with open(path, "w") as f:
    yaml.safe_dump(doc, f)
PYEOF
) 200>"$ROOT/configs/.models.yaml.lock"
echo "wrote $ROOT/configs/models.yaml (node=$NODE, model=$MODEL_KEY)"

# Keep this the step's foreground process so the vLLM child stays alive.
wait "$PID"
