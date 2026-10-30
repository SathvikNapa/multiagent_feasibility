"""Thin client over vLLM's OpenAI-compatible server.

Model endpoints are read from configs/models.yaml, which is written by
serve/launch_vllm.sh once the servers are up (it records the resolved
compute-node hostname and port, since SLURM assigns the node dynamically).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import yaml
from openai import OpenAI


@dataclass
class ModelSpec:
    name: str
    base_url: str
    api_key: str
    served_model_name: str
    thinking_mode: bool = False


# Mirrors serve/launch_vllm.sh's --max-model-len buckets. Used to pick each
# model's own max *acceptable* completion-token budget rather than one
# number for every model: a flat budget either wastes a short-context
# model's headroom or (as seen with nemotron on ImpossibleBench, and with
# gemma4-31b/qwen38-27b before their context was widened) truncates a
# long-context reasoning model's answer mid-thought.
MAX_MODEL_LEN = {
    "nemotron35-30b": 16384,
    "qwen38-27b": 16384,
    "gemma4-31b": 16384,
}
DEFAULT_MAX_MODEL_LEN = 8192
PROMPT_MARGIN = 4096  # reserved for the prompt (ImpossibleBench's longest run a few thousand tokens)


def max_acceptable_output_tokens(model_name: str) -> int:
    """The largest completion budget that still leaves PROMPT_MARGIN tokens
    of headroom in this model's configured context window."""
    return MAX_MODEL_LEN.get(model_name, DEFAULT_MAX_MODEL_LEN) - PROMPT_MARGIN


def load_model_specs(path: str) -> dict[str, ModelSpec]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    specs = {}
    for name, d in cfg["models"].items():
        specs[name] = ModelSpec(name=name, **d)
    return specs


class LLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.client = OpenAI(base_url=spec.base_url, api_key=spec.api_key)

    def chat(self, messages: list[dict], temperature: float, max_tokens: int,
              retries: int = 3) -> tuple[str, dict]:
        last_err = None
        for attempt in range(retries):
            try:
                extra_body = {}
                if "qwen" in self.spec.served_model_name.lower() and not self.spec.thinking_mode:
                    # Qwen3.5 defaults to thinking mode; the blueprint requires
                    # fixing this explicitly rather than leaving it ambient.
                    # Phi-4-mini's chat template has no such switch, so we
                    # don't send it a kwarg its template doesn't expect.
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                resp = self.client.chat.completions.create(
                    model=self.spec.served_model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body or None,
                )
                text = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                    "finish_reason": resp.choices[0].finish_reason,
                }
                return text, usage
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")
