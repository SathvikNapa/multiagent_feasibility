"""Stage 1 of the stratified-dataset pipeline: for each of the 200 new
LiveCodeBench/APPS problems, ask an OpenAI model which ONE of C1-C4 (or
NONE) is the most natural fit for an F1/F2/F3 construction in the same
style as this pilot's 8 hand-built families, with a short sketch.

This is deliberately cheap/fast and run on the FULL 200 before any
detailed construction, so the type distribution reflects what these
problems actually support rather than a forced 50/50/50/50 split --
not every competitive-programming problem admits every constraint type
(e.g. C2's "restricted operation budget" framing only fits problems whose
solution is naturally a sequence of discrete state-changing operations).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
with open(ROOT / ".." / ".env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENAI_API_KEY"):
            _, _, v = line.partition("=")
            os.environ["OPENAI_API_KEY"] = v.strip().strip('"').strip("'")

from openai import OpenAI  # noqa: E402

client = OpenAI()

SYSTEM_PROMPT = """You are helping extend a research pilot on multi-agent LLM feasibility testing. The pilot already has 8 hand-built task families, one per constraint type, each with F1 (feasible)/F2 (feasible-but-constrained)/F3 (certified infeasible) variants. Study these exact patterns:

C1 (output/semantic requirements) -- a rule about the output's correctness itself.
  Example: ordered_sort_pairs. Sort a list, but ALSO require that for certain (i,j) index pairs, the element originally at i ends up before the element originally at j. F1: no such pairs. F2: pairs that are already consistent with ascending order. F3: inject one pair that directly contradicts ascending order on a specific input (value at i > value at j) -- provably impossible since ascending order forces the opposite relative position.

C2 (operation/computation budget) -- solve using ONLY a restricted interface (peek + exactly one state-changing operation), under a cap on how many times that operation may be called.
  Example: adj_swap_sort. Sort via workspace.peek(i) and workspace.swap_adjacent(i) only. F1: no budget. F2: budget = exact inversion count (provably tight). F3: budget = inversion_count - 1 (provably unreachable, since each swap removes at most 1 inversion).

C3 (workspace/state budget) -- solve using ONLY a fixed-size RegisterBank (N registers, each holding an integer in a bounded range) as memory, reading a single-pass stream.
  Example: bitvector_recall. Reconstruct N bits read one at a time via stream.next_bit(), using only registers.set(i,v)/registers.get(i). F1/F2: registers have >= N total bits of capacity (feasible, various packings). F3: registers have fewer bits of capacity than N (provably impossible by pigeonhole: more required output states than representable register states).

C4 (information/access) -- solve with only a restricted VIEW of the input: peek(i) raises an error outside a visible prefix, the rest is NEVER reachable (not just unobserved).
  Example: prefix_restricted_max / hidden_suffix_search. F1: answer determinable from the visible prefix alone. F2: a stated guarantee about the hidden part makes it determinable. F3: no guarantee, and two inputs sharing the identical visible prefix would require different true answers -- provably impossible for any function of only the visible part.

Given a NEW problem (from LiveCodeBench or APPS), decide which ONE of C1/C2/C3/C4 is the most natural fit for building a similar F1/F2/F3 triple -- or NONE if the problem's structure doesn't cleanly support any of the four without forcing something artificial. Most problems will fit at most one type well; many problems (e.g. pure number-theory counting, graph problems with no natural operation-budget or memory-budget framing) will fit NONE, and that is a correct, expected answer -- don't force a fit.

Reply with ONLY a JSON object (no markdown fences, no prose outside the JSON):
{"constraint_type": "C1"|"C2"|"C3"|"C4"|"NONE", "confidence": <float 0..1>, "sketch": "<2-3 sentences: if not NONE, specifically how you'd construct the F2 restriction and the F3 restriction+certificate for THIS problem under that type; if NONE, why it doesn't fit any type>"}"""


def classify_one(problem_text: str, title: str, max_tokens: int = 6000) -> dict:
    user_prompt = f"Problem title: {title}\n\nProblem statement:\n{problem_text[:4000]}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # gpt-5 spends part of max_completion_tokens on hidden reasoning tokens
    # before any visible content; on harder problems that can consume the
    # whole budget and leave an empty response. Retry once with a much
    # larger budget before giving up.
    for attempt_tokens in (max_tokens, max_tokens * 3):
        resp = client.chat.completions.create(
            model="gpt-5", messages=messages, max_completion_tokens=attempt_tokens,
        )
        text = resp.choices[0].message.content or ""
        if text.strip():
            break
    try:
        first, last = text.index("{"), text.rindex("}")
        return json.loads(text[first:last + 1])
    except (ValueError, json.JSONDecodeError) as e:
        return {"constraint_type": "PARSE_ERROR", "confidence": 0.0, "sketch": f"raw={text[:300]!r} err={e}"}


def main():
    lcb = json.load(open(ROOT / "tasks" / "stratified_lcb_full.json"))
    apps = json.load(open(ROOT / "tasks" / "stratified_apps_full.json"))

    items = []
    for r in lcb:
        items.append({"source": "livecodebench", "id": r["question_id"], "title": r["title"], "text": r["content"]})
    for r in apps:
        items.append({"source": "apps", "id": r["id"], "title": f"apps_{r['id']}", "text": r["question"]})

    out_path = ROOT / "tasks" / "stratified_classification.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[classify] {len(done_ids)} already done, resuming")

    with open(out_path, "a") as out:
        for i, item in enumerate(items):
            if item["id"] in done_ids:
                continue
            t0 = time.time()
            try:
                result = classify_one(item["text"], item["title"])
            except Exception as e:  # noqa: BLE001
                result = {"constraint_type": "API_ERROR", "confidence": 0.0, "sketch": str(e)}
            record = {"source": item["source"], "id": item["id"], "title": item["title"], **result}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i+1}/{len(items)}] {item['source']}:{item['id']} -> {result.get('constraint_type')} "
                  f"(conf={result.get('confidence')}) ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
