"""Stage 2 of the stratified-dataset pipeline: for a classified problem,
ask an OpenAI model to draft an F1/F2/F3 construction, in a schema designed
so EVERY claim can be independently re-derived by stratified_verify.py
rather than trusted from the model's stated certificate -- the model
supplies an oracle (a correct reference solution to the ORIGINAL,
unrestricted problem) plus concrete data; the verifier recomputes ground
truth itself (brute force for C3's pigeonhole count, BFS for C2's exact
minimum operation count, direct predicate checks for C1, adversarial-pair
oracle comparison for C4).
"""
from __future__ import annotations

import json
import os
import re
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

SCHEMAS = {
    "C1": """{
  "constraint_type": "C1",
  "oracle_code": "<python: def oracle(input): ... -> return the correct answer for the ORIGINAL problem, any valid input>",
  "f1": {"input": <a concrete input value for F1 -- JSON-serializable: int/str/list/dict as this problem's input type>, "extra_constraint_code": "<python: def extra_ok(input, output): ... -> bool, an ADDITIONAL requirement on the output beyond base correctness, that oracle(f1_input)'s true answer SATISFIES>"},
  "f2": {"input": <a concrete input for F2>, "extra_constraint_code": "<python def extra_ok(input, output): ... that oracle(f2_input)'s true answer ALSO satisfies, but construction-wise feels tighter/less automatic than F1's>"},
  "f3": {"input": <a concrete input for F3>, "extra_constraint_code": "<python def extra_ok(input, output): ... that oracle(f3_input)'s true answer VIOLATES -- since the true answer is unique, no output can be both correct and satisfy this>"}
}""",
    "C2": """{
  "constraint_type": "C2",
  "oracle_code": "<python: def oracle(input): ... -> the correct answer for the ORIGINAL problem, any valid input>",
  "state_input": <a concrete input, JSON-serializable, that will be turned into mutable state for F1/F2/F3 -- e.g. a list>,
  "init_state_code": "<python: def init_state(input): ... -> return the initial mutable state object (e.g. list(input)) that apply_op will mutate>",
  "apply_op_code": "<python: def apply_op(state, *args): ... -> mutates state IN PLACE, representing ONE allowed primitive operation (e.g. swap two adjacent elements, increment a counter, flip one bit) -- this is the ONLY state-changing operation the agent may use>",
  "op_arity": <int: how many int arguments apply_op takes besides state, e.g. 1 for swap_adjacent(state, i)>,
  "peek_code": "<python: def peek(state, i): ... -> return an observable value at position/index i, read-only>",
  "goal_code": "<python: def goal_reached(state, input): ... -> bool, True iff state currently represents a correct solution to oracle(input)>",
  "state_space_note": "<one sentence: confirm the state space reachable by repeated apply_op calls from init_state(state_input) is SMALL ENOUGH to brute-force search exhaustively (e.g. permutations of a length-<=8 list, or a bounded integer counter) -- this will be used to compute the EXACT minimum number of apply_op calls needed to reach any goal state, so it must terminate quickly>"
}""",
    "C3": """{
  "constraint_type": "C3",
  "oracle_code": "<python: def oracle(input): ... -> the correct answer for the ORIGINAL problem, any valid input>",
  "stream_length": <int N, the number of symbols the single-pass stream reveals -- keep N <= 12 so alphabet^N is exhaustively enumerable>,
  "alphabet_size": <int, number of distinct symbol values per stream position, e.g. 2 for bits>,
  "stream_to_input_code": "<python: def stream_to_input(symbols): ... -> takes a list of N symbols (each in range(alphabet_size)) and returns the ORIGINAL problem's input that this stream represents>",
  "num_registers_f1": <int>, "bits_per_register_f1": <int>,
  "num_registers_f2": <int>, "bits_per_register_f2": <int>,
  "num_registers_f3": <int>, "bits_per_register_f3": <int, chosen so num_registers_f3 * bits_per_register_f3 is LESS than ceil(log2(number of distinct oracle() outputs across all alphabet_size**stream_length possible streams)) -- you don't need to compute this exactly, just make it clearly small, the verifier will brute-force check it>"
}""",
    "C4": """{
  "constraint_type": "C4",
  "oracle_code": "<python: def oracle(input): ... -> the correct answer for the ORIGINAL problem, any valid input (input is a list)>",
  "visible_k": <int, how many leading elements of the input list are visible>,
  "f1_input": <a concrete input list where the answer is determinable from just the first visible_k elements>,
  "f2_input": <a concrete input list where a stated guarantee about the hidden tail (describe it in f2_guarantee_text) makes the answer determinable>,
  "f2_guarantee_text": "<one sentence describing the guarantee about the hidden part, stated as a given fact in the F2 prompt>",
  "f3_input_a": <a concrete input list, full length>,
  "f3_input_b": <a DIFFERENT concrete input list that shares the IDENTICAL first visible_k elements as f3_input_a, but oracle(f3_input_a) != oracle(f3_input_b) -- this is the crux of the F3 certificate, so double check the first visible_k elements truly match and the oracle outputs truly differ>"
}""",
}

SYSTEM_PROMPT_TEMPLATE = """You are extending a research pilot on multi-agent LLM feasibility testing. You previously classified a problem as constraint type {ctype}. Now produce a FULL construction: F1 (feasible)/F2 (feasible-but-constrained)/F3 (certified infeasible) variants, in a schema designed so a verifier can independently recompute every claim (it will NOT trust your reasoning, only re-derive ground truth from the code and data you provide).

Reply with ONLY a JSON object matching EXACTLY this schema (no markdown fences, no prose outside the JSON). All *_code fields are raw Python source text (as a JSON string, properly escaped), using ONLY built-in types and operators -- no imports, no I/O, pure functions only:

{schema}

Critical requirements:
- oracle_code must be genuinely correct for the ORIGINAL problem -- it will be checked against the problem's own known test cases before anything else is trusted.
- All code must be self-contained pure Python (def statements only, no imports, no classes, no global state).
- For C2: state_input/init_state_code/apply_op_code must define a state space small enough to brute-force (the verifier will BFS from the initial state applying apply_op in all argument combinations to find shortest paths -- if the space is too large this will time out and the item will be rejected)."""


def build_messages(ctype: str, title: str, problem_text: str, feedback: str | None = None) -> list[dict]:
    system = SYSTEM_PROMPT_TEMPLATE.format(ctype=ctype, schema=SCHEMAS[ctype])
    user = f"Problem title: {title}\n\nProblem statement:\n{problem_text[:5000]}"
    if feedback:
        user += (
            "\n\nYour previous attempt at this problem was rejected by an automated "
            f"verifier that independently re-executes your code. Specific failure:\n{feedback}\n\n"
            "Produce a corrected, complete construction from scratch (same JSON schema) "
            "that fixes this specific issue. Common causes: a field's JSON type doesn't "
            "match what your code expects (e.g. a string serialized as a bare number), "
            "an input too small/large for the stated capacity, or a state space too big "
            "to brute-force. Double check types and sizes before replying."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(text: str) -> dict:
    first, last = text.index("{"), text.rindex("}")
    return json.loads(text[first:last + 1])


def construct_one(ctype: str, title: str, problem_text: str, max_tokens: int = 8000) -> dict:
    messages = build_messages(ctype, title, problem_text)
    text = ""
    for attempt_tokens in (max_tokens, max_tokens * 2):
        resp = client.chat.completions.create(
            model="gpt-5.4-mini", messages=messages, max_completion_tokens=attempt_tokens,
        )
        text = resp.choices[0].message.content or ""
        if text.strip():
            break
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return {"constraint_type": "PARSE_ERROR", "_raw": text[:2000], "_err": str(e)}


if __name__ == "__main__":
    import sys
    rows = [json.loads(l) for l in open(ROOT / "tasks" / "stratified_classification.jsonl")]
    lcb = {r["question_id"]: r for r in json.load(open(ROOT / "tasks" / "stratified_lcb_full.json"))}
    apps = {r["id"]: r for r in json.load(open(ROOT / "tasks" / "stratified_apps_full.json"))}

    by_type: dict[str, list[dict]] = {}
    for r in rows:
        if r["constraint_type"] in ("C1", "C2", "C3", "C4"):
            by_type.setdefault(r["constraint_type"], []).append(r)

    pilot_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    out_path = ROOT / "tasks" / "stratified_construction_pilot.jsonl"
    with open(out_path, "w") as out:
        for ctype in ["C1", "C2", "C3", "C4"]:
            top = sorted(by_type[ctype], key=lambda r: -r["confidence"])[:2]
            for item in top:
                if item["source"] == "livecodebench":
                    src = lcb[item["id"]]
                    title, text = src["title"], src["content"]
                else:
                    src = apps[item["id"]]
                    title, text = f"apps_{item['id']}", src["question"]
                print(f"constructing {ctype} {item['source']}:{item['id']} ({title[:40]})...")
                result = construct_one(ctype, title, text)
                record = {"source": item["source"], "id": item["id"], "title": title, **result}
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"  -> {result.get('constraint_type', 'ERR')}")
    print(f"wrote {out_path}")
