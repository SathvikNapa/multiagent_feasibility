"""Thin wrapper over the OpenAI Batch API: submit a list of chat-completion
requests as one batch job, poll until done, return parsed results keyed by
custom_id. Batch API gives a ~50% cost discount and avoids one-at-a-time
latency for large, non-interactive jobs like this dataset's construction
passes.
"""
from __future__ import annotations

import json
import os
import tempfile
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


def run_batch(requests: list[dict], poll_interval_s: int = 20, description: str = "") -> dict[str, dict]:
    """requests: [{"custom_id": str, "model": str, "messages": [...], "max_completion_tokens": int}, ...]
    Returns {custom_id: {"content": str, "error": str|None}}."""
    if not requests:
        return {}

    lines = []
    for r in requests:
        lines.append(json.dumps({
            "custom_id": r["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": r["model"],
                "messages": r["messages"],
                "max_completion_tokens": r["max_completion_tokens"],
            },
        }))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        tf.write("\n".join(lines) + "\n")
        batch_input_path = tf.name

    with open(batch_input_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    os.unlink(batch_input_path)

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description} if description else None,
    )
    print(f"[batch] submitted {batch.id} ({len(requests)} requests){': ' + description if description else ''}")

    while True:
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"[batch] {batch.id} status={batch.status} completed={counts.completed}/{counts.total} failed={counts.failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval_s)

    results: dict[str, dict] = {}
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["custom_id"]
            resp = rec.get("response") or {}
            if resp.get("status_code") == 200:
                body = resp["body"]
                text = body["choices"][0]["message"]["content"] or ""
                results[cid] = {"content": text, "error": None}
            else:
                results[cid] = {"content": "", "error": json.dumps(rec.get("error") or resp)}
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).text
        for line in err_content.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            if cid and cid not in results:
                results[cid] = {"content": "", "error": json.dumps(rec.get("response") or rec)}

    for r in requests:
        if r["custom_id"] not in results:
            results[r["custom_id"]] = {"content": "", "error": "missing from batch output (likely still pending or dropped)"}

    return results
