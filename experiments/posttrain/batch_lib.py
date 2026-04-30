# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Batch API helpers for chat-completions workloads.

Extracted from the pattern in `experiments/posttrain/judge_gpt51_batch.py`
into a reusable module. Use this for any new OpenAI workload so we get the
50% Batch-API discount with a 24-hour SLA.

Workflow:
    from openai import OpenAI
    from pathlib import Path
    import batch_lib as bl

    client = OpenAI()
    job_dir = Path("experiments/posttrain/my_job")

    # 1. Build requests
    requests = [
        bl.build_request(
            custom_id=f"item::{i:07d}",
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Item {i}"},
            ],
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        for i in range(N)
    ]

    # 2. Submit (writes requests.jsonl, uploads, creates batch, persists state)
    state = bl.submit(client, requests, job_dir, metadata={"project": "my_project"})

    # 3. Poll until terminal (blocking, resumable)
    state = bl.poll(client, job_dir, interval=60)

    # 4. Collect outputs (downloads output.jsonl, returns parsed list)
    entries = bl.collect(client, job_dir)

    # 5. Extract content per request
    for entry in entries:
        content = bl.extract_content(entry)  # None on error
        if content is None:
            continue
        # join back to your own data via custom_id
        custom_id = entry["custom_id"]
        ...

Limits (OpenAI Batch API):
    - 200 MB per input file
    - 50,000 requests per batch
    - 24h completion window (only supported value)
    - Output file retained on OpenAI servers; download once and persist locally.

Reasoning models (gpt-5.x, o-series) quirks:
    - Reject `max_tokens` and `temperature`; use `max_completion_tokens`.
    - Include `reasoning_effort` to control reasoning tokens. **Default here
      is "none". DO NOT override this default** unless you have explicitly
      decided reasoning will help THIS particular call (it almost never does
      for our workloads). Reasoning tokens are pure overhead that OpenAI
      bills for and the user never sees. The single-day cost blowup on
      2026-04-20 ($53 on the gpt-5.1-batch-output line) came from one
      script overriding this default to "medium" on a generation step.
      See `.agents/logbooks/gpt5_correlation.md` EXP-028g post-mortem.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger("batch_lib")

BATCH_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"  # only value OpenAI supports currently
MAX_FILE_MB = 200
MAX_REQUESTS_PER_BATCH = 50_000
TERMINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #


def is_next_gen(model: str) -> bool:
    """Models that use `max_completion_tokens` and reject a temperature override."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def build_request(
    custom_id: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    response_format: dict | None = None,
    reasoning_effort: str | None = "none",
) -> dict:
    """Build one JSONL entry for the Batch API.

    Args:
        custom_id: Your join key into the output. Must be unique in the batch.
        model: OpenAI model name.
        messages: chat-completions-style message list.
        max_tokens: Output token cap. Translates to `max_completion_tokens`
            for reasoning models automatically.
        temperature: Sampling temperature. Ignored for reasoning models
            (they reject overrides).
        response_format: e.g., `{"type": "json_object"}` to force strict JSON.
        reasoning_effort: For gpt-5.x / o-series only. "none" disables
            reasoning tokens (recommended for structured workloads). Pass
            None to omit the field entirely.

    Returns:
        dict conforming to the OpenAI Batch API input schema.
    """
    body: dict[str, Any] = {"model": model, "messages": messages}
    if is_next_gen(model):
        body["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": body,
    }


# --------------------------------------------------------------------------- #
# Submit / poll / collect
# --------------------------------------------------------------------------- #


def submit(
    client: OpenAI,
    requests: list[dict],
    job_dir: Path,
    metadata: dict[str, str] | None = None,
) -> dict:
    """Write requests.jsonl, upload, create a batch, persist state.

    State file at `{job_dir}/batch_state.json` contains batch_id,
    input_file_id, status, submitted_at, metadata. Safe to re-read if the
    submitting process dies.

    Raises if file size > 200 MB or request count > 50,000 (OpenAI limits).
    """
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "requests.jsonl"
    with input_path.open("w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info("wrote %s (%d requests, %.1f MB)", input_path, len(requests), size_mb)

    if size_mb > MAX_FILE_MB:
        raise RuntimeError(
            f"Batch input {size_mb:.1f} MB > {MAX_FILE_MB} MB cap; split into chunks."
        )
    if len(requests) > MAX_REQUESTS_PER_BATCH:
        raise RuntimeError(
            f"{len(requests)} requests > {MAX_REQUESTS_PER_BATCH} cap; split into chunks."
        )

    with input_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info("uploaded input_file_id=%s", file_obj.id)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata=metadata or {},
    )

    state = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "status": batch.status,
        "num_requests": len(requests),
        "submitted_at": datetime.datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
        "endpoint": BATCH_ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
    }
    _save_state(job_dir, state)
    logger.info("created batch %s (status=%s, %d requests)", batch.id, batch.status, len(requests))
    return state


def poll(
    client: OpenAI,
    job_dir: Path,
    interval: int = 60,
    max_wait: float = 25 * 3600,
) -> dict:
    """Block until the batch reaches a terminal state. Returns final state.

    Resumable: rereads state every loop, so if this dies mid-poll another
    process can take over. Default `max_wait` is slightly over 24h (Batch
    API SLA).
    """
    state = _load_state(job_dir)
    start = time.time()
    while True:
        batch = client.batches.retrieve(state["batch_id"])
        counts = getattr(batch, "request_counts", None)
        done = getattr(counts, "completed", None) if counts else None
        failed = getattr(counts, "failed", None) if counts else None
        total = getattr(counts, "total", None) if counts else None
        if total is None:
            total = state["num_requests"]
        logger.info(
            "batch %s status=%s done=%s/%s failed=%s",
            batch.id, batch.status, done if done is not None else "-", total,
            failed if failed is not None else "-",
        )
        state["status"] = batch.status
        state["output_file_id"] = getattr(batch, "output_file_id", None)
        state["error_file_id"] = getattr(batch, "error_file_id", None)
        _save_state(job_dir, state)
        if batch.status in TERMINAL_STATES:
            return state
        if time.time() - start > max_wait:
            raise TimeoutError(
                f"Batch {batch.id} status={batch.status} after {max_wait}s"
            )
        time.sleep(interval)


def collect(client: OpenAI, job_dir: Path) -> list[dict]:
    """Download output.jsonl for a completed batch. Returns list of entries.

    Each entry is one dict from the batch output: it has `custom_id`,
    `response`, and optionally `error`. Use `extract_content(entry)` to
    pull the assistant message.
    """
    state = _load_state(job_dir)
    if state["status"] != "completed":
        raise RuntimeError(
            f"batch status is {state['status']}, not completed; run poll() first"
        )
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("batch completed but no output_file_id in state")

    content = client.files.content(output_file_id).content
    output_path = job_dir / "output.jsonl"
    output_path.write_bytes(content)
    logger.info("downloaded %s (%d bytes)", output_path, len(content))

    entries: list[dict] = []
    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    logger.info("parsed %d entries from output", len(entries))
    return entries


def extract_content(entry: dict) -> str | None:
    """Pull the assistant message text from one batch-output entry.

    Returns None if the entry has an error, a non-200 status, or is missing
    the usual choices[0].message.content structure. Caller should skip /
    handle those cases explicitly.
    """
    if entry.get("error"):
        return None
    response = entry.get("response") or {}
    if response.get("status_code") != 200:
        return None
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    return (choices[0].get("message") or {}).get("content")


def extract_usage(entry: dict) -> dict:
    """Return per-entry token usage, including reasoning_tokens for gpt-5.x."""
    response = entry.get("response") or {}
    body = response.get("body") or {}
    usage = body.get("usage") or {}
    ctd = usage.get("completion_tokens_details") or {}
    ptd = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": int(ctd.get("reasoning_tokens") or 0),
        "cached_tokens": int(ptd.get("cached_tokens") or 0),
    }


# --------------------------------------------------------------------------- #
# State I/O
# --------------------------------------------------------------------------- #


def _state_path(job_dir: Path) -> Path:
    return job_dir / "batch_state.json"


def _load_state(job_dir: Path) -> dict:
    path = _state_path(job_dir)
    if not path.exists():
        raise FileNotFoundError(f"no batch_state.json at {path}; did you submit()?")
    return json.loads(path.read_text())


def _save_state(job_dir: Path, state: dict) -> None:
    _state_path(job_dir).write_text(json.dumps(state, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Higher-level convenience: status across many jobs
# --------------------------------------------------------------------------- #


def refresh_status(client: OpenAI, job_dir: Path) -> dict:
    """Fetch current batch status and rewrite local state. Returns new state."""
    state = _load_state(job_dir)
    batch = client.batches.retrieve(state["batch_id"])
    state["status"] = batch.status
    state["output_file_id"] = getattr(batch, "output_file_id", None)
    state["error_file_id"] = getattr(batch, "error_file_id", None)
    _save_state(job_dir, state)
    return state


def list_recent_batches(client: OpenAI, limit: int = 20) -> list[dict]:
    """Recovery helper: list recent batches you submitted, including metadata."""
    batches = client.batches.list(limit=limit)
    out = []
    for b in batches.data:
        out.append({
            "id": b.id,
            "status": b.status,
            "input_file_id": b.input_file_id,
            "output_file_id": getattr(b, "output_file_id", None),
            "metadata": getattr(b, "metadata", None) or {},
        })
    return out
