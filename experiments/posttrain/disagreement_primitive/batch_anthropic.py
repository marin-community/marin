# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal Anthropic Batch API helper. 50% discount, up to 24h SLA.

Workflow:
    import batch_anthropic as ba

    requests = [
        ba.build_request(
            custom_id=f"item::{i:07d}",
            model="claude-sonnet-4-6",
            system=SYSTEM,
            messages=[{"role": "user", "content": "..."}],
            max_tokens=600,
            tools=[JUDGMENT_TOOL],
            tool_choice={"type": "tool", "name": "submit_judgment"},
            thinking={"type": "disabled"},
        )
        for i in range(N)
    ]

    state = ba.submit(api_key, requests, job_dir, name="my_batch")
    state = ba.poll(api_key, job_dir, interval=60)  # blocks until terminal
    entries = ba.collect(api_key, job_dir)  # list of result dicts
    args = ba.extract_tool_args(entries[0])  # parsed input dict, or None
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

API_BASE = "https://api.anthropic.com/v1"
HEADERS_BASE = {"anthropic-version": "2023-06-01", "content-type": "application/json"}


_EPHEMERAL = {"type": "ephemeral"}

JUDGE_USER_PREFIX_MARKER = "USER QUERY:"


def prefix_before(content: str, marker: str = JUDGE_USER_PREFIX_MARKER) -> str | None:
    """Return ``content`` up to (but not including) the first occurrence of
    ``marker``, or None if not found / marker is at position 0.

    The default marker is the boundary used by every DART judge user prompt
    between the static spec/examples/rubric preamble and the variable
    USER QUERY / ASSISTANT RESPONSE per-cell tail. Pass the return value to
    ``build_request(..., cache_user_prefix=...)`` to extend the ephemeral
    cache breakpoint past system+tools through the static user-message
    preamble.
    """
    idx = content.find(marker)
    return content[:idx] if idx > 0 else None


def build_request(
    custom_id: str,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    thinking: dict[str, Any] | None = None,
    temperature: float = 0,
    cache: bool = True,
    cache_user_prefix: str | None = None,
) -> dict[str, Any]:
    """Build a single Anthropic Messages-API request envelope.

    With ``cache=True`` (the default) the request opts in to ephemeral prompt
    caching on the shared prefix. The cached prefix runs system → last tool;
    cells 2..N of a homogeneous batch read it at 10% input rate.

    If ``cache_user_prefix`` is given, ``messages`` must be a single user
    message whose ``content`` is a string starting with that prefix. The
    prefix is split off into its own cache_control block; the remainder
    (typically per-cell query + response) stays uncached. This extends the
    cache breakpoint past system+tools through the static user-message
    preamble — usually the largest savings on judge calls.
    """
    if cache:
        system_value: Any = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
    else:
        system_value = system

    if cache_user_prefix is not None:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("cache_user_prefix requires a single user message")
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise ValueError("cache_user_prefix requires messages[0].content to be a string")
        if not content.startswith(cache_user_prefix):
            raise ValueError("cache_user_prefix must be a prefix of messages[0].content")
        suffix = content[len(cache_user_prefix) :]
        blocks: list[dict[str, Any]] = [{"type": "text", "text": cache_user_prefix, "cache_control": _EPHEMERAL}]
        if suffix:
            blocks.append({"type": "text", "text": suffix})
        messages_value: list[dict[str, Any]] = [{"role": "user", "content": blocks}]
    else:
        messages_value = messages

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_value,
        "messages": messages_value,
    }
    if tools is not None:
        if cache and tools:
            params["tools"] = [*tools[:-1], {**tools[-1], "cache_control": _EPHEMERAL}]
        else:
            params["tools"] = tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if thinking is not None:
        params["thinking"] = thinking
    return {"custom_id": custom_id, "params": params}


def submit(api_key: str, requests: list[dict[str, Any]], job_dir: Path, name: str = "batch") -> dict[str, Any]:
    """Submit a batch. Persists requests + state to job_dir. Returns state dict."""
    job_dir.mkdir(parents=True, exist_ok=True)

    requests_path = job_dir / f"{name}_requests.jsonl"
    with requests_path.open("w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    headers = {**HEADERS_BASE, "x-api-key": api_key}
    resp = httpx.post(
        f"{API_BASE}/messages/batches",
        headers=headers,
        json={"requests": requests},
        timeout=600.0,
    )
    if resp.status_code not in (200, 201):
        snippet = resp.text[:600].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"submit HTTP {resp.status_code}: {snippet}")
    batch = resp.json()

    state = {
        "name": name,
        "batch_id": batch["id"],
        "submitted_at": time.time(),
        "n_requests": len(requests),
        "requests_path": str(requests_path),
        "batch_response": batch,
    }
    state_path = job_dir / f"{name}_state.json"
    state_path.write_text(json.dumps(state, indent=2))
    return state


def get_status(api_key: str, batch_id: str) -> dict[str, Any]:
    headers = {**HEADERS_BASE, "x-api-key": api_key}
    resp = httpx.get(f"{API_BASE}/messages/batches/{batch_id}", headers=headers, timeout=60.0)
    if resp.status_code != 200:
        snippet = resp.text[:400].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"status HTTP {resp.status_code}: {snippet}")
    return resp.json()


def poll(
    api_key: str, job_dir: Path, name: str = "batch", interval: float = 60.0, timeout: float = 86400.0
) -> dict[str, Any]:
    """Block until batch reaches terminal status. Returns latest batch dict."""
    state_path = job_dir / f"{name}_state.json"
    state = json.loads(state_path.read_text())
    batch_id = state["batch_id"]
    t0 = time.time()
    while True:
        b = get_status(api_key, batch_id)
        status = b.get("processing_status")
        counts = b.get("request_counts", {})
        if status == "ended":
            return b
        if time.time() - t0 > timeout:
            raise TimeoutError(f"batch {batch_id} not done after {timeout}s")
        elapsed = time.time() - t0
        print(f"  [batch_anthropic.poll] {name} status={status} counts={counts} elapsed={elapsed:.0f}s")
        time.sleep(interval)


def collect(api_key: str, job_dir: Path, name: str = "batch") -> list[dict[str, Any]]:
    """Download results for a completed batch. Returns parsed list."""
    state_path = job_dir / f"{name}_state.json"
    state = json.loads(state_path.read_text())
    batch_id = state["batch_id"]
    b = get_status(api_key, batch_id)
    if b.get("processing_status") != "ended":
        raise RuntimeError(f"batch {batch_id} not ended yet (status={b.get('processing_status')})")
    results_url = b.get("results_url")
    if not results_url:
        raise RuntimeError(f"batch {batch_id} ended but no results_url")

    headers = {"anthropic-version": "2023-06-01", "x-api-key": api_key}
    resp = httpx.get(results_url, headers=headers, timeout=600.0, follow_redirects=True)
    if resp.status_code != 200:
        snippet = resp.text[:400].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"results HTTP {resp.status_code}: {snippet}")

    results_path = job_dir / f"{name}_results.jsonl"
    results_path.write_bytes(resp.content)

    entries = []
    for line in results_path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def extract_tool_args(entry: dict[str, Any], tool_name: str = "submit_judgment") -> dict[str, Any] | None:
    """Pull the tool_use input dict from a single batch result entry. None on error."""
    result = entry.get("result", {})
    if result.get("type") != "succeeded":
        return None
    msg = result.get("message", {})
    blocks = msg.get("content", []) or []
    tu = next((b for b in blocks if b.get("type") == "tool_use" and b.get("name") == tool_name), None)
    if tu is None:
        return None
    args = tu.get("input")
    return args if isinstance(args, dict) else None


def usage_of(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the per-result usage dict. Keys include ``input_tokens``,
    ``output_tokens``, ``cache_creation_input_tokens``, and
    ``cache_read_input_tokens`` when caching is in play."""
    return ((entry.get("result") or {}).get("message") or {}).get("usage") or {}


def cache_hit_rate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cache-hit metrics across a batch's results. Useful for
    sanity-checking that ephemeral caching actually took effect."""
    total_input = 0
    total_cache_read = 0
    total_cache_create = 0
    n = 0
    for entry in entries:
        u = usage_of(entry)
        if not u:
            continue
        n += 1
        total_input += int(u.get("input_tokens", 0) or 0)
        total_cache_read += int(u.get("cache_read_input_tokens", 0) or 0)
        total_cache_create += int(u.get("cache_creation_input_tokens", 0) or 0)
    cacheable_total = total_input + total_cache_read + total_cache_create
    return {
        "n_results": n,
        "input_tokens_uncached": total_input,
        "cache_read_tokens": total_cache_read,
        "cache_creation_tokens": total_cache_create,
        "cache_hit_fraction": (total_cache_read / cacheable_total) if cacheable_total else 0.0,
    }
