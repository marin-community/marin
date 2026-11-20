"""Bounded Mochi serving proof through SkyRL actors, without an eval judge."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

from rigging.filesystem.storage_path import StoragePath


class CompletionEngine(Protocol):
    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...


async def run_serving_probe(
    engines: list[CompletionEngine], settings: dict[str, Any], ready_seconds: float
) -> dict[str, Any]:
    """Require a completion on every rank and persist measured token counts."""
    expected = settings["engines"]
    if len(engines) != expected:
        raise ValueError(f"Expected {expected} ready engines, got {len(engines)}")
    print(f"MOCHI_ENGINE_PROBE_READY={len(engines)}", flush=True)
    semaphore = asyncio.Semaphore(settings["concurrency"])

    async def complete(index: int) -> dict[str, Any]:
        rank = max(index - 1, 0) % expected
        async with semaphore:
            started = time.monotonic()
            response = await asyncio.wait_for(
                engines[rank].completion({"json": {
                    "model": "model",
                    "prompt": f"Write a Python function that sums a list of integers. Example {index}:\n",
                    "max_tokens": settings["max_new_tokens"],
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": False,
                }, "headers": {"Content-Type": "application/json"}}),
                timeout=settings["request_timeout"],
            )
            if response.get("error") or not response.get("choices"):
                raise RuntimeError(f"Completion {index} on rank {rank} failed: {response}")
            usage = response.get("usage", {})
            if usage.get("completion_tokens") != settings["max_new_tokens"]:
                raise RuntimeError(f"Completion {index} has unexpected token usage: {usage}")
            return {"request": index, "rank": rank, "usage": usage,
                    "elapsed_seconds": time.monotonic() - started}

    warmup = await complete(0)
    started = time.monotonic()
    requests = await asyncio.gather(*(complete(index) for index in range(1, settings["measured_requests"] + 1)))
    elapsed = time.monotonic() - started
    tokens = sum(item["usage"]["completion_tokens"] for item in requests)
    receipt = {
        "result": "passed", "purpose": "serving proof only; no evaluation score",
        "ready_engines": len(engines), "ready_seconds": ready_seconds,
        "requests_completed": len(requests) + 1, "warmup": warmup,
        "requests": requests, "measured_completion_tokens": tokens,
        "elapsed_seconds": elapsed, "tokens_per_second": tokens / elapsed,
        "transport": "SkyRL Ray actor completion RPC; no HTTP status is inferred",
    }
    contents = json.dumps(receipt, indent=2) + "\n"
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "mochi-engine-serving-receipt.json"
    output.write_text(contents)
    StoragePath(settings["receipt_uri"]).write_bytes(contents.encode())
    print("MOCHI_ENGINE_PROBE_RESULT=" + json.dumps(receipt), flush=True)
    return {"serving_smoke/requests_completed": len(requests) + 1,
            "serving_smoke/completion_tokens": tokens,
            "serving_smoke/tokens_per_second": tokens / elapsed}
