# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured pool readiness and immutable replacement receipt history."""

import asyncio
import hashlib
import json
import time

import httpx
from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath

WARM_CONCURRENCY = 128
WARM_OUTPUT_TOKENS = 256


async def warm_throughput(
    endpoint: str, model: str, gpu_count: int, api_key: str | None = None, concurrency: int = WARM_CONCURRENCY
) -> dict:
    """Measure completed output tokens over a concurrent warm batch, including HTTP overhead."""
    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(
        timeout=1800, limits=httpx.Limits(max_connections=concurrency), headers=headers
    ) as client:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Continue counting integers separated by spaces, starting at 1."}],
            "max_tokens": WARM_OUTPUT_TOKENS,
            "temperature": 0.7,
        }
        if gpu_count:
            body["ignore_eos"] = True

        async def request() -> dict:
            response = await client.post(endpoint.rstrip("/") + "/chat/completions", json=body)
            response.raise_for_status()
            return response.json()["usage"]

        await request()
        started = time.monotonic()
        usages = await asyncio.gather(*(request() for _ in range(concurrency)))
        elapsed = time.monotonic() - started
    output = sum(usage["completion_tokens"] for usage in usages)
    return {
        "requests": len(usages),
        "concurrency": concurrency,
        "output_tokens": output,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output / elapsed,
        "output_tokens_per_second_per_gpu": output / elapsed / gpu_count if gpu_count else None,
        "basis": "completed decode output tokens / concurrent-batch wall time, including prefill and HTTP overhead",
    }


def publish_pool_receipt(registry: str, solver: Solver, receipt: dict) -> None:
    """Retain ready generations before replacing the active solver endpoint."""
    target = StoragePath(registry) / f"{solver.value}.json"
    encoded = json.dumps(receipt, sort_keys=True) + "\n"
    versions = [encoded]
    if target.exists():
        previous = target.read_text()
        previous_receipt = json.loads(previous)
        for field in ("solver", "model", "revision"):
            if previous_receipt[field] != receipt[field]:
                raise ValueError(f"Cannot replace solver pool with a different {field}")
        versions.insert(0, previous)
    history = target.parent / "history" / solver.value
    history.mkdirs()
    for version in versions:
        digest = hashlib.sha256(version.encode()).hexdigest()
        (history / f"{digest}.json").write_text(version)
    # S3 publishes each complete object on close; readers retain the previous
    # ready endpoint until this replacement succeeds.
    target.write_text(encoded)
