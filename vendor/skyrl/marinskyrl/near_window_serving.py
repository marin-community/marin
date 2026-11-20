"""Qualify a rendered near-window request through the native serving actors."""

import asyncio
import csv
import hashlib
import io
import json
import os
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from rigging.filesystem.storage_path import StoragePath


class CompletionEngine(Protocol):
    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...


class PromptTokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool) -> list[int]: ...


@dataclass(frozen=True)
class NearWindowProbeSettings:
    engines: int
    minimum_prompt_tokens: int
    max_new_tokens: int
    context_tokens: int
    served_model_name: str
    temperature: float
    top_p: float
    request_timeout: float
    memory_sample_interval: float
    receipt_uri: str


def node_gpu_memory() -> list[dict]:
    """Read allocated host GPU memory without creating a CUDA context."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return [
        {"hostname": socket.gethostname(), "gpu_uuid": uuid.strip(), "used_mib": float(used), "total_mib": float(total)}
        for uuid, used, total in csv.reader(io.StringIO(result.stdout))
    ]


remote_gpu_memory = ray.remote(num_cpus=0)(node_gpu_memory)


async def capture_gpu_memory() -> list[dict]:
    """Sample every allocated Ray host, including workers outside the driver host."""
    requests = [
        remote_gpu_memory.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
        ).remote()
        for node in ray.nodes() if node["Alive"]
    ]
    hosts = await asyncio.wait_for(asyncio.gather(*requests), timeout=30)
    return [gpu for host in hosts for gpu in host]


def rendered_near_window_prompt(tokenizer: PromptTokenizer, settings: NearWindowProbeSettings) -> list[int]:
    """Render a synthetic long chat once, retaining the complete template boundaries."""
    messages = [
        {"role": "system", "content": "Read the input and continue with a short explanation."},
        {"role": "user", "content": " x" * settings.minimum_prompt_tokens + "\nExplain the final symbol."},
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not settings.minimum_prompt_tokens <= len(tokens) <= settings.context_tokens - settings.max_new_tokens:
        raise ValueError(f"Rendered prompt has {len(tokens)} tokens outside the declared near-window budget")
    return tokens


async def run_near_window_probe(
    engines: list[CompletionEngine], tokenizer: PromptTokenizer, settings: dict[str, Any], ready_seconds: float,
    *, memory_reader: Callable[[], Awaitable[list[dict]]] = capture_gpu_memory,
) -> dict[str, float]:
    """Require every rank to serve, then qualify one full rendered prompt and its memory use."""
    config = NearWindowProbeSettings(**settings)
    if len(engines) != config.engines:
        raise ValueError(f"Expected {config.engines} ready engines, got {len(engines)}")
    tokens = rendered_near_window_prompt(tokenizer, config)
    receipt: dict[str, Any] = {
        "result": "running", "purpose": "serving-only near-window qualification; no trainer or judge",
        "ready_engines": len(engines), "ready_seconds": ready_seconds,
        "context_tokens": config.context_tokens, "requested_output_tokens": config.max_new_tokens,
        "rendered_input_tokens": len(tokens),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "prompt_rendering": "Pinned tokenizer chat template, add_generation_prompt=True; exact rendered IDs sent to completions",
        "memory_samples": [],
    }
    started = time.monotonic()
    stopped = asyncio.Event()

    async def memory_sample() -> None:
        gpus = await memory_reader()
        if len(gpus) != config.engines or len({gpu["gpu_uuid"] for gpu in gpus}) != config.engines:
            raise ValueError("GPU memory evidence does not cover the complete inference group")
        receipt["memory_samples"].append({"elapsed_seconds": time.monotonic() - started, "gpus": gpus})

    async def sample_memory() -> None:
        while not stopped.is_set():
            await memory_sample()
            try:
                await asyncio.wait_for(stopped.wait(), timeout=config.memory_sample_interval)
            except TimeoutError:
                continue

    async def complete(rank: int, prompt: str | list[int], output_tokens: int) -> dict:
        before = time.monotonic()
        response = await asyncio.wait_for(engines[rank].completion({"json": {
            "model": config.served_model_name, "prompt": prompt, "max_tokens": output_tokens,
            "temperature": config.temperature, "top_p": config.top_p, "ignore_eos": True, "stream": False,
        }, "headers": {"Content-Type": "application/json"}}), timeout=config.request_timeout)
        if response.get("error") or not response.get("choices"):
            raise RuntimeError(f"Serving request on rank {rank} failed: {response}")
        usage = response["usage"]
        if usage["completion_tokens"] != output_tokens:
            raise ValueError(f"Serving output differs from its token budget: {usage}")
        if isinstance(prompt, list) and usage["prompt_tokens"] != len(prompt):
            raise ValueError(f"Server prompt usage differs from the rendered input: {usage}")
        return {"rank": rank, "usage": usage, "elapsed_seconds": time.monotonic() - before}

    try:
        await memory_sample()
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(sample_memory())
            try:
                receipt["rank_readiness_requests"] = await asyncio.gather(*(
                    complete(rank, "Explain why one plus one is two.", 16) for rank in range(len(engines))
                ))
                receipt["near_window_request"] = await complete(0, tokens, config.max_new_tokens)
            finally:
                stopped.set()
        await memory_sample()
        receipt["peak_used_mib_by_gpu"] = {
            uuid: max(gpu["used_mib"] for sample in receipt["memory_samples"] for gpu in sample["gpus"] if gpu["gpu_uuid"] == uuid)
            for uuid in {gpu["gpu_uuid"] for gpu in receipt["memory_samples"][0]["gpus"]}
        }
        receipt["result"] = "passed"
    except Exception as error:
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        contents = json.dumps(receipt, indent=2) + "\n"
        (Path(os.environ["IRIS_OUTPUT_DIR"]) / "near-window-serving-receipt.json").write_text(contents)
        StoragePath(config.receipt_uri).write_bytes(contents.encode())
    return {"serving_probe/input_tokens": len(tokens), "serving_probe/output_tokens": config.max_new_tokens,
            "serving_probe/ready_engines": len(engines)}
