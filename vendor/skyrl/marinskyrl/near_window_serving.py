"""Qualify a rendered near-window request through the native serving actors."""

import asyncio
import csv
import hashlib
import io
import json
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class GpuMemory:
    gpu_uuid: str
    used_mib: float
    total_mib: float
    hostname: str | None = None


class CompletionEngine(Protocol):
    async def get_served_model_name(self) -> str: ...

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...


class PromptTokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool, return_dict: bool) -> list[int]: ...


@dataclass(frozen=True)
class NearWindowProbeSettings:
    engines: int
    minimum_prompt_tokens: int
    max_new_tokens: int
    context_tokens: int
    temperature: float
    top_p: float
    request_timeout: float
    memory_sample_interval: float
    receipt_uri: str


def node_gpu_memory() -> list[GpuMemory]:
    """Read allocated host GPU memory without creating a CUDA context."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return [
        GpuMemory(hostname=socket.gethostname(), gpu_uuid=uuid.strip(), used_mib=float(used), total_mib=float(total))
        for uuid, used, total in csv.reader(io.StringIO(result.stdout))
    ]


remote_gpu_memory = ray.remote(num_cpus=0)(node_gpu_memory)


async def capture_gpu_memory() -> list[GpuMemory]:
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
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if not settings.minimum_prompt_tokens <= len(tokens) <= settings.context_tokens - settings.max_new_tokens:
        raise ValueError(f"Rendered prompt has {len(tokens)} tokens outside the declared near-window budget")
    return tokens


async def run_near_window_probe(
    engines: list[CompletionEngine], tokenizer: PromptTokenizer, settings: NearWindowProbeSettings, ready_seconds: float,
    *, receipt_directory: Path, memory_reader: Callable[[], Awaitable[list[GpuMemory]]] = capture_gpu_memory,
) -> dict[str, float]:
    """Serve on every rank and persist one near-window request with memory evidence.

    Returns:
        Metrics for rendered input tokens, completed output tokens, and ready engines.
    """
    config = settings
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
    memory_snapshots: list[list[GpuMemory]] = []
    started = time.monotonic()
    stopped = asyncio.Event()
    served_model_names: list[str] = []

    async def memory_sample() -> None:
        gpus = await memory_reader()
        if len(gpus) != config.engines or len({gpu.gpu_uuid for gpu in gpus}) != config.engines:
            raise ValueError("GPU memory evidence does not cover the complete inference group")
        memory_snapshots.append(gpus)
        receipt["memory_samples"].append({"elapsed_seconds": time.monotonic() - started, "gpus": [asdict(gpu) for gpu in gpus]})

    async def sample_memory() -> None:
        while not stopped.is_set():
            await memory_sample()
            try:
                await asyncio.wait_for(stopped.wait(), timeout=config.memory_sample_interval)
            except TimeoutError:
                continue

    async def complete(rank: int, prompt: list[int], output_tokens: int) -> dict:
        before = time.monotonic()
        response = await asyncio.wait_for(engines[rank].completion({"json": {
            "model": served_model_names[rank], "prompt": prompt, "max_tokens": output_tokens,
            "temperature": config.temperature, "top_p": config.top_p, "ignore_eos": True, "stream": False,
        }, "headers": {"Content-Type": "application/json"}}), timeout=config.request_timeout)
        if response.get("error") or not response.get("choices"):
            raise RuntimeError(f"Serving request on rank {rank} failed: {response}")
        if response["model"] != served_model_names[rank]:
            raise ValueError("Completion model differs from the actor's registered serving name")
        usage = response["usage"]
        if usage["completion_tokens"] != output_tokens:
            raise ValueError(f"Serving output differs from its token budget: {usage}")
        if usage["prompt_tokens"] != len(prompt):
            raise ValueError(f"Server prompt usage differs from the rendered input: {usage}")
        return {"rank": rank, "model": response["model"], "usage": usage, "elapsed_seconds": time.monotonic() - before}

    try:
        served_model_names = await asyncio.gather(*(engine.get_served_model_name() for engine in engines))
        if len(set(served_model_names)) != 1 or not served_model_names[0]:
            raise ValueError("Inference actors disagree on the registered serving model")
        receipt["served_model_names"] = served_model_names
        await memory_sample()
        readiness_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain why one plus one is two."}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(sample_memory())
            try:
                receipt["rank_readiness_requests"] = await asyncio.gather(*(
                    complete(rank, readiness_tokens, 16) for rank in range(len(engines))
                ))
                receipt["near_window_request"] = await complete(0, tokens, config.max_new_tokens)
            finally:
                stopped.set()
        await memory_sample()
        receipt["peak_used_mib_by_gpu"] = {
            uuid: max(gpu.used_mib for sample in memory_snapshots for gpu in sample if gpu.gpu_uuid == uuid)
            for uuid in {gpu.gpu_uuid for gpu in memory_snapshots[0]}
        }
        receipt["result"] = "passed"
    except Exception as error:
        receipt.update(result="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        contents = json.dumps(receipt, indent=2) + "\n"
        (receipt_directory / "near-window-serving-receipt.json").write_text(contents)
        StoragePath(config.receipt_uri).write_bytes(contents.encode())
    return {"serving_probe/input_tokens": len(tokens), "serving_probe/output_tokens": config.max_new_tokens,
            "serving_probe/ready_engines": len(engines)}
