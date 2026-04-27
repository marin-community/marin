# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run IFBench rollouts through a local vLLM server on an Iris TPU worker."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import threading
import time
from typing import Any

import fsspec
import httpx

from experiments.ifbench.data.prepare import PreparedRow
from experiments.ifbench.rollout.backend import Rollout, SamplingConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

DEFAULT_PREPARED = "gs://marin-us-central2/scratch/ifbench/overnight_20k/prompts_20k.jsonl"
DEFAULT_CACHE_DIR = "/app/.hf_cache"
DEFAULT_SAMPLING = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
STARTUP_HEARTBEAT_SECONDS = 60


def load_prompt_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with fsspec.open(path, "rt") as f:
        for line in f:
            prompt_id = line.strip()
            if prompt_id:
                ids.add(prompt_id)
    if not ids:
        raise ValueError(f"no prompt ids found in {path}")
    return ids


def load_prompts(path: str, n: int, prompt_ids: set[str] | None = None) -> list[PreparedRow]:
    rows: list[PreparedRow] = []
    scanned = 0
    with fsspec.open(path, "rt") as f:
        for line in f:
            if scanned >= n:
                break
            scanned += 1
            row = PreparedRow(**json.loads(line))
            if prompt_ids is None or row.prompt_id in prompt_ids:
                rows.append(row)
    if scanned < n:
        raise RuntimeError(f"asked for {n} source rows but only found {scanned} in {path}")
    if prompt_ids is not None:
        found_ids = {row.prompt_id for row in rows}
        missing = prompt_ids - found_ids
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(f"{len(missing)} prompt ids were not found in first {n} rows of {path}: {sample}")
    return rows


def shard_paths(shard_output_dir: str) -> list[str]:
    pattern = f"{shard_output_dir.rstrip('/')}/part-*.jsonl"
    fs, _, paths = fsspec.core.get_fs_token_paths(pattern)
    return sorted(fs.unstrip_protocol(path) for path in paths)


def read_rollout_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fsspec.open(path, "rt") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_prompt_ids_from_shards(shard_output_dir: str) -> set[str]:
    completed: set[str] = set()
    for path in shard_paths(shard_output_dir):
        for row in read_rollout_rows(path):
            completed.add(row["prompt_id"])
    return completed


def write_rollout_shard(shard_output_dir: str, run_id: str, shard_index: int, rollouts: list[Rollout]) -> str:
    path = f"{shard_output_dir.rstrip('/')}/part-{run_id}-{shard_index:05d}.jsonl"
    write_rollouts(path, rollouts)
    return path


def write_consolidated_shards(shard_output_dir: str, output: str) -> int:
    by_prompt: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path in shard_paths(shard_output_dir):
        for row in read_rollout_rows(path):
            prompt_id = row["prompt_id"]
            if prompt_id not in by_prompt:
                order.append(prompt_id)
            by_prompt[prompt_id] = row

    with fsspec.open(output, "wt") as f:
        for prompt_id in order:
            f.write(json.dumps(by_prompt[prompt_id], ensure_ascii=False) + "\n")
    return len(by_prompt)


async def _one_completion(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    *,
    server_url: str,
    served_model_id: str,
    requested_model_id: str,
    row: PreparedRow,
    seed: int,
    sampling: SamplingConfig,
    request_timeout: float,
) -> Rollout:
    async with sem:
        try:
            response = await client.post(
                f"{server_url}/chat/completions",
                json={
                    "model": served_model_id,
                    "messages": row.messages,
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "max_tokens": sampling.max_new_tokens,
                },
                timeout=request_timeout,
            )
            if response.status_code >= 400:
                logger.warning("vLLM %s: %d %s", row.prompt_id, response.status_code, response.text[:200])
                return _empty_rollout(row, requested_model_id, seed, sampling, f"http_{response.status_code}")
            payload = response.json()
        except Exception as exc:
            logger.warning("vLLM %s exception: %s", row.prompt_id, exc)
            return _empty_rollout(row, requested_model_id, seed, sampling, type(exc).__name__)

        choices = payload.get("choices", [])
        text = ""
        finish_reason = None
        if choices:
            choice = choices[0]
            text = choice.get("message", {}).get("content", "") or ""
            finish_reason = choice.get("finish_reason")
        usage = payload.get("usage", {}) or {}
        return Rollout(
            prompt_id=row.prompt_id,
            model_id=requested_model_id,
            backend="iris_vllm",
            response_text=text,
            finish_reason=finish_reason,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            thinking_tokens=None,
            seed=seed,
            sampling_config_hash=sampling.hash_short(),
            raw_provider_metadata={"finish_reason": finish_reason, "served_model_id": served_model_id},
        )


def _empty_rollout(row: PreparedRow, model_id: str, seed: int, sampling: SamplingConfig, finish_reason: str) -> Rollout:
    return Rollout(
        prompt_id=row.prompt_id,
        model_id=model_id,
        backend="iris_vllm",
        response_text="",
        finish_reason=finish_reason,
        input_tokens=None,
        output_tokens=None,
        thinking_tokens=None,
        seed=seed,
        sampling_config_hash=sampling.hash_short(),
        raw_provider_metadata={"finish_reason": finish_reason},
    )


async def run_rollouts(
    rows: list[PreparedRow],
    *,
    server_url: str,
    served_model_id: str,
    requested_model_id: str,
    concurrency: int,
    seed: int,
    sampling: SamplingConfig,
    shard_output_dir: str | None,
    shard_size: int,
    request_timeout: float,
) -> list[Rollout]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=request_timeout) as client:
        tasks = [
            asyncio.create_task(
                _one_completion(
                    client,
                    sem,
                    server_url=server_url,
                    served_model_id=served_model_id,
                    requested_model_id=requested_model_id,
                    row=row,
                    seed=seed,
                    sampling=sampling,
                    request_timeout=request_timeout,
                )
            )
            for row in rows
        ]
        out: list[Rollout] = []
        shard_buffer: list[Rollout] = []
        shard_index = 0
        run_id = str(int(time.time()))
        started = time.time()
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            rollout = await task
            out.append(rollout)
            if shard_output_dir is not None:
                shard_buffer.append(rollout)
                if len(shard_buffer) >= shard_size:
                    shard_path = write_rollout_shard(shard_output_dir, run_id, shard_index, shard_buffer)
                    logger.info("Wrote rollout shard %s (%d rows)", shard_path, len(shard_buffer))
                    shard_buffer = []
                    shard_index += 1
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - started
                logger.info("vLLM progress %d/%d (%.1f req/s)", i, len(tasks), i / max(elapsed, 1.0))
        if shard_output_dir is not None and shard_buffer:
            shard_path = write_rollout_shard(shard_output_dir, run_id, shard_index, shard_buffer)
            logger.info("Wrote rollout shard %s (%d rows)", shard_path, len(shard_buffer))
        return out


def write_rollouts(path: str, rollouts: list[Rollout]) -> None:
    with fsspec.open(path, "wt") as f:
        for rollout in rollouts:
            f.write(json.dumps(dataclasses.asdict(rollout), ensure_ascii=False) + "\n")


def configure_model_cache(cache_dir: str) -> None:
    cache_path = pathlib.Path(cache_dir)
    hub_path = cache_path / "hub"
    for path in (cache_path, hub_path, cache_path / "transformers", cache_path / "xdg", cache_path / "vllm-assets"):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    os.environ["HF_HUB_CACHE"] = str(hub_path)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(cache_path / "xdg")
    os.environ["VLLM_ASSETS_CACHE"] = str(cache_path / "vllm-assets")
    result = subprocess.run(
        ["df", "-h", "/", "/app", "/tmp", str(cache_path)], check=False, capture_output=True, text=True
    )
    logger.info("Configured model cache under %s\n%s%s", cache_path, result.stdout, result.stderr)


def start_startup_heartbeat(model_id: str) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.wait(STARTUP_HEARTBEAT_SECONDS):
            logger.info("Still waiting for vLLM startup for %s", model_id)

    thread = threading.Thread(target=_heartbeat, name="vllm-startup-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def stop_startup_heartbeat(stop: threading.Event, thread: threading.Thread) -> None:
    stop.set()
    thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prepared", default=DEFAULT_PREPARED)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mode", choices=["native", "docker"], default="native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--prompt-ids-file")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_SAMPLING.max_new_tokens)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--shard-output-dir")
    parser.add_argument("--resume-from-shards", action="store_true")
    parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.resume_from_shards and not args.shard_output_dir:
        raise ValueError("--resume-from-shards requires --shard-output-dir")
    configure_model_cache(args.cache_dir)
    prompt_ids = load_prompt_ids(args.prompt_ids_file) if args.prompt_ids_file else None
    rows = load_prompts(args.prepared, args.n, prompt_ids)
    if args.resume_from_shards and args.shard_output_dir:
        completed = completed_prompt_ids_from_shards(args.shard_output_dir)
        before = len(rows)
        rows = [row for row in rows if row.prompt_id not in completed]
        logger.info("Skipping %d prompts already present in %s", before - len(rows), args.shard_output_dir)
    sampling = dataclasses.replace(DEFAULT_SAMPLING, max_new_tokens=args.max_new_tokens)
    logger.info(
        "Loaded %d prompts from %s%s; max_new_tokens=%d",
        len(rows),
        args.prepared,
        f" filtered by {args.prompt_ids_file}" if args.prompt_ids_file else "",
        sampling.max_new_tokens,
    )

    model = ModelConfig(
        name=args.model_id,
        path=None,
        engine_kwargs={
            "max_model_len": args.max_model_len,
        },
    )
    extra_args = [
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.max_num_batched_tokens is not None:
        extra_args.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    with remove_tpu_lockfile_on_exit():
        heartbeat_stop, heartbeat_thread = start_startup_heartbeat(args.model_id)
        try:
            with VllmEnvironment(model=model, mode=args.mode, timeout_seconds=3600, extra_args=extra_args) as env:
                stop_startup_heartbeat(heartbeat_stop, heartbeat_thread)
                if env.model_id is None:
                    raise RuntimeError("vLLM server did not expose a model id")
                rollouts = asyncio.run(
                    run_rollouts(
                        rows,
                        server_url=env.server_url,
                        served_model_id=env.model_id,
                        requested_model_id=args.model_id,
                        concurrency=args.concurrency,
                        seed=args.seed,
                        sampling=sampling,
                        shard_output_dir=args.shard_output_dir,
                        shard_size=args.shard_size,
                        request_timeout=args.request_timeout,
                    )
                )
        finally:
            stop_startup_heartbeat(heartbeat_stop, heartbeat_thread)
    if args.shard_output_dir:
        n_consolidated = write_consolidated_shards(args.shard_output_dir, args.output)
        logger.info("Wrote %d consolidated sharded rollouts to %s", n_consolidated, args.output)
    else:
        write_rollouts(args.output, rollouts)
        logger.info("Wrote %d rollouts to %s", len(rollouts), args.output)


if __name__ == "__main__":
    main()
