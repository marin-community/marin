# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""INF-004a: packed TP=2x2 pure-inference benchmark on a single v5p-8.

Runs two independent TP=2 vLLM engines on one v5p-8 host by partitioning
chips with `TPU_VISIBLE_CHIPS`. This benchmarks the candidate packed topology
against the TP=4 single-engine baselines from INF-001 / INF-003.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fray.v2 import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
    current_client,
)
from marin.rl.environments.inference_ctx import InferenceRequestKind
from marin.training.training import _add_run_env_variables

logger = logging.getLogger(__name__)

CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

DEFAULT_N_PROMPTS = 64
DEFAULT_N_GENERATIONS = 16
DEFAULT_NUM_BATCHES = 25
DEFAULT_WARMUP_BATCHES = 3
DEFAULT_SUFFIX = "inf004-tp2x2"
DEFAULT_REGION = "us-central1"
DEFAULT_CHIP_GROUPS = ("0,1", "2,3")
PROD_TPU_WORKER_RAM = "400g"


@dataclass
class PackedInferenceBenchmarkConfig:
    """Configuration for the packed TP=2x2 benchmark."""

    model_path: str
    canonical_model_name: str
    n_prompts: int
    n_generations: int
    num_batches: int
    warmup_batches: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    chip_groups: list[str]
    wandb_project: str
    wandb_name: str
    wandb_tags: list[str] = field(default_factory=list)


def _parse_chip_groups(raw_value: str) -> list[str]:
    groups = [group.strip() for group in raw_value.split(";") if group.strip()]
    if not groups:
        raise ValueError("Expected at least one chip group")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("parent", "worker"),
        default="parent",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_SUFFIX,
        help="Short suffix used in run IDs and W&B names.",
    )
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS, help="Total prompts per batch.")
    parser.add_argument("--n-generations", type=int, default=DEFAULT_N_GENERATIONS, help="Completions per prompt.")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES, help="Total batches to run.")
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES, help="Batches to discard.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor-parallel degree for each packed replica.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory budget for each packed replica.",
    )
    parser.add_argument(
        "--chip-groups",
        default=";".join(DEFAULT_CHIP_GROUPS),
        help='Semicolon-separated chip groups, e.g. "0,1;2,3".',
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="TPU region.")

    # Internal worker-only arguments.
    parser.add_argument("--model-path", default=MODEL_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--canonical-model-name", default=CANONICAL_MODEL_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--visible-chips", default="", help=argparse.SUPPRESS)
    parser.add_argument("--num-replicas", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--results-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--wandb-name", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    import numpy as np

    return float(np.percentile(sorted_values, percentile))


def _median(values: list[float]) -> float:
    import numpy as np

    return float(np.median(values))


def _mean(values: list[float]) -> float:
    import numpy as np

    return float(np.mean(values))


def _run_worker(args: argparse.Namespace) -> None:
    if args.worker_index < 0:
        raise ValueError("--worker-index is required in worker mode")
    if not args.visible_chips:
        raise ValueError("--visible-chips is required in worker mode")
    if args.num_replicas <= 0:
        raise ValueError("--num-replicas is required in worker mode")
    if not args.results_path:
        raise ValueError("--results-path is required in worker mode")
    if args.n_prompts % args.num_replicas != 0:
        raise ValueError(f"n_prompts={args.n_prompts} must divide evenly across {args.num_replicas} replicas")

    visible_chip_list = [chip.strip() for chip in args.visible_chips.split(",") if chip.strip()]
    if len(visible_chip_list) != args.tensor_parallel_size:
        raise ValueError(f"visible chips {args.visible_chips!r} do not match TP={args.tensor_parallel_size}")

    # These must be set before importing JAX / vLLM.
    os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = f"1,{len(visible_chip_list)},1"
    os.environ["TPU_VISIBLE_CHIPS"] = args.visible_chips
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import time

    import jax
    import numpy as np
    from iris.logging import configure_logging
    from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
    from marin.rl.environments.inference_ctx.vllm import vLLMInferenceContext
    from marin.rl.environments.math_env import MathEnv
    from marin.utils import remove_tpu_lockfile_on_exit

    configure_logging(level=logging.INFO)
    logger.info(
        "INF-004 worker %d starting (chips=%s, TP=%d, gpu_mem=%.2f)",
        args.worker_index,
        args.visible_chips,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
    )

    def get_tpu_hbm_usage() -> dict[str, float]:
        stats: dict[str, float] = {}
        in_use_values: list[float] = []
        for i, device in enumerate(jax.local_devices()):
            mem = device.memory_stats()
            if not mem:
                continue
            bytes_in_use = mem.get("bytes_in_use", 0)
            peak = mem.get("peak_bytes_in_use", 0)
            stats[f"hbm/chip_{i}_bytes_in_use"] = float(bytes_in_use)
            stats[f"hbm/chip_{i}_peak_bytes_in_use"] = float(peak)
            in_use_values.append(float(bytes_in_use))
        if in_use_values:
            stats["hbm/total_bytes_in_use"] = float(sum(in_use_values))
            stats["hbm/max_chip_bytes_in_use"] = float(max(in_use_values))
            stats["hbm/max_chip_gib"] = float(max(in_use_values) / (1024**3))
        return stats

    def build_prompts(env: MathEnv, total_n_prompts: int, rng: np.random.Generator) -> list[list[dict[str, str]]]:
        examples = env.train_examples
        indices = rng.choice(len(examples), size=min(total_n_prompts, len(examples)), replace=False)
        selected = [examples[int(index)] for index in indices]
        return [[*env.fewshot_prefix, {"role": "user", "content": example.processed_prompt}] for example in selected]

    def count_tokens(completions: list[Any]) -> dict[str, float]:
        total_prompt = 0
        total_output = 0
        total_completions = 0
        response_lengths: list[int] = []
        for completion in completions:
            if completion.usage:
                total_prompt += completion.usage.prompt_tokens
                total_output += completion.usage.completion_tokens
            total_completions += len(completion.choices)
            for choice in completion.choices:
                if hasattr(choice, "response_token_ids"):
                    response_lengths.append(len(choice.response_token_ids))

        result: dict[str, float] = {
            "total_prompt_tokens": float(total_prompt),
            "total_output_tokens": float(total_output),
            "total_completions": float(total_completions),
        }
        if response_lengths:
            result["mean_response_tokens"] = float(np.mean(response_lengths))
            result["median_response_tokens"] = float(np.median(response_lengths))
        return result

    def extract_kv_cache_info(inference_ctx: vLLMInferenceContext) -> dict[str, float]:
        kv_cache_info: dict[str, float] = {}
        try:
            engine = inference_ctx.llm
            if hasattr(engine, "llm_engine"):
                llm_engine = engine.llm_engine
                vllm_cfg = getattr(llm_engine, "vllm_config", None)
                if vllm_cfg and hasattr(vllm_cfg, "cache_config"):
                    cache_config = vllm_cfg.cache_config
                    block_size = getattr(cache_config, "block_size", 128)
                    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
                    if num_gpu_blocks is not None:
                        kv_cache_info["kv_cache/num_gpu_blocks"] = float(num_gpu_blocks)
                        kv_cache_info["kv_cache/block_size"] = float(block_size)
                        kv_cache_info["kv_cache/total_tokens"] = float(num_gpu_blocks * block_size)
                        kv_cache_info["kv_cache/max_concurrency_2048"] = float((num_gpu_blocks * block_size) / 2048)
                    gpu_mem = getattr(cache_config, "gpu_memory_utilization", None)
                    if gpu_mem is not None:
                        kv_cache_info["kv_cache/gpu_memory_utilization"] = float(gpu_mem)
        except Exception:
            logger.warning("Failed to extract KV cache info", exc_info=True)
        return kv_cache_info

    worker_prompt_count = args.n_prompts // args.num_replicas
    prompt_start = args.worker_index * worker_prompt_count
    prompt_end = prompt_start + worker_prompt_count
    result_payload: dict[str, Any] = {
        "status": "error",
        "worker_index": args.worker_index,
        "visible_chips": args.visible_chips,
        "worker_prompt_count": worker_prompt_count,
        "batches": [],
    }

    try:
        with remove_tpu_lockfile_on_exit():
            inference_config = vLLMInferenceContextConfig(
                model_name=args.model_path,
                canonical_model_name=args.canonical_model_name,
                max_model_len=2048,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                sampling_params=VLLMSamplingConfig(
                    temperature=1.0,
                    n=args.n_generations,
                    max_tokens=1024,
                    stop=["<|eot_id|>"],
                    include_stop_str_in_output=True,
                    logprobs=1,
                    top_k=4096,
                ),
                load_format="runai_streamer",
                enforce_eager=True,
                kv_cache_metrics=True,
            )
            inference_ctx = vLLMInferenceContext(inference_config)
            env = MathEnv(seed=42)

            kv_cache_info = extract_kv_cache_info(inference_ctx)
            hbm_init = get_tpu_hbm_usage()
            result_payload["init"] = {
                "local_device_count": int(jax.local_device_count()),
                "global_device_count": int(jax.device_count()),
                "hbm": hbm_init,
                "kv_cache": kv_cache_info,
            }

            logger.info(
                "worker %d ready: local_devices=%d, prompt_slice=%d:%d, HBM=%.1f GiB/chip",
                args.worker_index,
                jax.local_device_count(),
                prompt_start,
                prompt_end,
                hbm_init.get("hbm/max_chip_gib", 0.0),
            )
            if kv_cache_info:
                logger.info("worker %d KV cache info: %s", args.worker_index, kv_cache_info)

            measured_batch_times: list[float] = []
            measured_output_tokens: list[int] = []
            measured_request_counts: list[int] = []

            for batch_idx in range(args.num_batches):
                is_warmup = batch_idx < args.warmup_batches
                phase = "warmup" if is_warmup else "measure"

                rng = np.random.default_rng(batch_idx + 1000)
                all_prompts = build_prompts(env, args.n_prompts, rng)
                prompts = all_prompts[prompt_start:prompt_end]

                batch_start = time.time()
                completions = inference_ctx.batch_completions(
                    prompts=prompts,
                    request_kind=InferenceRequestKind.TRAIN,
                    temperature=1.0,
                    n=args.n_generations,
                    max_tokens=1024,
                    top_k=4096,
                    stop=["<|eot_id|>"],
                )
                batch_time = time.time() - batch_start

                tokens = count_tokens(completions)
                output_tokens = int(tokens["total_output_tokens"])
                n_completions = int(tokens["total_completions"])
                hbm = get_tpu_hbm_usage()

                batch_record: dict[str, Any] = {
                    "worker_index": args.worker_index,
                    "batch_idx": batch_idx,
                    "phase": phase,
                    "batch_time_seconds": float(batch_time),
                    "output_tokens_per_sec": float(output_tokens / batch_time if batch_time > 0 else 0.0),
                    "requests_per_sec": float(n_completions / batch_time if batch_time > 0 else 0.0),
                    "total_prompt_tokens": int(tokens["total_prompt_tokens"]),
                    "total_output_tokens": output_tokens,
                    "total_completions": n_completions,
                    "n_prompts": len(prompts),
                    "hbm": hbm,
                }
                if "mean_response_tokens" in tokens:
                    batch_record["mean_response_tokens"] = float(tokens["mean_response_tokens"])
                if "median_response_tokens" in tokens:
                    batch_record["median_response_tokens"] = float(tokens["median_response_tokens"])

                result_payload["batches"].append(batch_record)
                logger.info(
                    "worker %d batch %d/%d (%s): %.1fs, %d output tokens, %.0f tok/s",
                    args.worker_index,
                    batch_idx + 1,
                    args.num_batches,
                    phase,
                    batch_time,
                    output_tokens,
                    output_tokens / batch_time if batch_time > 0 else 0.0,
                )

                if not is_warmup:
                    measured_batch_times.append(batch_time)
                    measured_output_tokens.append(output_tokens)
                    measured_request_counts.append(n_completions)

            measured_tps = [
                tokens / batch_time
                for tokens, batch_time in zip(measured_output_tokens, measured_batch_times, strict=True)
            ]
            measured_rps = [
                count / batch_time
                for count, batch_time in zip(measured_request_counts, measured_batch_times, strict=True)
            ]

            result_payload["summary"] = {
                "measured_elapsed_seconds": float(sum(measured_batch_times)),
                "batch_time_median": _median(measured_batch_times),
                "batch_time_mean": _mean(measured_batch_times),
                "batch_time_p10": _percentile(measured_batch_times, 10),
                "batch_time_p90": _percentile(measured_batch_times, 90),
                "output_tps_median": _median(measured_tps),
                "output_tps_mean": _mean(measured_tps),
                "output_tps_p10": _percentile(measured_tps, 10),
                "output_tps_p90": _percentile(measured_tps, 90),
                "rps_median": _median(measured_rps),
                "rps_mean": _mean(measured_rps),
                "rps_p10": _percentile(measured_rps, 10),
                "rps_p90": _percentile(measured_rps, 90),
                "total_output_tokens": int(sum(measured_output_tokens)),
                "total_completions": int(sum(measured_request_counts)),
                "num_measured_batches": len(measured_batch_times),
            }
            result_payload["status"] = "ok"

            logger.info(
                "worker %d summary: %.1f tok/s median (%.1f-%.1f p10-p90), %.1fs/batch median",
                args.worker_index,
                result_payload["summary"]["output_tps_median"],
                result_payload["summary"]["output_tps_p10"],
                result_payload["summary"]["output_tps_p90"],
                result_payload["summary"]["batch_time_median"],
            )
            inference_ctx.shutdown()
    except Exception:
        result_payload["error"] = traceback.format_exc()
        raise
    finally:
        with open(args.results_path, "w") as result_file:
            json.dump(result_payload, result_file, indent=2, sort_keys=True)


def _load_worker_results(result_paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result_path in result_paths:
        with result_path.open() as result_file:
            results.append(json.load(result_file))
    return results


def _aggregate_results(results: list[dict[str, Any]], config: PackedInferenceBenchmarkConfig) -> dict[str, Any]:
    for result in results:
        if result.get("status") != "ok":
            raise RuntimeError(f"Worker {result.get('worker_index')} failed:\n{result.get('error', 'unknown error')}")

    num_batches = config.num_batches
    aggregate_batches: list[dict[str, Any]] = []

    for batch_idx in range(num_batches):
        worker_batches = [result["batches"][batch_idx] for result in results]
        phases = {batch["phase"] for batch in worker_batches}
        if len(phases) != 1:
            raise ValueError(f"Mismatched phases for batch {batch_idx}: {phases}")

        phase = worker_batches[0]["phase"]
        aggregate_batch_time = max(float(batch["batch_time_seconds"]) for batch in worker_batches)
        aggregate_output_tokens = int(sum(int(batch["total_output_tokens"]) for batch in worker_batches))
        aggregate_completions = int(sum(int(batch["total_completions"]) for batch in worker_batches))

        aggregate_batches.append(
            {
                "batch_idx": batch_idx,
                "phase": phase,
                "aggregate_batch_time_seconds_index_aligned": aggregate_batch_time,
                "aggregate_total_output_tokens": aggregate_output_tokens,
                "aggregate_total_completions": aggregate_completions,
                "aggregate_output_tokens_per_sec_index_aligned": float(
                    aggregate_output_tokens / aggregate_batch_time if aggregate_batch_time > 0 else 0.0
                ),
                "aggregate_requests_per_sec_index_aligned": float(
                    aggregate_completions / aggregate_batch_time if aggregate_batch_time > 0 else 0.0
                ),
                "worker_batches": worker_batches,
            }
        )

    measured_batches = [batch for batch in aggregate_batches if batch["phase"] == "measure"]
    measured_batch_times = [float(batch["aggregate_batch_time_seconds_index_aligned"]) for batch in measured_batches]
    measured_output_tps = [float(batch["aggregate_output_tokens_per_sec_index_aligned"]) for batch in measured_batches]
    measured_request_rates = [float(batch["aggregate_requests_per_sec_index_aligned"]) for batch in measured_batches]

    total_output_tokens = int(sum(int(result["summary"]["total_output_tokens"]) for result in results))
    total_completions = int(sum(int(result["summary"]["total_completions"]) for result in results))
    makespan_seconds = max(float(result["summary"]["measured_elapsed_seconds"]) for result in results)

    return {
        "batches": aggregate_batches,
        "summary": {
            "aggregate_output_tps_makespan": float(
                total_output_tokens / makespan_seconds if makespan_seconds > 0 else 0.0
            ),
            "aggregate_requests_per_sec_makespan": float(
                total_completions / makespan_seconds if makespan_seconds > 0 else 0.0
            ),
            "aggregate_measured_makespan_seconds": makespan_seconds,
            "aggregate_total_output_tokens": total_output_tokens,
            "aggregate_total_completions": total_completions,
            "aggregate_batch_time_median_index_aligned": _median(measured_batch_times),
            "aggregate_batch_time_mean_index_aligned": _mean(measured_batch_times),
            "aggregate_batch_time_p10_index_aligned": _percentile(measured_batch_times, 10),
            "aggregate_batch_time_p90_index_aligned": _percentile(measured_batch_times, 90),
            "aggregate_output_tps_median_index_aligned": _median(measured_output_tps),
            "aggregate_output_tps_mean_index_aligned": _mean(measured_output_tps),
            "aggregate_output_tps_p10_index_aligned": _percentile(measured_output_tps, 10),
            "aggregate_output_tps_p90_index_aligned": _percentile(measured_output_tps, 90),
            "aggregate_rps_median_index_aligned": _median(measured_request_rates),
            "aggregate_rps_mean_index_aligned": _mean(measured_request_rates),
            "aggregate_rps_p10_index_aligned": _percentile(measured_request_rates, 10),
            "aggregate_rps_p90_index_aligned": _percentile(measured_request_rates, 90),
            "num_measured_batches": len(measured_batches),
        },
    }


def run_packed_benchmark(config: PackedInferenceBenchmarkConfig) -> None:
    import wandb
    from iris.logging import configure_logging

    configure_logging(level=logging.INFO)
    logger.info(
        "INF-004a packed benchmark starting: %s (replicas=%d, TP=%d, gpu_mem=%.2f)",
        config.wandb_name,
        len(config.chip_groups),
        config.tensor_parallel_size,
        config.gpu_memory_utilization,
    )

    if config.n_prompts % len(config.chip_groups) != 0:
        raise ValueError(f"n_prompts={config.n_prompts} must divide evenly across {len(config.chip_groups)} replicas")

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        tags=config.wandb_tags,
        id=config.wandb_name,
        resume="allow",
    )

    config_metrics: dict[str, float | str] = {
        "config/n_prompts_total": config.n_prompts,
        "config/n_prompts_per_replica": config.n_prompts // len(config.chip_groups),
        "config/n_generations": config.n_generations,
        "config/completions_per_batch_total": config.n_prompts * config.n_generations,
        "config/completions_per_batch_per_replica": (config.n_prompts // len(config.chip_groups)) * config.n_generations,
        "config/tensor_parallel_size_per_replica": config.tensor_parallel_size,
        "config/gpu_memory_utilization": config.gpu_memory_utilization,
        "config/num_replicas": len(config.chip_groups),
        "config/num_batches": config.num_batches,
        "config/warmup_batches": config.warmup_batches,
    }
    for replica_index, chip_group in enumerate(config.chip_groups):
        config_metrics[f"config/replica_{replica_index}_chips"] = chip_group
    run.log(config_metrics)

    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="inf-004a-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        result_paths = [tmp_path / f"worker_{index}.json" for index in range(len(config.chip_groups))]
        processes: list[subprocess.Popen[bytes]] = []

        for worker_index, chip_group in enumerate(config.chip_groups):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            cmd = [
                sys.executable,
                "-u",
                str(script_path),
                "--mode",
                "worker",
                "--model-path",
                config.model_path,
                "--canonical-model-name",
                config.canonical_model_name,
                "--n-prompts",
                str(config.n_prompts),
                "--n-generations",
                str(config.n_generations),
                "--num-batches",
                str(config.num_batches),
                "--warmup-batches",
                str(config.warmup_batches),
                "--tensor-parallel-size",
                str(config.tensor_parallel_size),
                "--gpu-memory-utilization",
                str(config.gpu_memory_utilization),
                "--worker-index",
                str(worker_index),
                "--visible-chips",
                chip_group,
                "--num-replicas",
                str(len(config.chip_groups)),
                "--results-path",
                str(result_paths[worker_index]),
            ]

            logger.info("Launching worker %d on chips %s", worker_index, chip_group)
            processes.append(subprocess.Popen(cmd, env=env))

        exit_codes: dict[int, int] = {}
        running = list(enumerate(processes))
        while running:
            progress = False
            for process_index, process in list(running):
                exit_code = process.poll()
                if exit_code is None:
                    continue

                exit_codes[process_index] = exit_code
                running.remove((process_index, process))
                progress = True

                if exit_code != 0:
                    for _, other_process in running:
                        other_process.terminate()
                    for other_index, other_process in running:
                        exit_codes[other_index] = other_process.wait()
                    raise RuntimeError(f"At least one INF-004 worker failed: exit_codes={exit_codes}")

            if not progress:
                time.sleep(5)

        worker_results = _load_worker_results(result_paths)
        aggregate = _aggregate_results(worker_results, config)

        for worker_result in worker_results:
            worker_index = worker_result["worker_index"]
            init = worker_result["init"]
            run.log(
                {
                    f"worker_{worker_index}/init/local_device_count": init["local_device_count"],
                    f"worker_{worker_index}/init/global_device_count": init["global_device_count"],
                    **{f"worker_{worker_index}/init/{key}": value for key, value in init["hbm"].items()},
                    **{f"worker_{worker_index}/init/{key}": value for key, value in init["kv_cache"].items()},
                }
            )

        for batch in aggregate["batches"]:
            phase = batch["phase"]
            batch_metrics: dict[str, float] = {
                f"bench.{phase}/batch_idx": batch["batch_idx"],
                f"bench.{phase}/aggregate_batch_time_seconds_index_aligned": batch[
                    "aggregate_batch_time_seconds_index_aligned"
                ],
                f"bench.{phase}/aggregate_output_tokens_per_sec_index_aligned": batch[
                    "aggregate_output_tokens_per_sec_index_aligned"
                ],
                f"bench.{phase}/aggregate_requests_per_sec_index_aligned": batch[
                    "aggregate_requests_per_sec_index_aligned"
                ],
                f"bench.{phase}/aggregate_total_output_tokens": batch["aggregate_total_output_tokens"],
                f"bench.{phase}/aggregate_total_completions": batch["aggregate_total_completions"],
            }

            for worker_batch in batch["worker_batches"]:
                worker_index = worker_batch["worker_index"]
                batch_metrics[f"bench.{phase}/worker_{worker_index}/batch_time_seconds"] = worker_batch[
                    "batch_time_seconds"
                ]
                batch_metrics[f"bench.{phase}/worker_{worker_index}/output_tokens_per_sec"] = worker_batch[
                    "output_tokens_per_sec"
                ]
                batch_metrics[f"bench.{phase}/worker_{worker_index}/requests_per_sec"] = worker_batch["requests_per_sec"]
                batch_metrics[f"bench.{phase}/worker_{worker_index}/total_output_tokens"] = worker_batch[
                    "total_output_tokens"
                ]
                if "hbm" in worker_batch and "hbm/max_chip_gib" in worker_batch["hbm"]:
                    batch_metrics[f"bench.{phase}/worker_{worker_index}/hbm_max_chip_gib"] = worker_batch["hbm"][
                        "hbm/max_chip_gib"
                    ]

            run.log(batch_metrics, step=batch["batch_idx"])

        summary_metrics = {f"summary/{key}": value for key, value in aggregate["summary"].items()}
        for worker_result in worker_results:
            worker_index = worker_result["worker_index"]
            summary_metrics.update(
                {f"summary/worker_{worker_index}_{key}": value for key, value in worker_result["summary"].items()}
            )
        run.log(summary_metrics)

        logger.info(
            "Summary: packed makespan=%.1f tok/s, index-aligned median=%.1f tok/s (%.1f-%.1f p10-p90)",
            aggregate["summary"]["aggregate_output_tps_makespan"],
            aggregate["summary"]["aggregate_output_tps_median_index_aligned"],
            aggregate["summary"]["aggregate_output_tps_p10_index_aligned"],
            aggregate["summary"]["aggregate_output_tps_p90_index_aligned"],
        )

    run.finish()
    logger.info("INF-004a benchmark complete")


def submit_benchmark(config: PackedInferenceBenchmarkConfig, region: str) -> None:
    client = current_client()

    env = {"EQX_ON_ERROR": "nan"}
    env = _add_run_env_variables(env)

    job = client.submit(
        JobRequest(
            name=f"inf-004-{config.wandb_name}",
            entrypoint=Entrypoint.from_callable(run_packed_benchmark, args=(config,)),
            resources=ResourceConfig.with_tpu("v5p-8", regions=[region], ram=PROD_TPU_WORKER_RAM),
            environment=create_environment(env_vars=env, extras=["tpu", "vllm", "math"]),
            max_retries_failure=0,
            max_retries_preemption=3,
        )
    )
    logger.info("Submitted INF-004a job: %s", job.job_id)
    job.wait(raise_on_failure=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == "worker":
        _run_worker(args)
        return

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.experiment_name_suffix}-{datestamp}"
    chip_groups = _parse_chip_groups(args.chip_groups)

    config = PackedInferenceBenchmarkConfig(
        model_path=MODEL_PATH,
        canonical_model_name=CANONICAL_MODEL_NAME,
        n_prompts=args.n_prompts,
        n_generations=args.n_generations,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        chip_groups=chip_groups,
        wandb_project="marin_iris_rl_debug",
        wandb_name=name,
        wandb_tags=["inference-benchmark", "inf-004", "tp2x2-packed"],
    )

    logger.info(
        "INF-004a packed TP=2x2: %s (n_prompts=%d, n_generations=%d, batches=%d, warmup=%d, "
        "TP=%d per replica, gpu_mem=%.2f, chip_groups=%s, region=%s)",
        name,
        args.n_prompts,
        args.n_generations,
        args.num_batches,
        args.warmup_batches,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        chip_groups,
        args.region,
    )
    submit_benchmark(config, region=args.region)


if __name__ == "__main__":
    main()
