# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""INF-001: TP=4 pure-inference benchmark.

Establishes the single-engine vLLM inference baseline on v5p-8 (TP=4) using
the exact same model, prompts, and sampling config as production RL rollouts.
No training, weight transfer, or replay buffer — just batched inference with
timing and throughput metrics logged to W&B.

Part of the pure-inference experiment ladder documented in
.agents/logbooks/iris-rl-claude.md.
"""

import argparse
import datetime
import logging
from dataclasses import dataclass, field

from fray.v2 import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
    current_client,
)
from marin.training.training import _add_run_env_variables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — match production RL config
# ---------------------------------------------------------------------------

CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

DEFAULT_N_PROMPTS = 64
DEFAULT_N_GENERATIONS = 16
DEFAULT_NUM_BATCHES = 25
DEFAULT_WARMUP_BATCHES = 3
DEFAULT_SUFFIX = "inf001-tp4"
DEFAULT_REGION = "us-central1"
PROD_TPU_WORKER_RAM = "400g"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class InferenceBenchmarkConfig:
    """Pure-inference benchmark configuration."""

    model_path: str
    canonical_model_name: str
    n_prompts: int
    n_generations: int
    num_batches: int
    warmup_batches: int
    wandb_project: str
    wandb_name: str
    gpu_memory_utilization: float = 0.90
    wandb_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_SUFFIX,
        help="Short suffix used in run IDs and W&B names.",
    )
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS, help="Prompts per batch.")
    parser.add_argument("--n-generations", type=int, default=DEFAULT_N_GENERATIONS, help="Completions per prompt.")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES, help="Total batches to run.")
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES, help="Batches to discard.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory budget.")
    parser.add_argument("--region", default=DEFAULT_REGION, help="TPU region.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark runner (executes on TPU — all heavy imports deferred here)
# ---------------------------------------------------------------------------


def run_benchmark(config: InferenceBenchmarkConfig) -> None:
    """Run the pure-inference benchmark on a TPU node."""
    # Defer heavy imports so the CPU root node never needs vllm/jax/sympy.
    import time

    import jax
    import numpy as np
    import wandb

    from iris.logging import configure_logging
    from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
    from marin.rl.environments.inference_ctx.vllm import vLLMInferenceContext
    from marin.rl.environments.math_env import MathEnv
    from marin.utils import remove_tpu_lockfile_on_exit

    def get_tpu_hbm_usage() -> dict[str, float]:
        """Collect per-chip and aggregate HBM usage from local TPU devices."""
        stats: dict[str, float] = {}
        in_use_values: list[float] = []
        for i, dev in enumerate(jax.local_devices()):
            mem = dev.memory_stats()
            if not mem:
                continue
            bytes_in_use = mem.get("bytes_in_use", 0)
            peak = mem.get("peak_bytes_in_use", 0)
            stats[f"hbm/chip_{i}_bytes_in_use"] = bytes_in_use
            stats[f"hbm/chip_{i}_peak_bytes_in_use"] = peak
            in_use_values.append(bytes_in_use)
        if in_use_values:
            stats["hbm/total_bytes_in_use"] = sum(in_use_values)
            stats["hbm/max_chip_bytes_in_use"] = max(in_use_values)
            stats["hbm/max_chip_gib"] = max(in_use_values) / (1024**3)
        return stats

    def build_prompts(env: MathEnv, n_prompts: int, rng: np.random.Generator) -> list[list[dict]]:
        """Sample prompts and format with few-shot prefix, matching MathEnv.sample()."""
        examples = env.train_examples
        indices = rng.choice(len(examples), size=min(n_prompts, len(examples)), replace=False)
        selected = [examples[int(i)] for i in indices]
        return [[*env.fewshot_prefix, {"role": "user", "content": ex.processed_prompt}] for ex in selected]

    def count_tokens(completions: list) -> dict[str, float]:
        """Extract token counts from a batch of ChatCompletion objects."""
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
            "total_prompt_tokens": total_prompt,
            "total_output_tokens": total_output,
            "total_completions": total_completions,
        }
        if response_lengths:
            result["mean_response_tokens"] = float(np.mean(response_lengths))
            result["median_response_tokens"] = float(np.median(response_lengths))
        return result

    configure_logging(level=logging.INFO)
    logger.info("INF-001 benchmark starting: %s", config.wandb_name)

    with remove_tpu_lockfile_on_exit():
        # W&B tracker (avoids JAX init deadlock, same as rollout workers)
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            tags=config.wandb_tags,
            id=config.wandb_name,
            resume="allow",
        )

        # vLLM engine — exact production config with kv_cache_metrics enabled
        inference_config = vLLMInferenceContextConfig(
            model_name=config.model_path,
            canonical_model_name=config.canonical_model_name,
            max_model_len=2048,
            tensor_parallel_size=4,
            gpu_memory_utilization=config.gpu_memory_utilization,
            sampling_params=VLLMSamplingConfig(
                temperature=1.0,
                n=config.n_generations,
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
        logger.info("vLLM engine initialized (TP=4, max_model_len=2048)")

        # Extract KV cache init details from vLLM engine before they're lost
        kv_cache_info: dict[str, float] = {}
        try:
            engine = inference_ctx.llm
            # Try V1 engine path (vLLM >= 0.13)
            if hasattr(engine, "llm_engine"):
                eng = engine.llm_engine
                vllm_cfg = getattr(eng, "vllm_config", None)
                if vllm_cfg and hasattr(vllm_cfg, "cache_config"):
                    cc = vllm_cfg.cache_config
                    block_size = getattr(cc, "block_size", 128)
                    num_gpu_blocks = getattr(cc, "num_gpu_blocks", None)
                    if num_gpu_blocks is not None:
                        kv_cache_info["kv_cache/num_gpu_blocks"] = num_gpu_blocks
                        kv_cache_info["kv_cache/block_size"] = block_size
                        kv_cache_info["kv_cache/total_tokens"] = num_gpu_blocks * block_size
                        kv_cache_info["kv_cache/max_concurrency_2048"] = (num_gpu_blocks * block_size) / 2048
                    gpu_mem = getattr(cc, "gpu_memory_utilization", None)
                    if gpu_mem is not None:
                        kv_cache_info["kv_cache/gpu_memory_utilization"] = gpu_mem
            if kv_cache_info:
                logger.info("KV cache info: %s", kv_cache_info)
            else:
                logger.warning("Could not extract KV cache info from engine")
        except Exception:
            logger.warning("Failed to extract KV cache info", exc_info=True)

        # Load MATH training data (same distribution as RL rollouts)
        env = MathEnv(seed=42)
        logger.info("MathEnv loaded: %d train examples", len(env.train_examples))

        # HBM after model load
        hbm_init = get_tpu_hbm_usage()
        run.log({"init/" + k: v for k, v in hbm_init.items()})
        logger.info("HBM after model load: %.1f GiB (max chip)", hbm_init.get("hbm/max_chip_gib", 0))

        # Log config + KV cache init details
        config_metrics: dict[str, float] = {
            "config/n_prompts": config.n_prompts,
            "config/n_generations": config.n_generations,
            "config/completions_per_batch": config.n_prompts * config.n_generations,
            "config/tensor_parallel_size": 4,
            "config/max_model_len": 2048,
            "config/max_tokens": 1024,
            "config/gpu_memory_utilization": config.gpu_memory_utilization,
            "config/num_batches": config.num_batches,
            "config/warmup_batches": config.warmup_batches,
        }
        config_metrics.update(kv_cache_info)
        run.log(config_metrics)

        # Measurement accumulators (post-warmup only)
        batch_times: list[float] = []
        batch_output_tokens: list[int] = []
        batch_request_counts: list[int] = []

        for batch_idx in range(config.num_batches):
            is_warmup = batch_idx < config.warmup_batches
            phase = "warmup" if is_warmup else "measure"

            # Sample fresh prompts each batch
            rng = np.random.default_rng(batch_idx + 1000)
            prompts = build_prompts(env, config.n_prompts, rng)

            t0 = time.time()
            completions = inference_ctx.batch_completions(
                prompts=prompts,
                temperature=1.0,
                n=config.n_generations,
                max_tokens=1024,
                top_k=4096,
                stop=["<|eot_id|>"],
            )
            batch_time = time.time() - t0

            tokens = count_tokens(completions)
            output_toks = int(tokens["total_output_tokens"])
            n_completions = int(tokens["total_completions"])

            metrics: dict[str, float] = {
                f"bench.{phase}/batch_idx": batch_idx,
                f"bench.{phase}/batch_time_seconds": batch_time,
                f"bench.{phase}/output_tokens_per_sec": output_toks / batch_time if batch_time > 0 else 0,
                f"bench.{phase}/requests_per_sec": n_completions / batch_time if batch_time > 0 else 0,
                f"bench.{phase}/total_prompt_tokens": tokens["total_prompt_tokens"],
                f"bench.{phase}/total_output_tokens": output_toks,
                f"bench.{phase}/total_completions": n_completions,
                f"bench.{phase}/n_prompts": len(prompts),
            }
            if "mean_response_tokens" in tokens:
                metrics[f"bench.{phase}/mean_response_tokens"] = tokens["mean_response_tokens"]
            if "median_response_tokens" in tokens:
                metrics[f"bench.{phase}/median_response_tokens"] = tokens["median_response_tokens"]

            # HBM snapshot
            hbm = get_tpu_hbm_usage()
            metrics.update({f"bench.{phase}/{k}": v for k, v in hbm.items()})

            run.log(metrics, step=batch_idx)
            logger.info(
                "batch %d/%d (%s): %.1fs, %d output tokens, %.0f tok/s",
                batch_idx + 1,
                config.num_batches,
                phase,
                batch_time,
                output_toks,
                output_toks / batch_time if batch_time > 0 else 0,
            )

            if not is_warmup:
                batch_times.append(batch_time)
                batch_output_tokens.append(output_toks)
                batch_request_counts.append(n_completions)

        # Summary statistics
        if batch_times:
            tps = [t / bt for t, bt in zip(batch_output_tokens, batch_times, strict=True)]
            rps = [r / bt for r, bt in zip(batch_request_counts, batch_times, strict=True)]

            summary: dict[str, float] = {
                "summary/batch_time_median": float(np.median(batch_times)),
                "summary/batch_time_mean": float(np.mean(batch_times)),
                "summary/batch_time_p10": float(np.percentile(batch_times, 10)),
                "summary/batch_time_p90": float(np.percentile(batch_times, 90)),
                "summary/output_tps_median": float(np.median(tps)),
                "summary/output_tps_mean": float(np.mean(tps)),
                "summary/output_tps_p10": float(np.percentile(tps, 10)),
                "summary/output_tps_p90": float(np.percentile(tps, 90)),
                "summary/rps_median": float(np.median(rps)),
                "summary/rps_mean": float(np.mean(rps)),
                "summary/rps_p10": float(np.percentile(rps, 10)),
                "summary/rps_p90": float(np.percentile(rps, 90)),
                "summary/num_measured_batches": len(batch_times),
            }
            run.log(summary)
            logger.info(
                "Summary: %.1f tok/s median (%.1f-%.1f p10-p90), %.1fs/batch median",
                summary["summary/output_tps_median"],
                summary["summary/output_tps_p10"],
                summary["summary/output_tps_p90"],
                summary["summary/batch_time_median"],
            )

        run.finish()
        inference_ctx.shutdown()
        logger.info("INF-001 benchmark complete")


# ---------------------------------------------------------------------------
# Job submission (executes on CPU head node)
# ---------------------------------------------------------------------------


def submit_benchmark(config: InferenceBenchmarkConfig, region: str) -> None:
    """Submit the benchmark as a single TPU job via Fray v2."""
    client = current_client()

    env = {"EQX_ON_ERROR": "nan"}
    env = _add_run_env_variables(env)

    job = client.submit(
        JobRequest(
            name=f"inf-001-{config.wandb_name}",
            entrypoint=Entrypoint.from_callable(run_benchmark, args=(config,)),
            resources=ResourceConfig.with_tpu("v5p-8", regions=[region], ram=PROD_TPU_WORKER_RAM),
            environment=create_environment(env_vars=env, extras=["tpu", "vllm", "math"]),
            max_retries_failure=0,
            max_retries_preemption=3,
        )
    )
    logger.info("Submitted INF-001 job: %s", job.job_id)
    job.wait(raise_on_failure=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.experiment_name_suffix}-{datestamp}"

    config = InferenceBenchmarkConfig(
        model_path=MODEL_PATH,
        canonical_model_name=CANONICAL_MODEL_NAME,
        n_prompts=args.n_prompts,
        n_generations=args.n_generations,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        wandb_project="marin_iris_rl_debug",
        wandb_name=name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        wandb_tags=["inference-benchmark", "inf-001", "tp4-baseline"],
    )

    logger.info(
        "INF-001 TP=4 baseline: %s (n_prompts=%d, n_generations=%d, batches=%d, warmup=%d, " "gpu_mem=%.2f, region=%s)",
        name,
        args.n_prompts,
        args.n_generations,
        args.num_batches,
        args.warmup_batches,
        args.gpu_memory_utilization,
        args.region,
    )
    submit_benchmark(config, region=args.region)


if __name__ == "__main__":
    main()
