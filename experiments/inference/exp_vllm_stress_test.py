# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stress test vLLM on TPU with thousands of concurrent prompts.

Usage (via Iris):
    iris job run --tpu v5p-8 --memory 256GB --region us-central1 \
        --extra tpu --extra vllm \
        --job-name vllm-stress-70b \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/inference/exp_vllm_stress_test.py \
        --model gs://marin-us-central2/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b \
        --num-prompts 5000 \
        --max-concurrent 64 \
        --max-tokens 128 \
        --mode native
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from iris.marin_fs import marin_prefix, open_url

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)

# Duplicate raw stdout fd for Iris-visible logging — immune to JAX/vLLM
# redirecting sys.stdout.  See _iris_emit in vllm_inprocess.py.
_IRIS_LOG_FD = os.dup(1)


def _iris_log(message: str) -> None:
    """Write directly to container stdout fd so Iris dashboard captures it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    line = f"I{ts} 0 vllm.stress {message}\n"
    os.write(_IRIS_LOG_FD, line.encode())


# Diverse prompt templates for realistic workload
PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "Write a short paragraph about {topic}.",
    "What are the key differences between {topic_a} and {topic_b}?",
    "List 3 interesting facts about {topic}.",
    "Summarize the importance of {topic} in one paragraph.",
    "How does {topic} work? Give a brief explanation.",
    "What is the relationship between {topic_a} and {topic_b}?",
    "Describe {topic} as if explaining to a 10-year-old.",
    "What are the pros and cons of {topic}?",
    "Give a creative analogy for {topic}.",
]

TOPICS = [
    "quantum computing",
    "machine learning",
    "photosynthesis",
    "blockchain",
    "neural networks",
    "climate change",
    "DNA replication",
    "black holes",
    "supply chain management",
    "natural language processing",
    "game theory",
    "reinforcement learning",
    "protein folding",
    "renewable energy",
    "cryptography",
    "distributed systems",
    "evolutionary biology",
    "compiler design",
    "ocean currents",
    "microprocessor architecture",
    "gravitational waves",
    "epigenetics",
    "topology",
    "fluid dynamics",
    "signal processing",
    "operating systems",
    "thermodynamics",
    "organic chemistry",
    "number theory",
    "robotics",
    "immunology",
    "cloud computing",
    "graph theory",
    "semiconductor physics",
    "information theory",
    "plate tectonics",
    "group theory",
    "computer vision",
    "statistical mechanics",
    "algebra",
]


def generate_prompts(n: int) -> list[str]:
    """Generate n diverse prompts."""
    prompts = []
    for _ in range(n):
        template = random.choice(PROMPT_TEMPLATES)
        if "{topic_a}" in template:
            a, b = random.sample(TOPICS, 2)
            prompt = template.format(topic_a=a, topic_b=b)
        else:
            prompt = template.format(topic=random.choice(TOPICS))
        prompts.append(prompt)
    return prompts


def send_request(server_url: str, model_id: str, prompt: str, max_tokens: int, request_id: int) -> dict:
    """Send a single completion request and return timing info."""
    start = time.time()
    resp = requests.post(
        f"{server_url}/completions",
        json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    tokens_generated = data["usage"]["completion_tokens"]
    response_text = data["choices"][0]["text"]
    logger.info("[%04d] completed in %.3fs", request_id, elapsed)
    logger.info("[%04d] prompt: %s", request_id, prompt)
    logger.info("[%04d] response (%d tokens): %s", request_id, tokens_generated, response_text)
    return {
        "elapsed": elapsed,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
        "response": response_text,
    }


def run_stress_test(
    *,
    model_name_or_path: str,
    num_prompts: int,
    max_concurrent: int,
    max_tokens: int,
    max_model_len: int,
    startup_timeout: int,
    mode: str | None,
    native_startup_failure_mode: str,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    enforce_eager: bool = False,
) -> dict:
    engine_kwargs: dict = {"max_model_len": max_model_len}
    if tensor_parallel_size is not None:
        engine_kwargs["tensor_parallel_size"] = tensor_parallel_size
    if gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if enforce_eager:
        engine_kwargs["enforce_eager"] = True

    is_gcs = model_name_or_path.startswith("gs://")
    if is_gcs:
        model = ModelConfig(
            name="stress-test-model",
            path=model_name_or_path,
            engine_kwargs=engine_kwargs,
        )
    else:
        model = ModelConfig(
            name=model_name_or_path,
            path=None,
            engine_kwargs=engine_kwargs,
        )

    env = VllmEnvironment(
        model=model,
        host="127.0.0.1",
        port=8000,
        timeout_seconds=startup_timeout,
        mode=mode,
        native_startup_failure_mode=native_startup_failure_mode,
    )

    with env:
        prompts = generate_prompts(num_prompts)
        _iris_log(
            f"Stress test: prompts={num_prompts} concurrent={max_concurrent} "
            f"max_tokens={max_tokens} model={model_name_or_path}"
        )

        results = []
        errors = 0
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(send_request, env.server_url, env.model_id, p, max_tokens, i): i
                for i, p in enumerate(prompts)
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Request failed: {e}")

                if completed % 10 == 0 or completed == num_prompts:
                    elapsed = time.time() - total_start
                    rps = completed / elapsed if elapsed > 0 else 0
                    _iris_log(f"[{completed}/{num_prompts}] {elapsed:.1f}s elapsed, {rps:.1f} req/s, {errors} errors")

        total_elapsed = time.time() - total_start

        # Compute stats
        if results:
            latencies = [r["elapsed"] for r in results]
            tps_values = [r["tokens_per_sec"] for r in results]
            total_tokens = sum(r["tokens_generated"] for r in results)

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]

            stats = {
                "total_prompts": num_prompts,
                "successful": len(results),
                "errors": errors,
                "total_time_sec": round(total_elapsed, 1),
                "requests_per_sec": round(len(results) / total_elapsed, 2),
                "total_tokens_generated": total_tokens,
                "aggregate_tokens_per_sec": round(total_tokens / total_elapsed, 1),
                "latency_p50_sec": round(p50, 3),
                "latency_p95_sec": round(p95, 3),
                "latency_p99_sec": round(p99, 3),
                "avg_tokens_per_sec_per_request": round(sum(tps_values) / len(tps_values), 1),
            }
        else:
            stats = {"total_prompts": num_prompts, "successful": 0, "errors": errors}

        _iris_log("=== RESULTS ===")
        for k, v in stats.items():
            _iris_log(f"  {k}: {v}")

        # Write results to GCS
        _save_results_to_gcs(
            model_name_or_path=model_name_or_path,
            stats=stats,
            results=results,
            prompts=prompts,
        )

        return stats


def _save_results_to_gcs(
    *,
    model_name_or_path: str,
    stats: dict,
    results: list[dict],
    prompts: list[str],
) -> None:
    """Save stress test results and samples to GCS."""
    # Derive a clean model name for the path
    model_name = model_name_or_path.rstrip("/").split("/")[-1]
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = f"{marin_prefix()}/inference/{model_name}/stress_test"

    # Write aggregate results
    results_path = f"{output_dir}/results_{timestamp}.json"
    with open_url(results_path, "w") as f:
        json.dump(
            {
                "model": model_name_or_path,
                "timestamp": timestamp,
                "stats": stats,
            },
            f,
            indent=2,
        )
    _iris_log(f"Results saved to {results_path}")

    # Write per-sample results as JSONL
    samples_path = f"{output_dir}/samples_{timestamp}.jsonl"
    with open_url(samples_path, "w") as f:
        for i, result in enumerate(results):
            sample = {
                "prompt": prompts[i] if i < len(prompts) else "",
                "response": result.get("response", ""),
                "elapsed_sec": result.get("elapsed"),
                "tokens_generated": result.get("tokens_generated"),
                "tokens_per_sec": result.get("tokens_per_sec"),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    _iris_log(f"Samples ({len(results)} rows) saved to {samples_path}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Stress test vLLM on TPU.")
    parser.add_argument("--model", required=True, help="HF model id or gs:// path")
    parser.add_argument("--num-prompts", type=int, default=5000, help="Number of prompts (default: 5000)")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent requests (default: 64)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response (default: 128)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model sequence length (default: 4096)")
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=3600,
        help="Server startup timeout in seconds before failing (default: 3600)",
    )
    parser.add_argument(
        "--native-startup-failure-mode",
        choices=["fallback", "raise"],
        default="fallback",
        help="How native mode handles async-native startup failure (default: fallback)",
    )
    parser.add_argument("--mode", choices=["docker", "native"], default=None, help="vLLM mode")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size (e.g. 4 for 70B)")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of TPU HBM reserved for KV cache (e.g. 0.25)",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Disable XLA compilation (faster startup)")
    args = parser.parse_args(argv)

    run_stress_test(
        model_name_or_path=args.model,
        num_prompts=args.num_prompts,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        startup_timeout=args.startup_timeout,
        mode=args.mode,
        native_startup_failure_mode=args.native_startup_failure_mode,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
