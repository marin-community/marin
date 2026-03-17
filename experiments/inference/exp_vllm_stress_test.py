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
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)

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


def send_request(server_url: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    """Send a single completion request and return timing info."""
    start = time.time()
    resp = requests.post(
        f"{server_url}/completions",
        json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    tokens_generated = data["usage"]["completion_tokens"]
    return {
        "elapsed": elapsed,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
    }


def run_stress_test(
    *,
    model_name_or_path: str,
    num_prompts: int,
    max_concurrent: int,
    max_tokens: int,
    max_model_len: int,
    mode: str | None,
) -> dict:
    is_gcs = model_name_or_path.startswith("gs://")
    if is_gcs:
        model = ModelConfig(
            name="stress-test-model",
            path=model_name_or_path,
            engine_kwargs={"max_model_len": max_model_len},
        )
    else:
        model = ModelConfig(
            name=model_name_or_path,
            path=None,
            engine_kwargs={"max_model_len": max_model_len},
        )

    env = VllmEnvironment(
        model=model,
        host="127.0.0.1",
        port=8000,
        timeout_seconds=3600,
        mode=mode,
    )

    with env:
        prompts = generate_prompts(num_prompts)
        print(f"\n{'='*60}")
        print("vLLM Stress Test")
        print(f"Model: {model_name_or_path}")
        print(f"Prompts: {num_prompts}")
        print(f"Max concurrent: {max_concurrent}")
        print(f"Max tokens per response: {max_tokens}")
        print(f"{'='*60}\n")

        results = []
        errors = 0
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(send_request, env.server_url, env.model_id, p, max_tokens): i
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

                if completed % 100 == 0 or completed == num_prompts:
                    elapsed = time.time() - total_start
                    rps = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{num_prompts}] {elapsed:.1f}s elapsed, {rps:.1f} req/s, {errors} errors")

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

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
        print(json.dumps(stats, indent=2))

        return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Stress test vLLM on TPU.")
    parser.add_argument("--model", required=True, help="HF model id or gs:// path")
    parser.add_argument("--num-prompts", type=int, default=5000, help="Number of prompts (default: 5000)")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent requests (default: 64)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response (default: 128)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model sequence length (default: 4096)")
    parser.add_argument("--mode", choices=["docker", "native"], default=None, help="vLLM mode")
    args = parser.parse_args(argv)

    run_stress_test(
        model_name_or_path=args.model,
        num_prompts=args.num_prompts,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
