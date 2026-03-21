# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for vLLM inference on TPU.

Requires a v6e TPU cluster (vllm-tpu does not support v4).

Generates 100 simple math prompts ("What is 2 * x?") and runs them through
native vLLM-TPU via the OpenAI-compatible HTTP API.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster infra/marin-eu-west4-a.yaml \
        --extra vllm,tpu --env_vars WANDB_API_KEY=${WANDB_API_KEY} -- \
        python experiments/inference/exp_vllm_smoke_test.py
"""

import argparse
import time

import requests

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_prompts(n: int = 100) -> list[str]:
    return [f"What is 2 * {x}? Answer with just the number." for x in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM smoke test on TPU")
    parser.add_argument("--model", default=MODEL, help=f"HF model to use (default: {MODEL})")
    parser.add_argument("--max-model-len", type=int, default=512, help="Max model length (default: 512)")
    args = parser.parse_args()

    prompts = build_prompts(100)
    print(f"Built {len(prompts)} prompts")

    model_config = ModelConfig(
        name=args.model,
        path=None,
        engine_kwargs={"max_model_len": args.max_model_len},
    )

    env = VllmEnvironment(
        model=model_config,
        mode="native",
        host="127.0.0.1",
        timeout_seconds=1800,
    )

    with env:
        model_id = env.model_id
        server_url = env.server_url
        print(f"vLLM server ready at {server_url}, model_id={model_id}")

        print("Running inference...")
        start = time.time()
        results = []
        for prompt in prompts:
            response = requests.post(
                f"{server_url}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 32,
                },
                timeout=60,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
            results.append(text)

        elapsed = time.time() - start
        print(f"\nCompleted {len(results)} generations in {elapsed:.2f}s " f"({len(results) / elapsed:.1f} prompts/s)\n")

        # Print first 10 results
        print("=" * 60)
        print("Sample results (first 10):")
        print("=" * 60)
        for i, text in enumerate(results[:10]):
            print(f"  What is 2 * {i}?  ->  {text}")

        # Correctness check
        correct = 0
        for i, text in enumerate(results):
            if str(2 * i) in text:
                correct += 1
        print(f"\nCorrectness: {correct}/{len(results)} contain the expected answer")


if __name__ == "__main__":
    main()
