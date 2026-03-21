# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Stress test for vLLM inference on TPU.

Hammers the vLLM server with concurrent requests for a fixed duration (default
5 minutes) to measure sustained throughput and latency under load.

Usage:
    uv run iris --controller-url=http://127.0.0.1:51379 job run \
        --tpu v6e-4 --memory 16GB --extra vllm,tpu \
        -e WANDB_API_KEY $WANDB_API_KEY --no-wait \
        -- python experiments/inference/exp_vllm_stress_test.py
"""

import argparse
import asyncio
import statistics
import time

import aiohttp

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and Rust?",
    "Describe the water cycle step by step.",
    "Write a recipe for chocolate chip cookies.",
    "Explain how a CPU executes instructions.",
    "What caused the fall of the Roman Empire?",
    "Write a haiku about machine learning.",
    "Explain quantum entanglement to a 10-year-old.",
    "What are the pros and cons of nuclear energy?",
    "Describe the process of photosynthesis.",
    "Write a persuasive argument for space exploration.",
    "How does TCP/IP networking work?",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a limerick about a programmer debugging code.",
    "What is the significance of the Turing test?",
]


async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int]:
    """Send one chat completion request. Returns (latency_seconds, output_tokens)."""
    start = time.monotonic()
    async with session.post(
        f"{server_url}/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        },
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    latency = time.monotonic() - start
    output_tokens = data["usage"]["completion_tokens"]
    return latency, output_tokens


async def run_stress_test(
    server_url: str,
    model_id: str,
    duration_seconds: int,
    concurrency: int,
    max_tokens: int,
) -> None:
    """Fire concurrent requests for the given duration and report stats."""
    latencies: list[float] = []
    total_output_tokens = 0
    total_requests = 0
    errors = 0
    prompt_idx = 0

    start = time.monotonic()
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Semaphore limits in-flight requests
        sem = asyncio.Semaphore(concurrency)
        tasks: set[asyncio.Task] = set()

        async def worker() -> None:
            nonlocal prompt_idx, total_requests, total_output_tokens, errors
            async with sem:
                prompt = PROMPTS[prompt_idx % len(PROMPTS)]
                prompt_idx += 1
                try:
                    latency, out_tok = await send_request(session, server_url, model_id, prompt, max_tokens)
                    latencies.append(latency)
                    total_output_tokens += out_tok
                    total_requests += 1
                except Exception as e:
                    errors += 1
                    print(f"  Request error: {e}")

        last_report = start
        while time.monotonic() - start < duration_seconds:
            # Keep filling up to concurrency
            while len(tasks) < concurrency and time.monotonic() - start < duration_seconds:
                t = asyncio.create_task(worker())
                tasks.add(t)
                t.add_done_callback(tasks.discard)

            # Wait a bit before spawning more
            await asyncio.sleep(0.05)

            # Periodic progress report every 30s
            now = time.monotonic()
            if now - last_report >= 30:
                elapsed = now - start
                rps = total_requests / elapsed if elapsed > 0 else 0
                tps = total_output_tokens / elapsed if elapsed > 0 else 0
                print(
                    f"  [{elapsed:.0f}s] requests={total_requests} errors={errors} "
                    f"throughput={rps:.1f} req/s  {tps:.0f} tok/s"
                )
                last_report = now

        # Drain remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - start

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    print(f"  Duration:          {elapsed:.1f}s")
    print(f"  Concurrency:       {concurrency}")
    print(f"  Max tokens/req:    {max_tokens}")
    print(f"  Total requests:    {total_requests}")
    print(f"  Errors:            {errors}")
    print(f"  Total output toks: {total_output_tokens}")
    print(f"  Throughput:        {total_requests / elapsed:.1f} req/s")
    print(f"  Token throughput:  {total_output_tokens / elapsed:.0f} tok/s")
    if latencies:
        print(f"  Latency mean:      {statistics.mean(latencies):.2f}s")
        print(f"  Latency median:    {statistics.median(latencies):.2f}s")
        print(f"  Latency p95:       {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}s")
        print(f"  Latency p99:       {sorted(latencies)[int(len(latencies) * 0.99)]:.2f}s")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM stress test on TPU")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per request")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    args = parser.parse_args()

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
        print(
            f"Starting stress test: {args.duration}s, concurrency={args.concurrency}, " f"max_tokens={args.max_tokens}"
        )

        asyncio.run(
            run_stress_test(
                server_url=server_url,
                model_id=model_id,
                duration_seconds=args.duration,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
            )
        )


if __name__ == "__main__":
    main()
