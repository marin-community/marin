#!/usr/bin/env python3
"""Generate K batches using a Marin env (MathEnv) and an inference endpoint.

This example constructs MathEnv directly and calls ``step()`` K times (pull API).
"""

import argparse
import asyncio
import time
from collections import deque

from .datatypes import InferenceEndpoint, Rollout
from .envs.math_env import MathEnv, MathEnvConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate K rollouts with MathEnv")
    p.add_argument(
        "--inference", type=str, required=True, help="OpenAI-compatible base URL (e.g., https://api.openai.com/v1)"
    )
    p.add_argument("--k", type=int, default=1, help="Number of iterations/rollouts to generate")
    p.add_argument(
        "--data-source", type=str, default=MathEnvConfig.data_source, help="HF dataset name for MATH examples"
    )
    p.add_argument("--split", type=str, default=MathEnvConfig.split, help="Dataset split (train/test)")
    p.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name for completions")
    p.add_argument("--seed", type=int, default=MathEnvConfig.seed, help="Random seed for sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Use the same parameters a MathEnvConfig would use; construct MathEnv directly
    env = MathEnv(
        inference=InferenceEndpoint(args.inference, model=args.model),
        data_source=args.data_source,
        split=args.split,
        api_key=None,
        seed=args.seed,
    )

    async def run_k(k: int):
        collected: deque[Rollout] = deque()
        for _ in range(k):
            batch = await env.step()
            for r in batch:
                collected.append(r)
                print(f"[rollout] id={r.rollout_uid} env={r.environment} example={r.example_id}")
        return collected

    t0 = time.time()
    collected = asyncio.run(run_k(args.k))
    dt = time.time() - t0

    print(f"\nGenerated {len(collected)} rollouts in {dt:.2f}s")


if __name__ == "__main__":
    main()
