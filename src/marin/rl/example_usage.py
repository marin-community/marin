#!/usr/bin/env python3
"""Generate K rollouts using a MarinEnvConfig (MathEnv) and an inference endpoint.

This example shows how to instantiate a concrete Marin environment (MathEnv)
without Ray by constructing it directly with the same parameters as the
config, and running its async loop to produce K rollouts.
"""

import argparse
import asyncio
import time
from collections import deque

from .datatypes import InferenceEndpoint, RolloutGroup
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
    p.add_argument("--model", type=str, default=MathEnvConfig.model, help="Model name for completions")
    p.add_argument("--seed", type=int, default=MathEnvConfig.seed, help="Random seed for sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build the rollout sink that collects groups and prints a brief summary
    collected: deque[RolloutGroup] = deque()

    def sink(groups: list[RolloutGroup]) -> None:
        for g in groups:
            collected.append(g)
            print(
                f"[group] id={g.id} env={g.environment} example={g.example_id} "
                f"rollouts={len(g.rollouts)} ts={g.sealed_ts:.0f}"
            )

    # Use the same parameters a MathEnvConfig would use; we construct MathEnv directly
    env = MathEnv(
        inference=InferenceEndpoint(args.inference, model=args.model),
        rollout_sink=sink,  # type: ignore[arg-type]
        data_source=args.data_source,
        split=args.split,
        max_iters=args.k,
        api_key=None,
        seed=args.seed,
    )

    t0 = time.time()
    asyncio.run(env.run())
    dt = time.time() - t0

    print(f"\nGenerated {len(collected)} rollout group(s) in {dt:.2f}s")


if __name__ == "__main__":
    main()
