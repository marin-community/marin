# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU probe for `build_schedule` drop behaviour at expert-parallel scale.

Upstream MoK is dropless and traps on overflow, so every dropped assignment in
this backend comes from `build_schedule` clipping to a static capacity. The
probe sweeps routing imbalance against the capacity factor to show where that
clipping begins, on CPU and at any rank count.

It was written to explain the 18.84% drop fraction the first EP64 runs
reported. The answer is routing, not scale: balanced routing drops nothing at
sixty-four ranks even at the matched factor of 1.1, and drops begin only once
one expert's load passes the capacity factor times the balanced load.

Run directly: `uv run experiments/grug/moe_hero_ep/schedule_drop_probe.py`.
"""

import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from levanter.kernels.mixture_of_kittens.config import MokLikeConfig
from levanter.kernels.mixture_of_kittens.schedule import build_schedule, schedule_capacity


def route_uniform(rng: np.random.Generator, world: int, tokens: int, top_k: int, num_experts: int) -> np.ndarray:
    """Sample top-k routes uniformly without replacement per token."""
    routes = np.empty((world, tokens, top_k), dtype=np.int32)
    for w in range(world):
        for t in range(tokens):
            routes[w, t] = rng.choice(num_experts, size=top_k, replace=False)
    return routes


def route_from_logits(
    rng: np.random.Generator,
    world: int,
    tokens: int,
    top_k: int,
    num_experts: int,
    hidden: int,
) -> np.ndarray:
    """Route with a randomly initialized linear router over random token embeddings.

    This is the distribution an untrained model actually produces: correlated
    across tokens through the shared router matrix, so far less balanced than
    independent uniform sampling.
    """
    router = rng.normal(scale=hidden**-0.5, size=(hidden, num_experts))
    embeddings = rng.normal(size=(world * tokens, hidden))
    logits = embeddings @ router
    top = np.argpartition(-logits, top_k - 1, axis=1)[:, :top_k]
    return top.astype(np.int32).reshape(world, tokens, top_k)


def route_skewed(
    rng: np.random.Generator,
    world: int,
    tokens: int,
    top_k: int,
    num_experts: int,
    concentration: float,
) -> np.ndarray:
    """Route from a Dirichlet expert prior, so load imbalance is tunable.

    Low `concentration` concentrates mass on a few experts, which is what an
    unregularized router looks like when it collapses.
    """
    prior = rng.dirichlet(np.full(num_experts, concentration))
    routes = np.empty((world, tokens, top_k), dtype=np.int32)
    for w in range(world):
        for t in range(tokens):
            routes[w, t] = rng.choice(num_experts, size=top_k, replace=False, p=prior)
    return routes


def measure(routes: np.ndarray, num_local_experts: int, capacity: int, max_ranks: int | None = None) -> dict:
    """Return drop statistics aggregated over destination ranks."""
    world = routes.shape[0]
    num_ranks = world if max_ranks is None else min(world, max_ranks)
    builder = jax.jit(
        lambda values, rank: build_schedule(
            values,
            num_local_experts=num_local_experts,
            schedule_capacity=capacity,
            rank=rank,
        )
    )
    routes_device = jnp.asarray(routes)
    num_experts = world * num_local_experts
    expert_loads = np.bincount(routes.reshape(-1), minlength=num_experts)

    dropped = 0
    covered = 0
    for rank in range(num_ranks):
        schedule = builder(routes_device, jnp.asarray(rank, dtype=jnp.int32))
        dropped += int(schedule.dropped_assignments)
        first = rank * num_local_experts
        covered += int(expert_loads[first : first + num_local_experts].sum())

    mean_load = int(routes.size) / num_experts
    return {
        "dropped": dropped,
        "sampled_assignments": covered,
        "drop_fraction": dropped / covered if covered else 0.0,
        "capacity": capacity,
        "mean_expert_load": mean_load,
        "peak_expert_load": int(expert_loads.max()),
        "peak_over_mean": int(expert_loads.max()) / mean_load,
        "ranks_measured": num_ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, default=64, help="expert-parallel rank count")
    parser.add_argument("--tokens", type=int, default=512, help="tokens per rank (scaled down from 65536)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048, help="router input width for the logits distribution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capacity-factors",
        type=float,
        nargs="+",
        default=[1.1, 1.5, 2.0, 4.0],
    )
    parser.add_argument("--minibatch-size", type=int, default=4096, help="capacity rounding granularity")
    parser.add_argument(
        "--dirichlet",
        type=float,
        nargs="*",
        default=[1.0, 0.1],
        help="Dirichlet concentrations for skewed routing; lower is more collapsed",
    )
    parser.add_argument(
        "--max-ranks",
        type=int,
        default=None,
        help="measure only this many destination ranks (for full-scale token counts)",
    )
    args = parser.parse_args()

    if args.num_experts % args.world != 0:
        raise ValueError(f"num_experts={args.num_experts} must divide evenly across world={args.world}")
    num_local_experts = args.num_experts // args.world

    rng = np.random.default_rng(args.seed)
    distributions = {
        "uniform": route_uniform(rng, args.world, args.tokens, args.top_k, args.num_experts),
        "untrained-router": route_from_logits(rng, args.world, args.tokens, args.top_k, args.num_experts, args.hidden),
    }
    for concentration in args.dirichlet:
        distributions[f"dirichlet-{concentration:g}"] = route_skewed(
            rng, args.world, args.tokens, args.top_k, args.num_experts, concentration
        )

    print(
        f"world={args.world} tokens/rank={args.tokens} top_k={args.top_k} "
        f"num_experts={args.num_experts} local_experts={num_local_experts} "
        f"ranks_measured={args.max_ranks or args.world}"
    )
    header = f"{'routing':18} {'factor':>7} {'capacity':>9} {'peak/mean':>10} {'drop %':>8}"
    print(header)
    print("-" * len(header))
    for name, routes in distributions.items():
        for factor in args.capacity_factors:
            config = dataclasses.replace(
                MokLikeConfig(),
                schedule_capacity_factor=factor,
                minibatch_size=args.minibatch_size,
            )
            capacity = schedule_capacity(args.tokens, args.top_k, num_local_experts, config)
            stats = measure(routes, num_local_experts, capacity, max_ranks=args.max_ranks)
            print(
                f"{name:18} {factor:7.2f} {capacity:9d} "
                f"{stats['peak_over_mean']:10.3f} {100 * stats['drop_fraction']:8.2f}"
            )


if __name__ == "__main__":
    main()
