#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the MoE hillclimb benchmark on TPU with a pure-JAX token layout.

The token-ragged benchmark path normally uses the DeepEP layout helper for
token-to-rank reachability. DeepEP itself is GPU-only, but the layout semantics
are simple enough to reproduce in JAX for TPU transport experiments.
"""

import jax.numpy as jnp
from iris.runtime.jax_init import initialize_jax

from lib.levanter.scripts.bench import bench_moe_hillclimb as bench


def pure_jax_dispatch_layout(
    selected_experts: jnp.ndarray,
    *,
    num_ranks: int,
    num_experts: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    local_experts = num_experts // num_ranks
    ranks = selected_experts // local_experts
    rank_ids = jnp.arange(num_ranks, dtype=ranks.dtype)
    is_token_in_rank = jnp.any(ranks[:, :, None] == rank_ids, axis=1)
    num_tokens_per_rank = jnp.sum(is_token_in_rank.astype(jnp.int32), axis=0)
    num_tokens_per_expert = jnp.bincount(selected_experts.reshape(-1), length=num_experts).astype(jnp.int32)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def main() -> None:
    initialize_jax()
    bench.deepep_get_dispatch_layout = pure_jax_dispatch_layout
    bench.main()


if __name__ == "__main__":
    main()
