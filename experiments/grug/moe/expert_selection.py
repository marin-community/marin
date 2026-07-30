# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc expert-bank selection for Grug MoE checkpoints."""

from enum import StrEnum

import jax
import numpy as np

from experiments.grug.moe.model import Transformer


class ExpertSelectionMethod(StrEnum):
    PREFIX = "prefix"
    QB_BIAS_GREEDY = "qb_bias_greedy"
    ROUTER_NORM_GREEDY = "router_norm_greedy"
    HYBRID_GREEDY = "hybrid_greedy"
    RANDOM = "random"


def select_experts(
    model: Transformer,
    *,
    expert_count: int,
    method: ExpertSelectionMethod,
    seed: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Select one fixed expert bank per layer."""
    total_experts = model.config.num_experts
    if expert_count == total_experts:
        return None
    if expert_count <= 0 or expert_count > total_experts:
        raise ValueError(f"expert count must be in [1, {total_experts}], got {expert_count}")
    if method is ExpertSelectionMethod.PREFIX:
        return (tuple(range(expert_count)),)

    router = np.asarray(jax.device_get(model.stacked_blocks.stacked.mlp.router), dtype=np.float64)
    router_norm = np.linalg.norm(router, axis=1)
    router_bias = np.asarray(jax.device_get(model.stacked_blocks.stacked.mlp.router_bias), dtype=np.float64)
    if router_bias.ndim == 3:
        router_bias = router_bias[:, 0, :]
    if router_bias.shape != router_norm.shape:
        raise ValueError(f"router bias shape {router_bias.shape} does not match router scores {router_norm.shape}")

    if method is ExpertSelectionMethod.QB_BIAS_GREEDY:
        scores = -router_bias
    elif method is ExpertSelectionMethod.ROUTER_NORM_GREEDY:
        scores = router_norm
    elif method is ExpertSelectionMethod.HYBRID_GREEDY:
        bias_scale = np.std(router_bias, axis=-1, keepdims=True)
        norm_scale = np.std(router_norm, axis=-1, keepdims=True)
        bias_score = -(router_bias - np.mean(router_bias, axis=-1, keepdims=True)) / np.maximum(bias_scale, 1e-12)
        norm_score = (router_norm - np.mean(router_norm, axis=-1, keepdims=True)) / np.maximum(norm_scale, 1e-12)
        scores = bias_score + norm_score
    elif method is ExpertSelectionMethod.RANDOM:
        generator = np.random.default_rng(seed)
        return tuple(
            tuple(sorted(int(expert) for expert in generator.choice(total_experts, expert_count, replace=False)))
            for _ in range(model.config.num_layers)
        )
    else:
        raise ValueError(f"unsupported expert selection method: {method}")

    return tuple(
        tuple(sorted(int(expert) for expert in np.argsort(layer_scores)[-expert_count:])) for layer_scores in scores
    )
