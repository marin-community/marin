"""Install the Megatron router replay controller on a built Megatron model.

Attaches a :class:`MegatronRouterReplay` (via per-router
:class:`LayerReplayHandle` objects) to every local ``TopKRouter`` of a
Megatron-Bridge-built model, keyed by the global ``layer_number`` → capture
index mapping. mcore calls ``router.router_replay.get_replay_topk(...)`` from
``topk_routing_with_score_function`` for every score path that goes through
``compute_topk``; we keep ``moe_enable_routing_replay=False`` so mcore does not
create its own inference-oriented ``RouterReplay`` in that slot.

Flag-off invariant: this function is only called when router replay is
requested, so a flag-off model keeps ``router.router_replay is None``.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from loguru import logger
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.models.megatron_router_replay import (
    MIN_ROUTER_TOPK,
    LayerReplayHandle,
    MegatronRouterReplay,
    RouterScoreType,
    capture_layer_indices,
    expand_moe_layer_freq,
    num_moe_layers,
)


def _validate_replay_compatible_config(config) -> None:
    """Reject mcore routing features that bypass or corrupt the replay hook."""
    if config.moe_enable_routing_replay:
        raise ValueError(
            "router replay: moe_enable_routing_replay must stay False; the trainer owns the "
            "router_replay slot on each TopKRouter"
        )
    if config.moe_router_fusion:
        raise ValueError(
            "router replay: moe_router_fusion=True bypasses the router_replay hook "
            "(fused_topk_with_score_function returns before compute_topk)"
        )
    if config.moe_router_load_balancing_type == "sinkhorn":
        raise ValueError("router replay: moe_router_load_balancing_type='sinkhorn' uses a separate routing path")
    if config.moe_expert_capacity_factor is not None:
        raise ValueError(
            "router replay: moe_expert_capacity_factor drops tokens after top-k and would silently drop replayed tokens"
        )
    if config.moe_router_topk < MIN_ROUTER_TOPK:
        raise ValueError(
            f"router replay: moe_router_topk must be >= {MIN_ROUTER_TOPK} for the all-K sentinel convention"
        )


def install_megatron_router_replay(
    actor_module: Sequence[nn.Module], *, recompute_enabled: bool
) -> MegatronRouterReplay:
    """Install per-router replay handles on every local ``TopKRouter``.

    ``actor_module`` is the per-VPP-chunk model list the worker built. Fails
    fast (``ValueError`` naming the config field) on any mcore routing feature
    whose code path bypasses or corrupts the replay hook. Returns the
    controller; PP stages that own no MoE layers get a controller with no
    local layers (their forward never fires a router).
    """
    config = get_model_config(actor_module[0])
    _validate_replay_compatible_config(config)

    pattern = expand_moe_layer_freq(config.moe_layer_freq, config.num_layers)
    mapping = capture_layer_indices(pattern)

    chunk_layers = []
    for vp_stage, chunk in enumerate(actor_module):
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        num_local = get_num_layers_to_build(config, vp_stage=vp_stage)
        indices = tuple(
            sorted(
                mapping[layer_number]
                for layer_number in range(offset + 1, offset + num_local + 1)
                if layer_number in mapping
            )
        )
        chunk_layers.append((chunk, indices))

    expected_local = {idx for _, indices in chunk_layers for idx in indices}
    controller = MegatronRouterReplay(sorted(expected_local), recompute_enabled=recompute_enabled)
    controller.num_moe_layers_total = num_moe_layers(pattern)
    controller.topk = config.moe_router_topk
    controller.local_indices_for_module = {id(chunk): indices for chunk, indices in chunk_layers}

    found: set[int] = set()
    for chunk, _indices in chunk_layers:
        for module in chunk.modules():
            if not isinstance(module, TopKRouter):
                continue
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None:
                raise ValueError("router replay: TopKRouter.layer_number is None; MoELayer.set_layer_number never ran")
            if layer_number not in mapping:
                raise ValueError(
                    f"router replay: TopKRouter at layer_number={layer_number} is not a MoE layer by the pattern"
                )
            capture_idx = mapping[layer_number]
            declared_score_type = getattr(module, "replay_score_type", None)
            if declared_score_type is not None:
                score_type = declared_score_type
            elif module.score_function == "softmax" and not module.config.moe_router_pre_softmax:
                score_type = RouterScoreType.LOGITS
            elif module.score_function == "softmax" or module.expert_bias is None:
                score_type = RouterScoreType.PROBABILITIES
            else:
                score_type = RouterScoreType.BIASED_PROBABILITIES
            module.router_replay = LayerReplayHandle(controller, capture_idx, score_type)
            found.add(capture_idx)

    if found != expected_local:
        raise ValueError(
            f"router replay: found TopKRouter capture indices {sorted(found)} but this rank owns "
            f"{sorted(expected_local)} MoE layers by the layer pattern"
        )
    logger.info(
        f"router replay: installed handles on {len(found)} MoE router(s) "
        f"(capture indices {sorted(found)} of {controller.num_moe_layers_total} total, "
        f"recompute FIFO {'on' if recompute_enabled else 'off'})"
    )
    return controller
