# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A tiny hero model, packed batch and pipeline state shared by the pipeline tests."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator, Transformer
from experiments.grug.moe_hero_pipeline.pipeline import (
    GrugMoeAutomaticPipelineState,
    split_transformer,
)


def tiny_hero(
    qb_estimator: QbEstimator,
    context_axis_size: int = 1,
    *,
    devices: Sequence[jax.Device],
    num_layers: int = 5,
    log_scaling_n_floor: int | None = None,
    log_scaling_alpha: float | None = None,
    log_scaling_power: float | None = None,
) -> tuple[Mesh, Transformer]:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        latent_dim=4,
        shared_expert_intermediate_dim=4,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=num_layers,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=4,
        max_seq_len=6,
        sliding_window=2,
        global_every=3,
        sconv=True,
        sconv_kernel=3,
        qb_estimator=qb_estimator,
        # Keep this fixture's margins away from bin edges, where scan/stage
        # roundoff can produce a full-bin quantile jump.
        qb_hist_bins=33,
        attention_implementation="reference",
        moe_implementation="scatter",
        initializer_std=0.2,
        log_scaling_n_floor=log_scaling_n_floor,
        log_scaling_alpha=log_scaling_alpha,
        log_scaling_power=log_scaling_power,
    )
    mesh = Mesh(
        np.asarray(devices[:context_axis_size]).reshape(1, 1, 1, context_axis_size, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(42))
        # Nonzero delayed convolution taps exercise packed-document boundaries.
        model = eqx.tree_at(
            lambda current: (
                current.stacked_blocks.stacked.attn.sconv_k.weight,
                current.stacked_blocks.stacked.sconv_attn.weight,
                current.stacked_blocks.stacked.sconv_mlp.weight,
            ),
            model,
            replace_fn=lambda weights: weights.at[:, 1:, :].set(0.1),
        )
    return mesh, model


def packed_batch() -> GrugLmExample:
    segments = jnp.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    return GrugLmExample(
        tokens=jnp.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=jnp.int32),
        loss_weight=jnp.array([[1.0, 0.5, 0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.5, 0.0]]),
        attn_mask=AttentionMask.causal().with_segment_ids(segments),
    )


def pipeline_state(model, optimizer, layer_counts):
    stages = split_transformer(model, len(layer_counts), layer_counts=layer_counts)
    trainable = []
    for stage in stages:
        params, _ = eqx.partition(stage, eqx.is_array)
        for index in range(len(stage.blocks)):
            params = eqx.tree_at(lambda current, index=index: current.blocks[index].mlp.router_bias, params, None)
        trainable.append(params)
    return GrugMoeAutomaticPipelineState(
        trainable_params=tuple(trainable),
        opt_state=tuple(optimizer.init(params) for params in trainable),
        pending_qb_betas=tuple(
            jnp.arange(len(stage.blocks) * model.config.num_experts, dtype=jnp.float32).reshape(
                len(stage.blocks), model.config.num_experts
            )
            + 10 * stage.start_layer
            for stage in stages
        ),
    )
