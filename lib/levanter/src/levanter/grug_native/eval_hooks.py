# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging

import jax
import jax.numpy as jnp
from haliax import Axis
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.data.text import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.model import GrugModelConfig, Transformer
from levanter.models.lm_model import LmExample

from .runtime import GrugRuntime

logger = logging.getLogger(__name__)


def _default_grug_loss_fn(
    model: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mask: GrugAttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
) -> jax.Array:
    return model.next_token_loss(
        token_ids,
        loss_weight,
        mask=mask,
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    model_config: GrugModelConfig,
    max_seq_len: int,
    trainer_runtime: GrugRuntime,
    mesh: Mesh,
    max_eval_batches: int | None,
    compute_bpb: bool,
    eval_batch_pspec: P,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    Pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(Pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if max_eval_batches is not None:
        max_examples_per_dataset = max_eval_batches * trainer_runtime.EvalBatch.size

    tokenizer = data_config.the_tokenizer if compute_bpb else None
    if len(eval_batch_pspec) != 1:
        raise ValueError(f"eval_batch_pspec must describe a single logical batch axis, got {eval_batch_pspec}")
    batch_axis_resource = eval_batch_pspec[0]
    if batch_axis_resource is not None and not isinstance(batch_axis_resource, (str, tuple)):
        raise ValueError(f"eval_batch_pspec must map to mesh axis names, got {eval_batch_pspec}")
    eval_axis_mapping = {trainer_runtime.EvalBatch.name: batch_axis_resource}
    eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=trainer_runtime.EvalBatch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )
