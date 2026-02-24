# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable

import jax
import jax.numpy as jnp
from haliax import Axis
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.data.text import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator
from levanter.grug.model import GrugModelConfig, GrugModelParameters, loss_fn as default_grug_loss_fn
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    model_config: GrugModelConfig,
    max_seq_len: int,
    trainer_runtime: TrainerConfig,
    mesh: Mesh,
    max_eval_batches: int | None,
    compute_bpb: bool,
    loss_fn: Callable[..., jax.Array] = default_grug_loss_fn,
) -> TaggedEvaluator[LmExample | GrugLmExample, GrugModelParameters] | None:
    Pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(Pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if max_eval_batches is not None:
        max_examples_per_dataset = max_eval_batches * trainer_runtime.EvalBatch.size

    tokenizer = data_config.the_tokenizer if compute_bpb else None
    compute_axis_mapping = trainer_runtime.compute_axis_mapping
    batch_axis_resource = compute_axis_mapping.get(
        trainer_runtime.EvalBatch.name,
        compute_axis_mapping.get(trainer_runtime.batch_axis_name or "batch", compute_axis_mapping.get("batch", "data")),
    )
    eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

    def eval_loss_fn(
        model: GrugModelParameters, batch: LmExample | GrugLmExample
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = loss_fn(
            model,
            batch.tokens,
            batch.loss_weight,
            model_config,
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
        axis_mapping=trainer_runtime.compute_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )
