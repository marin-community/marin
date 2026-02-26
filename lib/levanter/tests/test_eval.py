# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis

from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.eval import LossFnOutput, TaggedEvaluator
from levanter.models.lm_model import LmExample
from levanter.utils.tree_utils import inference_mode

from .test_lm_model_loss import ToyLmConfig, ToyLmHeadModel
from .test_utils import use_test_mesh


def test_tagged_evaluator_accepts_grug_lm_examples():
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    Vocab = Axis("vocab", 32)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    EvalBatch = Axis("batch", len(jax.devices()))

    examples = []
    for i in range(EvalBatch.size):
        tokens = jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i
        tokens = jnp.mod(tokens, Vocab.size)
        examples.append(GrugLmExample.causal(tokens))

    dataset = ListAsyncDataset(examples)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(model, batch) -> LossFnOutput:
            model = inference_mode(model, True)
            if isinstance(batch, LmExample):
                named_batch = batch
            else:
                named_batch = named_lm_example_from_grug(batch, Pos=model.Pos, batch_axis=EvalBatch)
            with hax.axis_mapping({EvalBatch.name: ResourceAxis.DATA}):
                per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
            return per_pos_loss, named_batch.loss_weight.array, jnp.roll(named_batch.tokens.array, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.micro_avg_loss)
    assert "grug" in result.tag_micro_losses


def test_tagged_evaluator_accepts_mixed_lm_example_types():
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    Vocab = Axis("vocab", 32)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    batch_size = max(2, len(jax.devices()) * 2)
    EvalBatch = Axis("batch", batch_size)
    Pos = Axis("position", cfg.max_seq_len)

    named_examples = []
    grug_examples = []
    for i in range(batch_size):
        named_tokens = hax.named(jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i, Vocab.size), Pos)
        named_examples.append(LmExample.causal(named_tokens))

    for i in range(batch_size):
        grug_tokens = jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + (batch_size + i), Vocab.size)
        grug_examples.append(GrugLmExample.causal(grug_tokens))

    named_dataset = ListAsyncDataset(named_examples)
    grug_dataset = ListAsyncDataset(grug_examples)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(model, batch) -> LossFnOutput:
            model = inference_mode(model, True)
            if isinstance(batch, LmExample):
                named_batch = batch
            else:
                named_batch = named_lm_example_from_grug(batch, Pos=model.Pos, batch_axis=EvalBatch)
            with hax.axis_mapping({EvalBatch.name: ResourceAxis.DATA}):
                per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
            return per_pos_loss, named_batch.loss_weight.array, jnp.roll(named_batch.tokens.array, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(named_dataset, ["named"]), (grug_dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.micro_avg_loss)
    assert "named" in result.tag_micro_losses
    assert "grug" in result.tag_micro_losses


def test_tagged_evaluator_accepts_grug_loss_protocol():
    seq_len = 8
    vocab_size = 32
    EvalBatch = Axis("batch", len(jax.devices()))

    examples = []
    for i in range(EvalBatch.size):
        tokens = jnp.mod(jnp.arange(seq_len, dtype=jnp.int32) + i, vocab_size)
        examples.append(GrugLmExample.causal(tokens))

    dataset = ListAsyncDataset(examples)

    def fake_grug_loss_fn(
        _params,
        token_ids,
        _loss_weight,
        *,
        mask=None,
        reduction="mean",
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
    ):
        del mask, logsumexp_weight
        per_token = jnp.ones_like(token_ids, dtype=loss_dtype)
        if reduction == "none":
            return per_token
        if reduction == "sum":
            return jnp.sum(per_token)
        return jnp.mean(per_token)

    with use_test_mesh(tensor_parallelism=1) as mesh:

        def loss_fn(_model, batch: GrugLmExample) -> LossFnOutput:
            per_pos_loss = fake_grug_loss_fn(
                _model,
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="none",
                logsumexp_weight=None,
            )
            return per_pos_loss, batch.loss_weight, jnp.roll(batch.tokens, -1, axis=-1)

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["grug"])],
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(jnp.zeros((), dtype=jnp.float32))

    assert np.isfinite(result.micro_avg_loss)
    assert "grug" in result.tag_micro_losses
