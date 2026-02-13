# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis

from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.eval import TaggedEvaluator
from levanter.models.lm_model import LmExample

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
        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(dataset, ["grug"])],
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
    for i in range(batch_size // 2):
        named_tokens = hax.named(jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i, Vocab.size), Pos)
        named_examples.append(LmExample.causal(named_tokens))

    for i in range(batch_size // 2, batch_size):
        grug_tokens = jnp.mod(jnp.arange(cfg.max_seq_len, dtype=jnp.int32) + i, Vocab.size)
        grug_examples.append(GrugLmExample.causal(grug_tokens))

    named_dataset = ListAsyncDataset(named_examples)
    grug_dataset = ListAsyncDataset(grug_examples)

    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=[(named_dataset, ["named"]), (grug_dataset, ["grug"])],
            tokenizer=None,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: ResourceAxis.DATA},
        )
        result = evaluator.evaluate(model)

    assert np.isfinite(result.micro_avg_loss)
    assert "named" in result.tag_micro_losses
    assert "grug" in result.tag_micro_losses
