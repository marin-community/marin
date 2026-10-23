# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
import optax
import pytest

from levanter.grpo import GrpoConfig, KlGradient, grpo_objective_weights
from levanter.grpo_learner import GrpoBackend, SpmdGrpoBackend, grpo_step
from levanter.grpo_model import GrpoExample, grpo_model_loss
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState


def _batch():
    devices = len(jax.devices())
    Batch, Position, Response = hax.Axis("batch", 4 * devices), hax.Axis("position", 5), hax.Axis("response", 2)
    attention = np.tile([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], (devices, 1))
    response_mask = hax.named(attention[:, -2:].astype(np.float32), (Batch, Response))
    policy_weights, kl_weights = grpo_objective_weights(
        response_mask,
        hax.named(np.tile([0, 1, 1, 0], devices), Batch),
        Batch=Batch,
        Position=Response,
        num_partitions=2,
    )
    return GrpoExample(
        tokens=hax.named(
            np.tile([[1, 2, 3, 4, 5], [0, 6, 7, 8, 0], [9, 2, 4, 6, 8], [0, 3, 5, 7, 0]], (devices, 1)),
            (Batch, Position),
        ),
        attention_mask=hax.named(attention, (Batch, Position)),
        position_ids=hax.named(np.maximum(attention.cumsum(-1) - 1, 0), (Batch, Position)),
        response_mask=response_mask,
        advantages=hax.named(np.tile([[1, 1], [-1, 0], [2, 2], [-2, 0]], (devices, 1)), (Batch, Response)),
        policy_weights=policy_weights,
        kl_weights=kl_weights,
        old_logprobs=hax.full((Batch, Response), -9.0),
        reference_logprobs=None,
    )


def _arrays(tree):
    return [np.asarray(value).copy() for value in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _model():
    return LlamaConfig(
        max_seq_len=5,
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    ).build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(7))


@pytest.mark.timeout(180)
def test_spmd_grpo_step_preserves_global_objective_across_microbatches(tmp_path):
    batch = _batch()
    initial = _arrays(_model())
    outcomes = []
    for per_device_parallelism in (4, 2, 1):
        config = TrainerConfig(
            id=f"grpo-{per_device_parallelism}",
            train_batch_size=batch.tokens.axis_size("batch"),
            per_device_parallelism=per_device_parallelism,
            num_train_steps=1,
            tracker=NoopConfig(),
            require_accelerator=False,
            mp=jmp.get_policy("f32"),
            log_dir=tmp_path,
            log_jaxprs=False,
            log_xla_hlo=False,
            load_checkpoint=False,
        )
        loss = partial(
            grpo_model_loss,
            config=GrpoConfig(0.2, 0.2, 0.0, KlGradient.DETACHED),
            accumulation_steps=4 // per_device_parallelism,
            block_size=8,
        )
        with Trainer(config, optax.adam(1e-3), loss, add_default_hooks=False) as trainer:
            state = trainer.initial_state(jax.random.PRNGKey(17), model_init=_model)
            backend: GrpoBackend[TrainerState] = SpmdGrpoBackend(trainer, block_size=8)
            scored = backend.score(state, batch)
            assert not np.allclose(scored.old_logprobs.array, batch.old_logprobs.array)
            result = grpo_step(backend, state, batch)
            assert int(result.state.step) == 1
            np.testing.assert_allclose(
                result.loss, -(batch.advantages * batch.policy_weights).sum().scalar(), atol=1e-6
            )
            updated = _arrays(result.state.model)
            assert any(not np.array_equal(before, after) for before, after in zip(initial, updated, strict=True))
            outcomes.append((result.loss, _arrays(result.state)))
    for loss, arrays in outcomes[1:]:
        np.testing.assert_allclose(loss, outcomes[0][0], atol=1e-6, rtol=1e-5)
        for actual, expected in zip(arrays, outcomes[0][1], strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
