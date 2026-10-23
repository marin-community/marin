# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from levanter.policy import ClippedPolicyConfig, KlGradient
from levanter.policy_learner import PolicyLearner, SpmdPolicyLearner
from levanter.policy_model import PolicyBatch, policy_model_loss
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.tree_utils import inference_mode


def _batch():
    devices = len(jax.devices())
    Batch, Position, Response = hax.Axis("batch", 4 * devices), hax.Axis("position", 5), hax.Axis("response", 2)
    attention = np.tile([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], (devices, 1))
    response_mask = hax.named(attention[:, -2:].astype(np.float32), (Batch, Response))
    policy_weights = response_mask / response_mask.sum()
    kl_weights = response_mask / response_mask.sum(Response) / Batch.size
    return PolicyBatch(
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


@eqx.filter_jit
def _full_response_logprobs(model, batch):
    model = inference_mode(model, True)
    mask = AttentionMask.causal().with_segment_ids(batch.attention_mask.astype(jnp.int32))
    logits = model(batch.tokens, mask, pos_ids=batch.position_ids).rearrange(("batch", "position", "vocab"))
    predictions = logits.array[:, :-1].astype(jnp.float32)
    shifted = predictions - jax.lax.stop_gradient(jnp.max(predictions, axis=-1, keepdims=True))
    # The fused loss uses full-accuracy TPU transcendentals too.
    accuracy = jax.lax.AccuracyMode.HIGHEST if jax.default_backend() == "tpu" else None
    logprobs = shifted - jax.lax.log(
        jnp.sum(jax.lax.exp(shifted, accuracy=accuracy), axis=-1, keepdims=True), accuracy=accuracy
    )
    targets = batch.tokens.rearrange(("batch", "position")).array[:, 1:]
    selected = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
    return selected[:, -batch.response_mask.axis_size("response") :]


@pytest.mark.timeout(180)
def test_spmd_policy_updates_preserve_frozen_scores_and_objective_across_microbatches(tmp_path):
    batch = _batch()
    initial = _arrays(_model())
    outcomes = []
    for per_device_parallelism in (4, 2, 1):
        config = TrainerConfig(
            id=f"policy-{per_device_parallelism}",
            train_batch_size=batch.tokens.axis_size("batch"),
            per_device_parallelism=per_device_parallelism,
            num_train_steps=2,
            tracker=NoopConfig(),
            require_accelerator=False,
            mp=jmp.get_policy("f32"),
            log_dir=tmp_path,
            log_jaxprs=False,
            log_xla_hlo=False,
            load_checkpoint=False,
        )
        loss = partial(
            policy_model_loss,
            config=ClippedPolicyConfig(0.2, 0.2, 0.0, KlGradient.DETACHED),
            accumulation_steps=4 // per_device_parallelism,
            block_size=8,
        )
        with Trainer(config, optax.adam(1e-3), loss, add_default_hooks=False) as trainer:
            state = trainer.initial_state(jax.random.PRNGKey(17), model_init=_model)
            backend: PolicyLearner[TrainerState] = SpmdPolicyLearner(trainer, block_size=8)
            scored = backend.score(state, batch)
            assert not np.allclose(scored.old_logprobs.array, batch.old_logprobs.array)
            frozen = np.asarray(scored.old_logprobs.array).copy()
            advantages = np.asarray(scored.advantages.array)
            weights = np.asarray(scored.policy_weights.array)
            for step in (1, 2):
                # Full-vocabulary scoring plus NumPy's clipped-surrogate equation
                # is independent of the learner's blocked projection and loss.
                current = np.asarray(_full_response_logprobs(state.model, scored))
                ratio = np.exp(np.clip(current - frozen, -20, 20))
                expected = -np.sum(np.minimum(ratio * advantages, np.clip(ratio, 0.8, 1.2) * advantages) * weights)
                if step == 2:
                    assert np.max(np.abs(ratio[weights > 0] - 1)) > 1e-4
                    assert not np.isclose(expected, -np.sum(advantages * weights), atol=1e-6)
                result = backend.update(state, scored)
                state = result.state
                assert int(state.step) == step
                np.testing.assert_allclose(result.loss, expected, atol=1e-6)
                np.testing.assert_array_equal(scored.old_logprobs.array, frozen)
            updated = _arrays(result.state.model)
            assert any(not np.array_equal(before, after) for before, after in zip(initial, updated, strict=True))
            outcomes.append((result.loss, _arrays(result.state)))
    for loss, arrays in outcomes[1:]:
        np.testing.assert_allclose(loss, outcomes[0][0], atol=1e-6, rtol=1e-5)
        for actual, expected in zip(arrays, outcomes[0][1], strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
