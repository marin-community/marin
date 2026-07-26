# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np

import haliax as hax
import haliax.nn as hnn

from levanter.distillation import (
    DistillationModel,
    DistillationObjective,
    TaidConfig,
    TaidState,
    distillation_loss,
    distillation_trainable_filter,
    forward_kl_loss,
    model_with_layer_anchors,
    projected_hidden_loss,
    taid_loss_with_state_update,
    taid_target_logits,
    update_taid_state,
)
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmExample
from levanter.models.loss import materialized_next_token_nll
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.trainer_state import TrainerState, saveable_training_mask


def _tiny_model(key):
    config = Gpt2Config(max_seq_len=8, hidden_dim=16, num_layers=1, num_heads=2)
    return config.build(hax.Axis("vocab", 12), key=key)


def test_forward_kl_matches_positional_reference_with_loss_weights():
    Batch = hax.Axis("batch", 2)
    Pos = hax.Axis("position", 3)
    Vocab = hax.Axis("vocab", 4)
    student = hax.named(
        jnp.asarray(
            [
                [[1.0, -1.0, 0.0, 0.5], [0.0, 0.5, 1.0, -0.5], [3.0, 0.0, -2.0, 1.0]],
                [[-0.5, 0.0, 0.5, 1.0], [1.5, 0.5, -0.5, -1.5], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=jnp.bfloat16,
        ),
        (Batch, Pos, Vocab),
    )
    teacher = hax.named(
        jnp.asarray(
            [
                [[0.0, 1.0, -1.0, 0.5], [2.0, 0.0, -1.0, 1.0], [0.0, 3.0, -2.0, 1.0]],
                [[1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 1.0, 2.0], [1.0, -1.0, 0.5, -0.5]],
            ],
            dtype=jnp.bfloat16,
        ),
        (Batch, Pos, Vocab),
    )
    weights_array = jnp.asarray([[1.0, 0.25, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float32)
    weights = hax.named(weights_array, (Batch, Pos))

    actual = forward_kl_loss(student, teacher, weights, Vocab=Vocab)

    teacher_probs = jax.nn.softmax(teacher.array.astype(jnp.float32), axis=-1)
    student_log_probs = jax.nn.log_softmax(student.array.astype(jnp.float32), axis=-1)
    per_position = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    expected = jnp.sum(per_position * weights_array) / jnp.sum(weights_array)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_hard_label_next_token_loss_matches_reference():
    Batch = hax.Axis("batch", 2)
    Pos = hax.Axis("position", 3)
    Vocab = hax.Axis("vocab", 4)
    logits = hax.named(
        jnp.asarray(
            [
                [[1.0, -1.0, 0.0, 0.5], [0.0, 0.5, 1.0, -0.5], [3.0, 0.0, -2.0, 1.0]],
                [[-0.5, 0.0, 0.5, 1.0], [1.5, 0.5, -0.5, -1.5], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=jnp.bfloat16,
        ),
        (Batch, Pos, Vocab),
    )
    tokens = hax.named(jnp.asarray([[0, 2, 3], [3, 1, 0]], dtype=jnp.int32), (Batch, Pos))

    actual = materialized_next_token_nll(logits, tokens, Vocab=Vocab, Pos=Pos)

    target_ids = jnp.roll(tokens.array, -1, axis=-1)
    expected = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.array.astype(jnp.float32), axis=-1),
        target_ids[..., None],
        axis=-1,
    )[..., 0]
    np.testing.assert_allclose(actual.array, expected, rtol=1e-6, atol=1e-6)


def test_teacher_has_zero_gradient_and_is_not_saveable():
    student_key, teacher_key, token_key = jrandom.split(jrandom.PRNGKey(0), 3)
    model = DistillationModel(student=_tiny_model(student_key), teacher=_tiny_model(teacher_key))
    Pos = model.student.Pos
    example = LmExample.causal(hax.random.randint(token_key, Pos, 0, model.student.Vocab.size))

    grads = eqx.filter_grad(distillation_loss)(model, example)
    teacher_grads = jax.tree.leaves(grads.teacher, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    student_grads = jax.tree.leaves(grads.student, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    assert all(grad is None or np.all(np.asarray(grad.array) == 0) for grad in teacher_grads)
    assert any(grad is not None and np.any(np.asarray(grad.array) != 0) for grad in student_grads)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=1)
    trainable_filter = distillation_trainable_filter(model)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(1),
        is_trainable=trainable_filter,
        mp=jmp.get_policy("f32"),
    )
    save_mask = saveable_training_mask(state, trainable_filter)
    assert save_mask.model.teacher is False


def test_taid_target_detaches_student_and_teacher():
    Vocab = hax.Axis("vocab", 2)
    student = hax.named(jnp.asarray([1.0, -1.0]), Vocab)
    teacher = hax.named(jnp.asarray([-0.5, 0.5]), Vocab)

    def target_sum(student_array, teacher_array):
        student_logits = hax.named(student_array, Vocab)
        teacher_logits = hax.named(teacher_array, Vocab)
        return hax.sum(taid_target_logits(student_logits, teacher_logits, jnp.asarray(0.4))).scalar()

    student_grad, teacher_grad = jax.grad(target_sum, argnums=(0, 1))(student.array, teacher.array)
    np.testing.assert_array_equal(student_grad, jnp.zeros_like(student_grad))
    np.testing.assert_array_equal(teacher_grad, jnp.zeros_like(teacher_grad))


def test_taid_update_uses_training_loss_and_linear_floor():
    config = TaidConfig()
    state = TaidState.init(config, initial_loss=2.0)
    updated = update_taid_state(
        state,
        loss=jnp.asarray(1.5),
        num_train_steps=100,
        config=config,
    )

    expected_floor = config.start
    assert float(updated.interpolation) >= expected_floor
    assert float(updated.interpolation) <= config.end
    assert float(updated.previous_loss) == 1.5


def test_taid_state_is_overwritten_by_loss_gradient():
    config = TaidConfig()
    state = TaidState.init(config, initial_loss=2.0)

    def loss_fn(taid_state):
        loss = jnp.asarray(1.5)
        return taid_loss_with_state_update(loss, taid_state, 100, config)

    state_update = jax.grad(loss_fn)(state)
    assert float(state_update.previous_loss) == 1.5
    assert int(state_update.update_count) == 1
    assert bool(state_update.has_previous_loss)


def test_taid_controller_persists_through_trainer_step():
    student_key, teacher_key, token_key = jrandom.split(jrandom.PRNGKey(2), 3)
    taid_config = TaidConfig()
    model = DistillationModel(
        student=_tiny_model(student_key),
        teacher=_tiny_model(teacher_key),
        taid_state=TaidState.init(taid_config, initial_loss=2.0),
    )
    example = LmExample.causal(hax.random.randint(token_key, model.student.Pos, 0, model.student.Vocab.size))

    def loss_fn(distillation_model):
        loss = distillation_loss(
            distillation_model,
            example,
            objective=DistillationObjective.TAID,
            taid_state=distillation_model.taid_state,
        )
        return taid_loss_with_state_update(loss, distillation_model.taid_state, 100, taid_config)

    optimizer = AdamConfig(learning_rate=1e-3).build(num_train_steps=100)
    trainable_filter = distillation_trainable_filter(model)
    state = TrainerState.init(
        optimizer,
        model,
        key=jrandom.PRNGKey(3),
        is_trainable=trainable_filter,
        mp=jmp.get_policy("f32"),
    )
    grads = eqx.filter_grad(loss_fn)(state.model)
    new_state, _ = state.take_step(grads, key=jrandom.PRNGKey(4))

    assert new_state.model.taid_state is not None
    assert int(new_state.model.taid_state.update_count) == 1
    assert float(new_state.model.taid_state.previous_loss) != 2.0


def test_layer_anchor_capture_matches_explicit_qwen_forward():
    config = Qwen3Config(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    Vocab = hax.Axis("vocab", 12)
    model = config.build(Vocab, key=jrandom.PRNGKey(0))
    tokens = hax.random.randint(jrandom.PRNGKey(1), config.max_Pos, 0, Vocab.size)
    example = LmExample.causal(tokens)

    logits, anchors = model_with_layer_anchors(
        model,
        example.tokens,
        example.attn_mask,
        (0, 2),
    )

    hidden = model.embeddings.embed(tokens)
    expected_anchors = []
    for index, layer in enumerate(model.transformer.layers.unstacked()):
        hidden = layer(hidden, mask=example.attn_mask)
        if index in (0, 2):
            expected_anchors.append(hidden)
    expected_anchors = hax.stack(hax.Axis("anchor", 2), expected_anchors)
    np.testing.assert_allclose(anchors.array, expected_anchors.array, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logits.array, model(tokens, example.attn_mask).array, rtol=1e-5, atol=1e-5)


def test_projected_hidden_loss_matches_identical_anchors():
    Batch = hax.Axis("batch", 2)
    Pos = hax.Axis("position", 3)
    Anchor = hax.Axis("anchor", 2)
    Embed = hax.Axis("embed", 4)
    TeacherEmbed = hax.Axis("teacher_embed", 4)
    anchors = hax.random.normal(jrandom.PRNGKey(0), (Batch, Pos, Anchor, Embed))
    teacher_anchors = anchors.rename({Embed: TeacherEmbed})
    weights = hax.ones((Batch, Pos))
    identity = hax.named(jnp.eye(Embed.size), (TeacherEmbed, Embed))
    projector = hnn.Linear(weight=identity, bias=None, In=Embed, Out=TeacherEmbed)

    loss = projected_hidden_loss(
        anchors,
        teacher_anchors,
        weights,
        projector,
        TeacherEmbed=TeacherEmbed,
        Anchor=Anchor,
    )
    np.testing.assert_allclose(loss, 0.0, atol=2e-5)
