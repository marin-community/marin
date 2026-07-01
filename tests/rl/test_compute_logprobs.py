# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from marin.rl.rl_losses import importance_sampling_ratio
from marin.rl.scoring import (
    LocalScoreSource,
    ModelRoles,
    ScoreRequirements,
    TokenScoreInputs,
    compute_logprobs,
    compute_logprobs_and_entropy,
)
from marin.rl.types import SequenceBatch, TrainingBatch


class DummyNamedArray:
    def __init__(self, array):
        self.array = array


class DummyModelOutput:
    def __init__(self, array):
        self.array = array


class DummyModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, *, input_ids, attn_mask, pos_ids, key):
        return DummyModelOutput(self._logits)


def _named(array, axes):
    return hax.named(jnp.asarray(array), axes)


def _make_training_batch() -> TrainingBatch:
    return TrainingBatch(
        input_ids=_named([[0, 1]], ["batch", "position"]),
        position_ids=_named([[0, 1]], ["batch", "position"]),
        loss_weights=_named([[0.0, 1.0]], ["batch", "position"]),
        loss_masks=_named([[0.0, 1.0]], ["batch", "position"]),
        policy_logprobs=_named([[0.0, -0.75]], ["batch", "position"]),
        temperature=_named([1.0], ["batch"]),
        top_k=_named([-1], ["batch"]),
        truncated=jnp.asarray([False]),
        max_output_tokens=2,
    )


def _make_sequence_batch() -> SequenceBatch:
    return SequenceBatch(
        input_ids=_named([[0, 1]], ["batch", "position"]),
        position_ids=_named([[0, 1]], ["batch", "position"]),
        prompt_mask=_named([[1.0, 0.0]], ["batch", "position"]),
        response_mask=_named([[0.0, 1.0]], ["batch", "position"]),
        behavior_logprobs=_named([[0.0, -0.25]], ["batch", "position"]),
        sampling_temperature=_named([1.0], ["batch"]),
        sampling_top_k=_named([-1], ["batch"]),
        truncated=jnp.asarray([False]),
        max_output_tokens=2,
    )


def test_compute_logprobs_one_hot_next_token():
    logits = jnp.zeros((1, 2, 10), dtype=jnp.float32)
    logits = logits.at[0, 0, 1].set(1.0)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs = compute_logprobs(model, batch, key=None)

    expected_next_logprob = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)[0, 0, 1]

    assert logprobs.shape == (1, 2)
    assert logprobs[0, 0] == pytest.approx(0.0)
    assert logprobs[0, 1] == pytest.approx(float(expected_next_logprob))


def test_compute_logprobs_and_entropy_can_skip_entropy():
    logits = jnp.zeros((1, 2, 10), dtype=jnp.float32)
    logits = logits.at[0, 0, 1].set(1.0)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=False)

    expected_next_logprob = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)[0, 0, 1]

    assert entropy is None
    assert logprobs.shape == (1, 2)
    assert logprobs[0, 0] == pytest.approx(0.0)
    assert logprobs[0, 1] == pytest.approx(float(expected_next_logprob))


def test_compute_logprobs_and_entropy_uniform_logits():
    vocab_size = 5
    logits = jnp.zeros((1, 3, vocab_size), dtype=jnp.float32)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1, 2]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1, 2]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert logprobs.shape == (1, 3)
    assert entropy.shape == (1, 3)
    assert entropy[0, 0] == pytest.approx(0.0)
    assert entropy[0, 1] == pytest.approx(float(jnp.log(vocab_size)))
    assert entropy[0, 2] == pytest.approx(float(jnp.log(vocab_size)))


def test_compute_logprobs_and_entropy_peaked_logits_have_lower_entropy():
    uniform_logits = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    peaked_logits = uniform_logits.at[0, 0, 0].set(8.0)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )

    _uniform_logprobs, uniform_entropy = compute_logprobs_and_entropy(
        DummyModel(uniform_logits), batch, key=None, compute_entropy=True
    )
    _peaked_logprobs, peaked_entropy = compute_logprobs_and_entropy(
        DummyModel(peaked_logits), batch, key=None, compute_entropy=True
    )

    assert uniform_entropy is not None
    assert peaked_entropy is not None
    assert peaked_entropy[0, 1] < uniform_entropy[0, 1]


def test_compute_logprobs_and_entropy_temperature_scales_entropy():
    logits = jnp.zeros((2, 2, 4), dtype=jnp.float32)
    logits = logits.at[:, 0, 0].set(8.0)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([0.5, 2.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    _logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert entropy[1, 1] > entropy[0, 1]


def test_compute_logprobs_and_entropy_extreme_logits_are_finite():
    logits = jnp.array([[[1000.0, -1000.0, -1000.0], [0.0, 0.0, 0.0]]], dtype=jnp.float32)

    batch = TokenScoreInputs(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        sampling_temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    _logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert jnp.isfinite(entropy[0, 1])
    assert entropy[0, 1] == pytest.approx(0.0)


def test_local_score_source_scores_training_batch_student_behavior_and_reference():
    student_logits = jnp.zeros((1, 2, 10), dtype=jnp.float32).at[0, 0, 1].set(1.0)
    reference_logits = jnp.zeros((1, 2, 10), dtype=jnp.float32).at[0, 0, 1].set(2.0)
    batch = _make_training_batch()
    score_source = LocalScoreSource(
        score_requirements=ScoreRequirements(
            student_logprobs=True,
            behavior_logprobs=True,
            reference_logprobs=True,
        )
    )

    score_bundle = score_source.score(
        batch,
        info=None,
        roles=ModelRoles(student=DummyModel(student_logits), reference=DummyModel(reference_logits)),
        key=None,
    )

    expected_student = compute_logprobs(
        DummyModel(student_logits),
        TokenScoreInputs(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            sampling_temperature=batch.temperature,
        ),
        key=None,
    )
    expected_reference = compute_logprobs(
        DummyModel(reference_logits),
        TokenScoreInputs(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            sampling_temperature=batch.temperature,
        ),
        key=None,
    )

    assert score_bundle.behavior_logprobs is not None
    assert score_bundle.student_logprobs is not None
    assert score_bundle.reference_logprobs is not None
    assert score_bundle.student_pass_count == 1
    assert score_bundle.reference_pass_count == 1
    assert score_bundle.teacher_pass_count == 0
    assert jnp.array_equal(score_bundle.behavior_logprobs, batch.policy_logprobs.array)
    assert jnp.allclose(score_bundle.student_logprobs, expected_student)
    assert jnp.allclose(score_bundle.reference_logprobs, expected_reference)


def test_local_score_source_can_request_student_entropy():
    vocab_size = 5
    logits = jnp.zeros((1, 2, vocab_size), dtype=jnp.float32)
    batch = _make_training_batch()
    score_source = LocalScoreSource(
        score_requirements=ScoreRequirements(
            student_logprobs=True,
            student_entropy=True,
        )
    )

    score_bundle = score_source.score(
        batch,
        info=None,
        roles=ModelRoles(student=DummyModel(logits)),
        key=None,
    )

    assert score_bundle.student_logprobs is not None
    assert score_bundle.student_entropy is not None
    assert score_bundle.student_entropy.shape == (1, 2)
    assert score_bundle.student_entropy[0, 0] == pytest.approx(0.0)
    assert score_bundle.student_entropy[0, 1] == pytest.approx(float(jnp.log(vocab_size)))


def test_local_score_source_rejects_entropy_with_vocab_tiling():
    score_source = LocalScoreSource(
        score_requirements=ScoreRequirements(student_entropy=True),
        vocab_tile_size=1024,
    )

    with pytest.raises(ValueError, match="not supported with vocab_tile_size"):
        score_source.score(
            _make_training_batch(),
            info=None,
            roles=ModelRoles(student=DummyModel(jnp.zeros((1, 2, 10), dtype=jnp.float32))),
            key=None,
        )


def test_local_score_source_uses_sequence_batch_behavior_logprobs():
    logits = jnp.zeros((1, 2, 10), dtype=jnp.float32).at[0, 0, 1].set(1.0)
    batch = _make_sequence_batch()
    score_source = LocalScoreSource(
        score_requirements=ScoreRequirements(
            student_logprobs=True,
            behavior_logprobs=True,
        )
    )

    score_bundle = score_source.score(
        batch,
        info=None,
        roles=ModelRoles(student=DummyModel(logits)),
        key=None,
    )

    assert score_bundle.behavior_logprobs is not None
    assert score_bundle.student_logprobs is not None
    assert score_bundle.reference_logprobs is None
    assert score_bundle.student_pass_count == 1
    assert jnp.array_equal(score_bundle.behavior_logprobs, batch.behavior_logprobs.array)


def test_local_score_source_requires_reference_model_when_requested():
    score_source = LocalScoreSource(score_requirements=ScoreRequirements(reference_logprobs=True))

    with pytest.raises(ValueError, match="reference model is required"):
        score_source.score(
            _make_training_batch(),
            info=None,
            roles=ModelRoles(student=DummyModel(jnp.zeros((1, 2, 10), dtype=jnp.float32))),
            key=None,
        )


def test_current_and_policy_importance_sampling_ratio_equals_one_when_equal():
    current_logprobs = jnp.array([[0.0, 1.0, 2.0, 3.0]])
    policy_logprobs = jnp.array([[0.0, 1.0, 2.0, 3.0]])
    loss_masks = jnp.array([[1.0, 1.0, 1.0, 1.0]])

    ratio = importance_sampling_ratio(
        current_logprobs,
        policy_logprobs,
        loss_masks,
    )

    assert ratio.shape == (1, 4)
    assert ratio[0, 0] == pytest.approx(1.0)
    assert ratio[0, 1] == pytest.approx(1.0)
    assert ratio[0, 2] == pytest.approx(1.0)
    assert ratio[0, 3] == pytest.approx(1.0)
