# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.opd_losses import (
    HybridRLOOOPDSampledTokenReverseKLLoss,
    OPDSampledTokenReverseKLLoss,
    hybrid_rloo_sampled_token_reverse_kl_opd_loss,
    sampled_token_reverse_kl_opd_loss,
)
from marin.rl.rl_losses import rloo_loss_with_importance_sampling


class DummyNamedArray:
    def __init__(self, array):
        self.array = jnp.asarray(array)


def create_test_batch(
    *,
    behavior_logprobs,
    loss_masks,
    loss_weights=None,
    truncated=None,
):
    if loss_weights is None:
        loss_weights = jnp.zeros_like(behavior_logprobs)
    if truncated is None:
        truncated = jnp.zeros((behavior_logprobs.shape[0],), dtype=jnp.float32)

    return SimpleNamespace(
        policy_logprobs=DummyNamedArray(behavior_logprobs),
        loss_weights=DummyNamedArray(loss_weights),
        loss_masks=DummyNamedArray(loss_masks),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
        top_k=DummyNamedArray(jnp.array([0], dtype=jnp.int32)),
        truncated=truncated,
        max_output_tokens=behavior_logprobs.shape[-1],
    )


def test_opd_loss_rejects_missing_teacher_model():
    with pytest.raises(ValueError, match="teacher_model is required"):
        OPDSampledTokenReverseKLLoss().create_loss_fn(reference_model=None, train_model=None)


def test_sampled_token_opd_loss_is_zero_when_teacher_behavior_and_current_match():
    current_model = object()
    teacher_model = object()
    logprobs = jnp.array([[0.0, -0.5, -1.0]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(behavior_logprobs=logprobs, loss_masks=loss_masks)

    def compute_logprobs(model, _batch, _key):
        if model in {current_model, teacher_model}:
            return logprobs
        raise AssertionError("unexpected model")

    loss, metrics = sampled_token_reverse_kl_opd_loss(
        current_model,
        teacher_model,
        batch,
        key=None,
        synchronous=False,
        clip_epsilon_low=None,
        clip_epsilon_high=None,
        compute_policy_logprobs_fn=compute_logprobs,
    )

    assert loss == pytest.approx(0.0)
    assert metrics["opd/teacher_advantage_mean"].value() == pytest.approx(0.0)
    assert metrics["opd/ratio_mean"].value() == pytest.approx(1.0)


def test_sampled_token_opd_loss_uses_teacher_logprob_minus_behavior_logprob_advantage():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -2.0]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -0.5, -1.0]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(behavior_logprobs=behavior_logprobs, loss_masks=loss_masks)

    def compute_logprobs(model, _batch, _key):
        if model is current_model:
            return behavior_logprobs
        if model is teacher_model:
            return teacher_logprobs
        raise AssertionError("unexpected model")

    loss, metrics = sampled_token_reverse_kl_opd_loss(
        current_model,
        teacher_model,
        batch,
        key=None,
        synchronous=False,
        clip_epsilon_low=None,
        clip_epsilon_high=None,
        compute_policy_logprobs_fn=compute_logprobs,
    )

    assert loss == pytest.approx(-0.75)
    assert metrics["opd/teacher_advantage_mean"].value() == pytest.approx(0.75)
    assert metrics["opd/behavior_teacher_gap_mean"].value() == pytest.approx(-0.75)


def test_sampled_token_opd_loss_clips_importance_ratio_when_configured():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -1.0]], dtype=jnp.float32)
    current_logprobs = jnp.array([[0.0, -0.5, -0.5]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -0.25, -0.25]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(behavior_logprobs=behavior_logprobs, loss_masks=loss_masks)

    def compute_logprobs(model, _batch, _key):
        if model is current_model:
            return current_logprobs
        if model is teacher_model:
            return teacher_logprobs
        raise AssertionError("unexpected model")

    loss, metrics = sampled_token_reverse_kl_opd_loss(
        current_model,
        teacher_model,
        batch,
        key=None,
        synchronous=False,
        clip_epsilon_low=0.1,
        clip_epsilon_high=0.1,
        compute_policy_logprobs_fn=compute_logprobs,
    )

    assert loss == pytest.approx(-0.825)
    assert metrics["opd/clip_fraction"].value() == pytest.approx(1.0)


def test_sampled_token_opd_loss_clips_negative_advantages_like_ppo_objective():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -1.0]], dtype=jnp.float32)
    current_logprobs = jnp.array([[0.0, -2.0, -2.0]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -1.5, -1.5]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(behavior_logprobs=behavior_logprobs, loss_masks=loss_masks)

    def compute_logprobs(model, _batch, _key):
        if model is current_model:
            return current_logprobs
        if model is teacher_model:
            return teacher_logprobs
        raise AssertionError("unexpected model")

    loss, metrics = sampled_token_reverse_kl_opd_loss(
        current_model,
        teacher_model,
        batch,
        key=None,
        synchronous=False,
        clip_epsilon_low=0.1,
        clip_epsilon_high=0.1,
        compute_policy_logprobs_fn=compute_logprobs,
    )

    assert loss == pytest.approx(0.45)
    assert metrics["opd/clip_fraction"].value() == pytest.approx(1.0)


def test_hybrid_loss_rejects_missing_required_models():
    no_kl_loss = HybridRLOOOPDSampledTokenReverseKLLoss(
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=0.1,
    )
    kl_loss = HybridRLOOOPDSampledTokenReverseKLLoss(
        kl=KLConfig(mode=KLMode.K2_LOSS, beta=0.1),
        opd_coef=0.1,
    )

    with pytest.raises(ValueError, match="teacher_model is required"):
        no_kl_loss.create_loss_fn(reference_model=None, train_model=None)
    with pytest.raises(ValueError, match="reference_model is required"):
        kl_loss.create_loss_fn(reference_model=None, train_model=None, teacher_model=object())


def test_hybrid_loss_uses_rloo_reward_advantages():
    loss = HybridRLOOOPDSampledTokenReverseKLLoss(
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=0.1,
    )
    rollouts = [
        SimpleNamespace(episode_reward=0.0),
        SimpleNamespace(episode_reward=1.0),
        SimpleNamespace(episode_reward=0.0),
    ]

    np.testing.assert_allclose(loss.compute_advantages(rollouts), np.array([-0.5, 1.0, -0.5]))


def test_hybrid_loss_with_zero_opd_coef_matches_rloo_reward_loss():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -1.0]], dtype=jnp.float32)
    reward_advantages = jnp.array([[0.0, 0.25, -0.5]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(
        behavior_logprobs=behavior_logprobs,
        loss_weights=reward_advantages,
        loss_masks=loss_masks,
    )

    def compute_policy_stats(model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        if model in {current_model, teacher_model}:
            return behavior_logprobs, None
        raise AssertionError("unexpected model")

    hybrid_loss, hybrid_metrics = hybrid_rloo_sampled_token_reverse_kl_opd_loss(
        current_model,
        reference_model=None,
        teacher_model=teacher_model,
        batch=batch,
        key=None,
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats,
    )
    rloo_loss, rloo_metrics = rloo_loss_with_importance_sampling(
        current_model,
        reference_model=None,
        batch=batch,
        key=None,
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats,
    )

    assert hybrid_loss == pytest.approx(rloo_loss)
    assert hybrid_metrics["hybrid/opd_loss"].value() == pytest.approx(0.0)
    assert hybrid_metrics["reinforce_loss"].value() == pytest.approx(rloo_metrics["reinforce_loss"].value())


def test_hybrid_loss_with_zero_reward_advantage_matches_sampled_token_opd_loss():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -2.0]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -0.5, -1.0]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(behavior_logprobs=behavior_logprobs, loss_masks=loss_masks)

    def compute_policy_stats(model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        if model is current_model:
            return behavior_logprobs, None
        if model is teacher_model:
            return teacher_logprobs, None
        raise AssertionError("unexpected model")

    def compute_logprobs(model, _batch, _key):
        logprobs, _entropy = compute_policy_stats(model, _batch, _key, compute_entropy=False)
        return logprobs

    hybrid_loss, hybrid_metrics = hybrid_rloo_sampled_token_reverse_kl_opd_loss(
        current_model,
        reference_model=None,
        teacher_model=teacher_model,
        batch=batch,
        key=None,
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=1.0,
        clip_epsilon_low=10.0,
        clip_epsilon_high=10.0,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats,
    )
    opd_loss, _opd_metrics = sampled_token_reverse_kl_opd_loss(
        current_model,
        teacher_model,
        batch,
        key=None,
        synchronous=False,
        clip_epsilon_low=None,
        clip_epsilon_high=None,
        compute_policy_logprobs_fn=compute_logprobs,
    )

    assert hybrid_loss == pytest.approx(opd_loss)
    assert hybrid_metrics["hybrid/opd_advantage_mean"].value() == pytest.approx(0.75)


def test_hybrid_loss_combines_reward_and_opd_advantages():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -2.0]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -0.5, -1.0]], dtype=jnp.float32)
    reward_advantages = jnp.array([[0.0, 1.0, -0.5]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(
        behavior_logprobs=behavior_logprobs,
        loss_weights=reward_advantages,
        loss_masks=loss_masks,
    )

    def compute_policy_stats(model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        if model is current_model:
            return behavior_logprobs, None
        if model is teacher_model:
            return teacher_logprobs, None
        raise AssertionError("unexpected model")

    loss, metrics = hybrid_rloo_sampled_token_reverse_kl_opd_loss(
        current_model,
        reference_model=None,
        teacher_model=teacher_model,
        batch=batch,
        key=None,
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=0.5,
        clip_epsilon_low=10.0,
        clip_epsilon_high=10.0,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats,
    )

    assert metrics["hybrid/reward_advantage_mean"].value() == pytest.approx(0.25)
    assert metrics["hybrid/opd_advantage_mean"].value() == pytest.approx(0.75)
    assert metrics["hybrid/combined_advantage_mean"].value() == pytest.approx(0.625)
    assert metrics["hybrid/reward_loss"].value() == pytest.approx(-0.25)
    assert metrics["hybrid/opd_loss"].value() == pytest.approx(-0.375)
    assert metrics["hybrid/combined_reinforce_loss"].value() == pytest.approx(-0.625)
    assert metrics["hybrid/opd_coef"].value() == pytest.approx(0.5)
    assert metrics["hybrid/current_teacher_gap_mean"].value() == pytest.approx(-0.75)
    assert metrics["hybrid/behavior_teacher_gap_mean"].value() == pytest.approx(-0.75)
    assert metrics["kl_loss"].value() == pytest.approx(0.0)
    assert loss == pytest.approx(-0.625)


def test_hybrid_loss_clips_negative_combined_advantages_like_ppo_objective():
    current_model = object()
    teacher_model = object()
    behavior_logprobs = jnp.array([[0.0, -1.0, -1.0]], dtype=jnp.float32)
    current_logprobs = jnp.array([[0.0, -2.0, -2.0]], dtype=jnp.float32)
    teacher_logprobs = jnp.array([[0.0, -1.5, -1.5]], dtype=jnp.float32)
    reward_advantages = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    batch = create_test_batch(
        behavior_logprobs=behavior_logprobs,
        loss_weights=reward_advantages,
        loss_masks=loss_masks,
    )

    def compute_policy_stats(model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        if model is current_model:
            return current_logprobs, None
        if model is teacher_model:
            return teacher_logprobs, None
        raise AssertionError("unexpected model")

    loss, metrics = hybrid_rloo_sampled_token_reverse_kl_opd_loss(
        current_model,
        reference_model=None,
        teacher_model=teacher_model,
        batch=batch,
        key=None,
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        opd_coef=1.0,
        clip_epsilon_low=0.1,
        clip_epsilon_high=0.1,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats,
    )

    assert loss == pytest.approx(0.45)
    assert metrics["clip_fraction"].value() == pytest.approx(1.0)
