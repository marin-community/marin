# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from marin.rl.rl_losses import (
    RLOOLoss,
    compute_ppo_loss_objective,
    compute_rloo_advantages,
    rloo_loss_with_importance_sampling,
)
from marin.rl.types import Rollout


class DummyNamedArray:
    def __init__(self, array):
        self.array = jnp.asarray(array)


def create_test_rollout(
    prompt_len: int = 8,
    response_len: int = 8,
    env_name: str = "test_env",
    episode_reward: float = 1.0,
    unique_id: int = 12345,
) -> Rollout:
    """Create a test rollout with predictable token values."""
    prompt_tokens = np.full(prompt_len, unique_id, dtype=np.int32)
    response_tokens = np.arange(response_len, dtype=np.int32) + 1000
    response_logprobs = np.full(response_len, -0.5, dtype=np.float32)
    token_rewards = np.full(response_len, 0.1, dtype=np.float32)

    return Rollout(
        env_name=env_name,
        env_example_id=f"example_{unique_id}",
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        token_rewards=token_rewards,
        episode_reward=episode_reward,
        temperature=1.0,
        top_k=None,
        is_truncated=False,
    )


def create_test_training_batch():
    policy_logprobs = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    loss_weights = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    loss_masks = jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32)

    return SimpleNamespace(
        policy_logprobs=DummyNamedArray(policy_logprobs),
        loss_weights=DummyNamedArray(loss_weights),
        loss_masks=DummyNamedArray(loss_masks),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
        top_k=DummyNamedArray(jnp.array([0], dtype=jnp.int32)),
        truncated=jnp.array([0], dtype=jnp.float32),
        max_output_tokens=3,
    )


def test_compute_rloo_advantages():
    rollout_group_rewards = [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
    ]

    rollout_groups = []
    for rewards in rollout_group_rewards:
        rollout_group = []
        for i, reward in enumerate(rewards):
            rollout = create_test_rollout(unique_id=i, episode_reward=reward)
            rollout_group.append(rollout)
        rollout_groups.append(rollout_group)

    expected_advantages = [
        [-0.5, 1.0, -0.5],
        [0.0, 0.0, 0.0],
        [-0.75, 0.0, 0.75],
    ]
    for i, rollout_group in enumerate(rollout_groups):
        advantages = compute_rloo_advantages(rollout_group)
        np.testing.assert_array_equal(advantages, expected_advantages[i])


@pytest.mark.parametrize(
    (
        "importance_sampling_ratio",
        "loss_weights",
        "loss_masks",
        "clip_epsilon",
        "trainer_inference_importance_sampling_ratio",
        "expected_loss",
    ),
    [
        # Simple no padding case
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
            0.2,
            None,
            -(0.0 + 0.5 + 1.0) / 3.0,
        ),
        # Case with padding and trainer inference importance sampling ratio
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            -(0.5 * 0.5 + 1.0 * 1.0 + 0.0 * 0.0) / 3.0,
        ),
        # Case with negative advantages
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, -0.5, -1.0, 1.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            None,
            -(-0.5 - 1.0 + 1.0) / 3.0,
        ),
        # Multi sequence case
        (
            np.array(
                [
                    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            None,
            -0.0,  # symmetric case should be 0
        ),
    ],
)
def test_ppo_objective(
    importance_sampling_ratio,
    loss_weights,
    loss_masks,
    clip_epsilon,
    trainer_inference_importance_sampling_ratio,
    expected_loss,
):
    loss, _ = compute_ppo_loss_objective(
        importance_sampling_ratio,
        loss_weights,
        loss_masks,
        clip_epsilon_low=clip_epsilon,
        clip_epsilon_high=clip_epsilon,
        max_output_tokens=loss_masks.shape[-1],
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
    )

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)


def test_rloo_loss_needs_reference_model_only_when_kl_enabled():
    assert not RLOOLoss(kl_coef=0.0).needs_reference_model()
    assert RLOOLoss(kl_coef=0.01).needs_reference_model()


def test_rloo_loss_rejects_missing_reference_model_when_kl_enabled():
    with pytest.raises(ValueError, match="reference_model is required"):
        RLOOLoss(kl_coef=0.01).create_loss_fn(reference_model=None, train_model=None)


def test_rloo_loss_policy_entropy_metric_is_opt_in_and_loss_neutral():
    batch = create_test_training_batch()
    current_logprobs = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    current_policy_entropy = jnp.array([[100.0, 0.25, 0.75]], dtype=jnp.float32)

    def compute_policy_stats_fn(_model, _batch, _key, *, compute_entropy: bool):
        assert compute_entropy
        return current_logprobs, current_policy_entropy

    loss, metrics = rloo_loss_with_importance_sampling(
        model=None,
        reference_model=None,
        batch=batch,
        key=None,
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        log_policy_entropy=True,
        compute_policy_stats_fn=compute_policy_stats_fn,
    )

    assert loss == pytest.approx(-1.0)
    assert metrics["current_policy_entropy"].value() == pytest.approx(0.5)
    assert "current_entropy" in metrics
    assert "policy_entropy" in metrics
    assert metrics["current_policy_entropy"].value() != pytest.approx(metrics["current_entropy"].value())


def test_rloo_loss_skips_policy_entropy_when_metric_disabled():
    batch = create_test_training_batch()
    current_logprobs = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

    def compute_policy_stats_fn(_model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        return current_logprobs, None

    loss, metrics = rloo_loss_with_importance_sampling(
        model=None,
        reference_model=None,
        batch=batch,
        key=None,
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        log_policy_entropy=False,
        compute_policy_stats_fn=compute_policy_stats_fn,
    )

    assert loss == pytest.approx(-1.0)
    assert "current_policy_entropy" not in metrics
    assert "current_entropy" in metrics
    assert "policy_entropy" in metrics


def test_rloo_loss_policy_entropy_does_not_request_reference_entropy():
    batch = create_test_training_batch()
    current_model = object()
    reference_model = object()
    current_logprobs = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    current_policy_entropy = jnp.array([[100.0, 0.25, 0.75]], dtype=jnp.float32)
    calls = []

    def compute_policy_stats_fn(model, _batch, _key, *, compute_entropy: bool):
        calls.append((model, compute_entropy))
        if model is reference_model:
            assert not compute_entropy
            return current_logprobs, None
        assert model is current_model
        assert compute_entropy
        return current_logprobs, current_policy_entropy

    loss, metrics = rloo_loss_with_importance_sampling(
        model=current_model,
        reference_model=reference_model,
        batch=batch,
        key=None,
        kl_coef=0.1,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        log_policy_entropy=True,
        compute_policy_stats_fn=compute_policy_stats_fn,
    )

    assert loss == pytest.approx(-1.0)
    assert metrics["current_policy_entropy"].value() == pytest.approx(0.5)
    assert metrics["kl_loss"].value() == pytest.approx(0.0)
    assert calls == [(current_model, True), (reference_model, False)]


def test_rloo_loss_rejects_missing_policy_entropy_when_metric_enabled():
    batch = create_test_training_batch()

    def compute_policy_stats_fn(_model, _batch, _key, *, compute_entropy: bool):
        assert compute_entropy
        return jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32), None

    with pytest.raises(ValueError, match="must return entropy"):
        rloo_loss_with_importance_sampling(
            model=None,
            reference_model=None,
            batch=batch,
            key=None,
            kl_coef=0.0,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.2,
            tis_importance_sampling_ratio_max=2.0,
            log_policy_entropy=True,
            compute_policy_stats_fn=compute_policy_stats_fn,
        )


def test_rloo_loss_module_rejects_policy_entropy_with_vocab_tiling():
    with pytest.raises(ValueError, match="not supported with vocab_tile_size"):
        RLOOLoss(kl_coef=0.0, log_policy_entropy=True, vocab_tile_size=1024).create_loss_fn(
            reference_model=None, train_model=None
        )


def test_rloo_loss_module_allows_vocab_tiling_when_policy_entropy_disabled():
    RLOOLoss(kl_coef=0.0, log_policy_entropy=False, vocab_tile_size=1024).create_loss_fn(
        reference_model=None, train_model=None
    )
