# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from marin.rl.decoding import DecodingConfig
from marin.rl.kl_regularization import (
    KLConfig,
    KLMode,
    k2_from_log_ratio,
    k3_from_log_ratio,
    kl_penalty_from_log_ratio,
    masked_response_mean,
)
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
        decoding=DecodingConfig(temperature=1.0).as_trace(),
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
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
    )

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)


def test_kl_penalty_dispatches_selected_estimator():
    log_ratio = np.array([[-0.3, 0.0, 0.5]], dtype=np.float32)

    np.testing.assert_allclose(kl_penalty_from_log_ratio(log_ratio, KLMode.K1_LOSS), log_ratio)
    np.testing.assert_allclose(kl_penalty_from_log_ratio(log_ratio, KLMode.K2_LOSS), 0.5 * np.square(log_ratio))
    np.testing.assert_allclose(
        kl_penalty_from_log_ratio(log_ratio, KLMode.K3_LOSS),
        np.expm1(-log_ratio) + log_ratio,
        rtol=1e-6,
        atol=0.0,
    )


def test_k3_helper_matches_stable_closed_form():
    log_ratio = np.array([[-0.3, 0.0, 0.5]], dtype=np.float32)
    expected = np.expm1(-log_ratio) + log_ratio

    np.testing.assert_allclose(k3_from_log_ratio(log_ratio), expected, rtol=1e-6, atol=0.0)


def test_k2_helper_matches_half_squared_log_ratio():
    log_ratio = np.array([[-0.3, 0.0, 0.5]], dtype=np.float32)
    expected = 0.5 * np.square(log_ratio)

    np.testing.assert_allclose(k2_from_log_ratio(log_ratio), expected)


def test_k2_and_k3_zero_at_zero_log_ratio():
    log_ratio = np.zeros((2, 3), dtype=np.float32)

    np.testing.assert_allclose(k2_from_log_ratio(log_ratio), log_ratio)
    np.testing.assert_allclose(k3_from_log_ratio(log_ratio), log_ratio)


def test_k3_helper_preserves_small_near_zero_values():
    log_ratio = np.array([[-1e-4, 1e-4, -1e-5, 1e-5]], dtype=np.float32)
    expected = 0.5 * np.square(log_ratio)
    actual = np.asarray(k3_from_log_ratio(log_ratio))

    assert np.all(actual > 0.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-15)


def test_masked_response_mean_ignores_prompt_positions():
    values = np.array([[10.0, 10.0, 1.0, 3.0], [5.0, 5.0, 2.0, 6.0]], dtype=np.float32)
    loss_masks = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    np.testing.assert_allclose(masked_response_mean(values, loss_masks), 3.0)


def test_rloo_loss_needs_reference_model_only_when_kl_enabled():
    assert not RLOOLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0)).needs_reference_model()
    assert not RLOOLoss(kl=KLConfig(mode=KLMode.K1_LOSS, beta=0.0)).needs_reference_model()
    assert not RLOOLoss(kl=KLConfig(mode=KLMode.K2_LOSS, beta=0.0)).needs_reference_model()
    assert RLOOLoss(kl=KLConfig(mode=KLMode.K1_LOSS, beta=0.01)).needs_reference_model()
    assert RLOOLoss(kl=KLConfig(mode=KLMode.K3_LOSS, beta=0.01)).needs_reference_model()
    assert RLOOLoss(kl=KLConfig(mode=KLMode.K2_LOSS, beta=0.01)).needs_reference_model()


@pytest.mark.parametrize("mode", [KLMode.K1_LOSS, KLMode.K2_LOSS, KLMode.K3_LOSS])
def test_rloo_loss_rejects_missing_reference_model_when_kl_enabled(mode: KLMode):
    with pytest.raises(ValueError, match="reference_model is required"):
        RLOOLoss(kl=KLConfig(mode=mode, beta=0.01)).create_loss_fn(reference_model=None)


@pytest.mark.parametrize(
    ("mode", "expected_kl_loss"),
    [
        (KLMode.K1_LOSS, 0.0875),
        (KLMode.K2_LOSS, 0.018125),
        (KLMode.K3_LOSS, 0.01565768),
    ],
)
def test_rloo_loss_reports_selected_kl_loss(mode: KLMode, expected_kl_loss: float):
    current_model = object()
    reference_model = object()
    current_logprobs = np.array([[0.0, 0.0, -0.2, -0.1]], dtype=np.float32)
    reference_logprobs = np.array([[0.0, 0.0, -0.4, -0.6]], dtype=np.float32)

    batch = SimpleNamespace(
        policy_logprobs=SimpleNamespace(array=current_logprobs),
        loss_weights=SimpleNamespace(array=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)),
        loss_masks=SimpleNamespace(array=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)),
        temperature=SimpleNamespace(array=np.ones((1, 4), dtype=np.float32)),
        top_k=SimpleNamespace(array=np.full((1, 4), 16, dtype=np.float32)),
        truncated=np.array([0.0], dtype=np.float32),
        max_output_tokens=2,
    )

    def compute_policy_stats_fn(model, _batch, _key, *, compute_entropy: bool):
        assert not compute_entropy
        if model is current_model:
            return current_logprobs, None
        if model is reference_model:
            return reference_logprobs, None
        raise AssertionError("unexpected model passed to compute_policy_stats_fn")

    loss, metrics = rloo_loss_with_importance_sampling(
        current_model,
        reference_model,
        batch,
        key=None,
        kl=KLConfig(mode=mode, beta=0.25),
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        compute_policy_stats_fn=compute_policy_stats_fn,
    )

    assert set(("kl_beta", "kl_k1_mean", "kl_k2_mean", "kl_k3_mean", "kl_loss")) <= metrics.keys()
    np.testing.assert_allclose(float(metrics["kl_beta"]), 0.25)
    np.testing.assert_allclose(float(metrics["kl_k1_mean"]), 0.35, atol=1e-6)
    np.testing.assert_allclose(float(metrics["kl_k2_mean"]), 0.0725, atol=1e-6)
    np.testing.assert_allclose(float(metrics["kl_k3_mean"]), 0.06263071, atol=1e-6)
    np.testing.assert_allclose(float(metrics["kl_loss"]), expected_kl_loss, atol=1e-6)
    np.testing.assert_allclose(float(loss), -1.0 + expected_kl_loss, atol=1e-6)


def test_kl_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        KLConfig(mode=KLMode.K2_LOSS, beta=-0.01)

    with pytest.raises(ValueError, match=r"beta must be 0\.0"):
        KLConfig(mode=KLMode.NONE, beta=0.01)


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
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
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
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
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
        kl=KLConfig(mode=KLMode.K3_LOSS, beta=0.1),
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
            kl=KLConfig(mode=KLMode.NONE, beta=0.0),
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.2,
            tis_importance_sampling_ratio_max=2.0,
            log_policy_entropy=True,
            compute_policy_stats_fn=compute_policy_stats_fn,
        )


def test_rloo_loss_module_rejects_policy_entropy_with_vocab_tiling():
    with pytest.raises(ValueError, match="not supported with vocab_tile_size"):
        RLOOLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0), log_policy_entropy=True, vocab_tile_size=1024).create_loss_fn(
            reference_model=None
        )


def test_rloo_loss_module_allows_vocab_tiling_when_policy_entropy_disabled():
    RLOOLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0), log_policy_entropy=False, vocab_tile_size=1024).create_loss_fn(
        reference_model=None
    )
