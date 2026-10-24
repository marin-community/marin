import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import PolicyLossInputs, mixed_policy_grpo_loss, ppo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch


GAMMA = 0.1


@pytest.fixture
def mixed_config():
    return OmegaConf.create(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "loss_reduction": "token_mean",
            "use_kl_loss": False,
            "mixed_policy": {"gamma": GAMMA},
        }
    )


def mixed_inputs(log_probs, advantages, mask, behavior_sources, rollout_log_probs, old_log_probs=None):
    return PolicyLossInputs(
        log_probs=log_probs,
        old_log_probs=old_log_probs if old_log_probs is not None else log_probs.detach(),
        rollout_log_probs=rollout_log_probs,
        advantages=advantages,
        loss_mask=mask,
        behavior_sources=behavior_sources,
    )


def test_teacher_token_uses_shaped_unclipped_ratio(mixed_config):
    # Student token: ratio 1.0 (old == current) -> value -A. Teacher token: rollout
    # logprob -1, current -1 -> r = 1, shaped 1/(1+gamma) -> value -A/(1+gamma).
    log_probs = torch.tensor([[-1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    sources = torch.tensor([[0, 1]], dtype=torch.float64)
    rollout_log_probs = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    result = mixed_policy_grpo_loss(mixed_inputs(log_probs, advantages, mask, sources, rollout_log_probs), mixed_config)
    expected_student = -2.0
    expected_teacher = -2.0 / (1.0 + GAMMA)
    torch.testing.assert_close(result.values[0, 0], torch.tensor(expected_student, dtype=torch.float64))
    torch.testing.assert_close(result.values[0, 1], torch.tensor(expected_teacher, dtype=torch.float64))
    assert result.metrics["mixed/teacher_token_fraction"] == pytest.approx(0.5)


def test_teacher_gradient_matches_regularized_importance_sampling(mixed_config):
    # Column 0 is a student token: unclipped ratio gradient -A*r. Column 1 is a
    # teacher token: d/ddelta[-f(r)*A] with r = exp(delta), f(r) = r/(r+gamma)
    # equals -A * gamma * r / (r + gamma)^2 (LUFFY Eq. 7 with a real pi_phi).
    log_probs = torch.tensor([[-1.0, -2.0]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[1.5, -0.5]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    sources = torch.tensor([[0, 1]], dtype=torch.float64)
    rollout_log_probs = torch.tensor([[-3.0, -4.0]], dtype=torch.float64)
    inputs = mixed_inputs(
        log_probs, advantages, mask, sources, rollout_log_probs, old_log_probs=log_probs.detach() - 0.05
    )
    values = mixed_policy_grpo_loss(inputs, mixed_config).values
    grad = torch.autograd.grad(values.sum(), log_probs)[0]
    student_ratio = torch.exp(torch.tensor(0.05, dtype=torch.float64))
    torch.testing.assert_close(grad[0, 0], -advantages[0, 0] * student_ratio, rtol=1e-6, atol=1e-8)
    teacher_ratio = torch.exp(torch.tensor(2.0, dtype=torch.float64))
    expected = -advantages[0, 1] * GAMMA * teacher_ratio / (teacher_ratio + GAMMA) ** 2
    torch.testing.assert_close(grad[0, 1], expected, rtol=1e-6, atol=1e-8)


def test_without_teacher_tokens_the_loss_matches_regular_and_reports_zero_fraction(mixed_config):
    log_probs = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    rollout_log_probs = torch.tensor([[-1.5, -2.5]], dtype=torch.float64)
    result = mixed_policy_grpo_loss(mixed_inputs(log_probs, advantages, mask, None, rollout_log_probs), mixed_config)
    reference = ppo_policy_loss(
        PolicyLossInputs(log_probs, log_probs, rollout_log_probs, advantages, mask), mixed_config
    )
    torch.testing.assert_close(result.values, reference.values)
    assert result.metrics["mixed/teacher_token_fraction"] == 0.0


def test_teacher_ess_equals_token_count_for_uniform_weights(mixed_config):
    log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float64)
    sources = torch.tensor([[0, 1, 1, 1]], dtype=torch.float64)
    mask = torch.ones(1, 4, dtype=torch.float64)
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    rollout_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float64)
    result = mixed_policy_grpo_loss(mixed_inputs(log_probs, advantages, mask, sources, rollout_log_probs), mixed_config)
    assert result.metrics["mixed/teacher_effective_sample_size"] == pytest.approx(3.0)
    assert result.metrics["mixed/teacher_shaped_weight_mean"] == pytest.approx(1.0 / (1.0 + GAMMA))
    assert result.metrics["mixed/teacher_ratio_mean"] == pytest.approx(1.0)


def test_low_probability_teacher_tokens_are_reported(mixed_config):
    log_probs = torch.tensor([[-9.0]], dtype=torch.float64)
    sources = torch.tensor([[1]], dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    rollout_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    result = mixed_policy_grpo_loss(mixed_inputs(log_probs, advantages, mask, sources, rollout_log_probs), mixed_config)
    assert result.metrics["mixed/teacher_ratio_below_floor_fraction"] == pytest.approx(1.0)
    assert result.metrics["mixed/teacher_ratio_high_saturation_fraction"] == pytest.approx(0.0)


def test_missing_rollout_logprobs_raise(mixed_config):
    log_probs = torch.tensor([[-1.0]], dtype=torch.float64)
    sources = torch.tensor([[1]], dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="rollout logprobs"):
        mixed_policy_grpo_loss(mixed_inputs(log_probs, advantages, mask, sources, None), mixed_config)


def test_objective_micro_batch_carries_sanitized_behavior_sources():
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    behavior_sources = torch.tensor([[0, 1, 1]], dtype=torch.uint8)
    batch = build_objective_micro_batch(
        action_log_probs=torch.tensor([[-1.0, -1.0, -1.0]]),
        old_action_log_probs=torch.tensor([[-1.0, -1.0, -1.0]]),
        base_action_log_probs=None,
        advantages=torch.tensor([[1.0, 1.0, 1.0]]),
        loss_mask=loss_mask,
        rollout_logprobs=torch.tensor([[-1.0, -1.0, -1.0]]),
        response_span_tags=None,
        token_entropy=torch.tensor([[0.1, 0.1, 0.1]]),
        think_token_weight=1.0,
        teacher=None,
        behavior_sources=behavior_sources,
    )
    torch.testing.assert_close(batch.policy.behavior_sources, torch.tensor([[0.0, 1.0, 0.0]]))


def test_rank_metrics_match_when_only_one_rank_has_teacher_tokens(mixed_config):
    log_probs = torch.tensor([[-1.0, -1.0]])
    mask = torch.ones_like(log_probs)
    student = mixed_policy_grpo_loss(
        mixed_inputs(log_probs, mask, mask, torch.zeros_like(mask), log_probs), mixed_config
    )
    teacher = mixed_policy_grpo_loss(
        mixed_inputs(log_probs, mask, mask, torch.ones_like(mask), log_probs), mixed_config
    )
    # Distributed metric reduction stacks the same fields on every rank.
    assert student.metrics.keys() == teacher.metrics.keys()
    assert student.metrics["mixed/teacher_advantage_mean"] == 0
    assert teacher.metrics["mixed/teacher_advantage_mean"] == 1
