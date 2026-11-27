"""Recipe-preserving transport tests against SkyRL's actual advantage/objective code."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.objective.losses import ppo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import policy_data_weights, step_counts
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.advantage_estimators import (
    apply_loop_advantages,
    compute_rloo_n_outcome_advantage,
    drop_advantage_inputs,
)
from skyrl_train.workers.levanter_batch import forward_payload, policy_payload


def algorithm(**overrides):
    return OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "use_kl_loss": False,
            "use_entropy_loss": False,
            "off_policy_correction": None,
            "think_token_weight": 0.5,
            "rloo_n_filter_zero_reward_groups": True,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "kl_loss_coef": 0.0,
        }
        | overrides
    )


def finalized_batch():
    # The high-reward infrastructure failure is neutral; a real agent failure
    # contributes reward zero. A second group has only one eligible member, a
    # third has constant nonzero rewards, and the final row is a DP-padding clone.
    attention = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]] * 4)
    response = attention[:, -3:].clone()
    rewards = torch.zeros(8, 3)
    rewards[:, 0] = torch.tensor([1.0, 0.0, 99.0, 0.5, 99.0, 0.5, 0.5, 1.0])
    ids = np.array(["mixed"] * 3 + ["singleton"] * 2 + ["constant"] * 2 + ["pad"])
    excluded = np.array([False, False, True, False, True, False, False, True])
    advantages, _ = compute_rloo_n_outcome_advantage(
        rewards,
        response,
        ids,
        exclude_from_baseline=excluded,
        group_advantage_invariant=GroupAdvantageInvariant.minimum_baseline_eligible(
            physical_group_size=3, minimum_group_size=2
        ),
        config=algorithm(),
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([1.0, -1.0, 0, 0, 0, 0, 0, 0]))
    loss_mask = response.float()
    loss_mask[[2, 4, 7]] = 0
    loss_mask[0, 1] = 0  # Tool observation: retained context, no learner objective.
    loss_mask[1, 0] = 0.25  # Fractional loss eligibility must retain its weight.
    loop = torch.zeros_like(advantages)
    loop[0, 2] = -0.1  # Existing reasoning-span penalty, after outcome credit.
    data = TrainingInputBatch(
        {
            "sequences": torch.arange(48).reshape(8, 6),
            "attention_mask": attention,
            "response_mask": response,
            "loss_mask": loss_mask,
            "response_span_tags": torch.tensor([[1, 0, 0]] * 8),
            "advantages": advantages,
            "action_log_probs": torch.full((8, 3), -2.0),
            "rewards": rewards,
            "loop_advantages": loop,
        }
    )
    data.metadata = {"uids": ids, "response_length": 3, "global_step": 11}
    return drop_advantage_inputs(apply_loop_advantages(data))


def test_preserves_finalized_credit_and_exact_scoring_coordinates():
    data = finalized_batch()
    payload = policy_payload(data, algorithm())
    arrays = payload.arrays
    expected_advantage = torch.where(data["loss_mask"] > 0, data["advantages"], 0)
    np.testing.assert_array_equal(arrays["advantages"], expected_advantage)
    np.testing.assert_array_equal(arrays["tokens"], data["sequences"])
    np.testing.assert_array_equal(arrays["attention_mask"], data["attention_mask"])
    np.testing.assert_array_equal(arrays["response_mask"], data["response_mask"])
    np.testing.assert_array_equal(arrays["position_ids"][0], [0, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(arrays["position_ids"][1], [0, 1, 2, 3, 4, 0])
    assert arrays["policy_weights"][0, 1] == 0
    assert arrays["response_mask"][0, 1]
    assert not arrays["policy_weights"][7].any()
    assert arrays["response_mask"][7].any()
    assert payload.global_step == 11
    assert payload.as_dict()["response_length"] == 3
    # Transport owns a frozen snapshot; neither scores nor tokens alias Torch.
    data["action_log_probs"].fill_(-5)
    data["sequences"].zero_()
    assert arrays["old_logprobs"][0, 0] == -2
    assert arrays["tokens"][0, 1] == 1


@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_complete_minibatch_weights_match_skyrl_loss_and_gradient(microbatches):
    data = finalized_batch()
    config = algorithm()
    payload = policy_payload(data, config)
    # Exercise both clipped signs, unclipped tokens, variable masks, and global
    # normalization with very different eligible-token counts in each slice.
    deltas = torch.linspace(-0.5, 0.5, 24).reshape(8, 3)
    logprobs = (data["action_log_probs"] + deltas).requires_grad_()
    weights = policy_data_weights(data["loss_mask"], data["response_span_tags"], config.think_token_weight)
    counts = step_counts(
        [weights],
        [data["loss_mask"]],
        [torch.zeros_like(weights)],
        [data["advantages"]],
        6,
        lambda value: value,
    )
    expected = logprobs.new_zeros(())
    actual = logprobs.new_zeros(())
    for indices in torch.arange(8).chunk(microbatches):
        objective = build_objective_micro_batch(
            action_log_probs=logprobs[indices],
            old_action_log_probs=data["action_log_probs"][indices],
            base_action_log_probs=None,
            advantages=data["advantages"][indices],
            loss_mask=data["loss_mask"][indices],
            rollout_logprobs=None,
            response_span_tags=data["response_span_tags"][indices],
            token_entropy=torch.zeros_like(logprobs[indices]),
            think_token_weight=config.think_token_weight,
            teacher=None,
        )
        expected = (
            expected
            + compute_policy_objective(
                objective, loss=ppo_policy_loss, counts=counts, config=config, loss_scale=1, report_scale=1
            ).optimization_loss
        )
        ratio = torch.exp(
            (logprobs[indices] - torch.from_numpy(payload.arrays["old_logprobs"])[indices]).clamp(-20, 20)
        )
        advantage = torch.from_numpy(payload.arrays["advantages"])[indices]
        values = torch.maximum(-ratio * advantage, -ratio.clamp(0.8, 1.2) * advantage)
        actual = actual + (values * torch.from_numpy(payload.arrays["policy_weights"])[indices]).sum()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    expected_grad = torch.autograd.grad(expected, logprobs, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, logprobs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)


def test_masked_nan_scores_and_credit_are_sanitized_without_mutating_input():
    data = finalized_batch()
    inactive = data["loss_mask"] == 0
    data["advantages"][inactive] = float("nan")
    data["action_log_probs"][inactive] = float("nan")
    arrays = policy_payload(data, algorithm()).arrays
    assert np.isfinite(arrays["advantages"]).all()
    assert np.isfinite(arrays["old_logprobs"]).all()
    assert torch.isnan(data["action_log_probs"][inactive]).all()


def test_all_masked_minibatch_has_zero_objective_weights_and_credit():
    data = finalized_batch()
    data["loss_mask"].zero_()
    arrays = policy_payload(data, algorithm()).arrays
    assert not arrays["policy_weights"].any()
    assert not arrays["kl_weights"].any()
    assert not arrays["advantages"].any()
    assert arrays["response_mask"].any()


def test_bfloat16_snapshots_convert_to_exact_float32_host_arrays():
    data = finalized_batch()
    data["advantages"] = data["advantages"].bfloat16()
    data["action_log_probs"] = data["action_log_probs"].bfloat16()
    arrays = policy_payload(data, algorithm()).arrays
    assert arrays["old_logprobs"].dtype == np.float32
    expected = torch.where(data["loss_mask"] > 0, data["advantages"].float(), 0)
    np.testing.assert_array_equal(arrays["advantages"], expected)


@pytest.mark.parametrize(
    "change",
    [
        {"policy_loss_type": "gspo"},
        {"loss_reduction": "sequence_mean"},
        {"use_kl_loss": True},
        {"use_entropy_loss": True},
        {"off_policy_correction": "tis"},
    ],
)
def test_rejects_unmatched_objective_terms(change):
    with pytest.raises(ValueError, match="requires"):
        policy_payload(finalized_batch(), algorithm(**change))


def test_rejects_unfinalized_credit_and_misaligned_responses():
    data = finalized_batch()
    data["rewards"] = torch.zeros(8, 3)
    with pytest.raises(ValueError, match="finalized"):
        policy_payload(data, algorithm())
    data.pop("rewards")
    data["response_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="response_mask"):
        forward_payload(data)
