"""Group eligibility and sequence-loss contracts exercised through training math."""

import numpy as np

import pytest


import torch

from omegaconf import OmegaConf


from skyrl_train.objective.losses import TokenLoss

from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective

from skyrl_train.objective.reduction import step_counts

from skyrl_train.config.objective_spec import LossSpec, RatioAnchor

from skyrl_train.utils.advantage_estimators import (
    compute_advantages_and_returns,
)

from skyrl_train.group_admission import GroupAdvantageInvariant


from skyrl_train.utils.algorithm_registry import (
    PolicyLossRegistry,
)

from skyrl_train.utils.utils import validate_cfg

from tests.cpu.util import example_dummy_config


def _validatable_dummy_config():
    """A dummy config that passes validate_cfg (single-GPU placement, all batch sizes == 1)."""
    cfg = example_dummy_config()
    OmegaConf.update(
        cfg,
        "trainer",
        {
            "train_batch_size": 1,
            "policy_mini_batch_size": 1,
            "critic_mini_batch_size": 1,
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "placement": {
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 1,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 1,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": 1,
            },
        },
    )
    return cfg


@pytest.mark.parametrize(
    "loss_name,row_advantages,expected_gradient",
    [("custom_policy", [3.0, -2.0], [-1.5, 1.0]), ("gspo", [3.0, 3.0], [-1.5, -1.5])],
)
def test_validate_cfg_applies_sequence_level_loss_contract_to_training(loss_name, row_advantages, expected_gradient):
    def custom_policy_loss(inputs, config):
        return TokenLoss(-inputs.log_probs * inputs.advantages, {})

    PolicyLossRegistry.register(
        "custom_policy", custom_policy_loss, spec=LossSpec(RatioAnchor.NONE, sequence_level=True)
    )
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = loss_name
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    try:
        cfg.trainer.algorithm.loss_reduction = "token_mean"
        with pytest.raises(ValueError, match="requires trainer.algorithm.loss_reduction=sequence_mean"):
            validate_cfg(cfg)
        cfg.trainer.algorithm.loss_reduction = "sequence_mean"
        validate_cfg(cfg)
        cfg.trainer.algorithm.enable_token_reward_channel = True
        shaping = {"enable_token_reward_channel": True, "enable_pbs_shaping": True}
        OmegaConf.update(cfg, "terminal_bench_config", {"harbor": shaping}, force_add=True)
        with pytest.raises(ValueError, match="requires sequence-level advantages; use a token-level loss"):
            validate_cfg(cfg)
        OmegaConf.update(cfg, "terminal_bench_config.harbor.enable_pbs_shaping", False)
        validate_cfg(cfg)
        log_probs = torch.tensor([[-0.1, -0.5]], requires_grad=True)
        advantages = torch.tensor([row_advantages])
        mask = torch.ones_like(log_probs)
        batch = build_objective_micro_batch(
            action_log_probs=log_probs,
            old_action_log_probs=log_probs.detach(),
            base_action_log_probs=None,
            advantages=advantages,
            loss_mask=mask,
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(log_probs),
            think_token_weight=1,
            teacher=None,
        )
        counts = step_counts([mask], [mask], [], [advantages], 2, lambda value: value)
        result = compute_policy_objective(
            batch,
            loss=PolicyLossRegistry.get(loss_name),
            counts=counts,
            config=cfg.trainer.algorithm,
            loss_scale=1,
            report_scale=1,
        )
        result.optimization_loss.backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([expected_gradient]))
    finally:
        PolicyLossRegistry.unregister("custom_policy")


GROUP_ESTIMATORS = ["grpo", "rloo", "rloo_n"]


def _group_advantages(estimator, rewards, exclusions=None, *, minimum=2):
    invariant = (
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=len(rewards), minimum_group_size=minimum)
        if estimator == "rloo_n"
        else GroupAdvantageInvariant.exact_physical(physical_group_size=len(rewards))
    )
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    advantages, returns = compute_advantages_and_returns(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards),
        index=np.array(["g"] * len(rewards)),
        adv_estimator=estimator,
        config=OmegaConf.create({}),
        exclude_from_baseline=np.array(exclusions) if exclusions is not None else None,
        group_advantage_invariant=invariant,
    )
    torch.testing.assert_close(returns, advantages, rtol=0, atol=0)
    return advantages.squeeze(-1)


@pytest.mark.parametrize("estimator", GROUP_ESTIMATORS)
def test_group_estimators_exclude_masked_rows(estimator):
    expected = torch.cat([_group_advantages(estimator, [1.0, 0.0, 0.0]), torch.zeros(1)])
    actual = _group_advantages(estimator, [1.0, 0.0, 0.0, 0.0], [False, False, False, True])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    singleton = _group_advantages(estimator, [1.0, 0.0, 0.0, 0.0], [False, True, True, True])
    assert torch.equal(singleton, torch.zeros(4))


@pytest.mark.parametrize("estimator", GROUP_ESTIMATORS)
@pytest.mark.parametrize("near_flat", [False, True])
def test_group_estimators_zero_identical_and_nearly_flat_rewards(estimator, near_flat):
    rewards = [0.7] * 7
    if near_flat:
        rewards[-1] = float(np.nextafter(np.float32(0.7), np.float32(1.0)))
    assert torch.equal(_group_advantages(estimator, rewards), torch.zeros(7))


def test_rloo_n_zeroes_groups_below_its_configured_minimum():
    rewards, excluded = [1.0, 0.0, 0.0, 0.0], [False, False, False, True]
    assert torch.equal(_group_advantages("rloo_n", rewards, excluded, minimum=4), torch.zeros(4))
    torch.testing.assert_close(
        _group_advantages("rloo_n", rewards, excluded, minimum=3),
        torch.tensor([1.0, -0.5, -0.5, 0.0]),
        rtol=0,
        atol=1e-6,
    )
