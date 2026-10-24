from dataclasses import dataclass

import numpy as np
import pytest

from skyrl_train.dynamic_sampling import (
    DynamicSamplingType,
    GroupSelectionPolicy,
    GroupSelectionResult,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.group_admission import (
    AdmissionDecision,
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
    admission_stall_timeout,
    assert_training_groups_eligible,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards


@dataclass
class _Group:
    trajectory_batch: dict


def _group(
    *,
    loss_masks: list[list[int]],
    exclude_from_baseline: list[bool] | None = None,
    rollout_logprobs: list[np.ndarray] | None = None,
) -> _Group:
    group_size = len(loss_masks)
    trajectory_batch = {
        "prompt_token_ids": [[1] for _ in range(group_size)],
        "response_ids": [[2] for _ in range(group_size)],
        "rewards": [0.0 for _ in range(group_size)],
        "loss_masks": loss_masks,
        "stop_reasons": ["stop" for _ in range(group_size)],
        "rollout_metrics": {},
        "rollout_logprobs": rollout_logprobs,
        "exclude_from_baseline": exclude_from_baseline,
    }
    return _Group(trajectory_batch)


@pytest.mark.parametrize(
    ("history", "override", "expected"),
    [
        ([], None, 1800.0),
        ([100.0, 200.0, 300.0], None, 1000.0),
        ([1.0, 2.0, 3.0], None, 600.0),
        pytest.param([100.0, 200.0, 300.0], 20.0, 20.0, id="override-ignores-step-times"),
    ],
)
def test_admission_stall_timeout_scales_with_recent_step_times(history, override, expected):
    assert admission_stall_timeout(recent_step_times=history, timeout_override=override) == expected


@pytest.mark.parametrize(
    ("rejections", "fatal"),
    [
        ((), False),
        ((AdmissionRejection.FULLY_MASKED,), False),
        ((AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE,), False),
        ((AdmissionRejection.PHYSICAL_GROUP_SIZE,), True),
        ((AdmissionRejection.MISSING_ROLLOUT_LOGPROBS,), True),
    ],
)
def test_only_structural_rejections_fail_training(rejections, fatal):
    assert AdmissionDecision(rejections).fatal is fatal


def test_exact_group_accepts_masked_trial_when_group_can_train():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=3),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0], [1]]))

    assert decision.accepted


def test_fully_masked_group_is_rejected_independently_of_estimator():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=2),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[0], [0]]))

    assert decision.primary_rejection is AdmissionRejection.FULLY_MASKED


def test_non_group_estimator_bypasses_physical_cardinality():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=4),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1]]))

    assert decision.accepted


def test_minimum_group_counts_baseline_members_independently_of_loss_mask():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(
        _group(loss_masks=[[1], [0], [0]], exclude_from_baseline=[False, False, True]),
    )

    assert decision.accepted


def test_minimum_group_treats_missing_exclusions_as_all_baseline_eligible():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=2, minimum_group_size=2),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0]], exclude_from_baseline=None))

    assert decision.accepted


def test_minimum_group_rejects_cohort_below_floor():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
        rollout_logprobs_required=False,
    )

    decision = policy.evaluate(
        _group(loss_masks=[[1], [1], [0]], exclude_from_baseline=[False, True, True]),
    )

    assert decision.primary_rejection is AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE


def test_required_logprobs_reject_missing_values_only_for_trainable_group():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        rollout_logprobs_required=True,
    )

    decision = policy.evaluate(_group(loss_masks=[[1], [0]], rollout_logprobs=None))

    assert decision.primary_rejection is AdmissionRejection.MISSING_ROLLOUT_LOGPROBS


def test_required_logprobs_reject_non_array_rows():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        rollout_logprobs_required=True,
    )
    group = _group(loss_masks=[[1], [0]])
    group.trajectory_batch["rollout_logprobs"] = [[-0.5], [0.0]]

    with pytest.raises(ValueError, match="rollout_logprobs row 0 must be a one-dimensional array"):
        policy.evaluate(group)


@pytest.mark.parametrize(("trained_logprob", "accepted"), [(-0.5, True), (0.0, True), (0.5, False)])
def test_required_logprobs_validate_numpy_rows(trained_logprob, accepted):
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        rollout_logprobs_required=True,
    )
    group = _group(
        loss_masks=[[1], [0]],
        rollout_logprobs=[np.asarray([trained_logprob], dtype=np.float32), np.asarray([0.5], dtype=np.float32)],
    )

    decision = policy.evaluate(group)
    assert decision.accepted is accepted
    if not accepted:
        assert decision.primary_rejection is AdmissionRejection.MISSING_ROLLOUT_LOGPROBS
        assert decision.fatal


def test_malformed_group_fails_instead_of_being_retried():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        rollout_logprobs_required=False,
    )
    group = _group(loss_masks=[[1], [1]])
    group.trajectory_batch["loss_masks"] = [[1]]

    with pytest.raises(ValueError, match="loss_masks"):
        policy.evaluate(group)


def test_training_group_error_reports_observed_and_expected_physical_counts():
    group = _group(loss_masks=[[1], [1], [1], [1]])

    with pytest.raises(TrainingGroupInvariantError) as exc_info:
        assert_training_groups_eligible(
            group.trajectory_batch,
            ["duplicate"] * 4,
            GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        )

    assert exc_info.value.uid == "duplicate"
    assert exc_info.value.rejections == (AdmissionRejection.PHYSICAL_GROUP_SIZE,)
    assert exc_info.value.physical_count == 4
    assert exc_info.value.expected_physical_count == 2
    assert exc_info.value.row_indices == (0, 1, 2, 3)


def test_stepwise_group_counts_final_trials_but_checks_all_transitions_for_training():
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        rollout_logprobs_required=False,
    )
    group = _group(loss_masks=[[1], [0], [1], [0]])
    group.trajectory_batch["is_last_step"] = [False, True, False, True]

    decision = policy.evaluate(group)

    assert decision.accepted


def test_dynamic_filter_uses_final_unshaped_outcomes():
    policy = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("unshaped"))
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [0.0, 0.0, 0.0, -0.1],
            "unshaped_rewards": [0.0, 1.0, 0.0, 1.0],
            "is_last_step": [False, True, False, True],
        }
    )

    decision = policy.evaluate(group)

    assert decision is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_dynamic_filter_requires_unshaped_outcomes():
    policy = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("unshaped"))

    with pytest.raises(ValueError, match="requires unshaped_rewards"):
        policy.evaluate(_group(loss_masks=[[1], [1]]))


def test_dynamic_filter_can_admit_shaped_reward_variance():
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [0.0, 0.25, 0.5, 0.0],
            "unshaped_rewards": [0.0, 0.0, 0.0, 0.0],
        }
    )

    shaped = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("shaped"))
    unshaped = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("unshaped"))

    assert shaped.evaluate(group) is GroupSelectionResult.KEEP
    assert unshaped.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_unshaped_dynamic_filter_ignores_passthrough_optimization_penalty():
    group = _group(loss_masks=[[1], [1]])
    group.trajectory_batch.update(
        {
            "rewards": [1.0, 1.0],
            "unshaped_rewards": [1.0, 1.0],
            "stop_reasons": ["turn_cap_exhausted", "complete"],
            "exception_types": ["TurnCapExhaustedError", None],
            "error_treatments": ["passthrough", None],
        }
    )
    shape_trajectory_rewards(
        group.trajectory_batch,
        {"enabled": True, "passthrough": {"penalty": 0.25}},
    )

    shaped = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("shaped"))
    unshaped = GroupSelectionPolicy(DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("unshaped"))

    assert group.trajectory_batch["rewards"] == pytest.approx([0.75, 1.0])
    assert shaped.evaluate(group) is GroupSelectionResult.KEEP
    assert unshaped.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


def test_dynamic_filter_applies_minimum_reward_std():
    group = _group(loss_masks=[[1], [1], [1], [1]])
    group.trajectory_batch["rewards"] = [0.05, 0.05, 0.10, 0.05]

    default_floor = GroupSelectionPolicy(DynamicSamplingType.FILTER)
    high_floor = GroupSelectionPolicy(
        DynamicSamplingType.FILTER, criteria=resolve_dynamic_sampling_criteria("shaped", 0.1)
    )

    assert default_floor.evaluate(group) is GroupSelectionResult.KEEP
    assert high_floor.evaluate(group) is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([1.0, 1.0, 1.0, 0.0], GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD),
        ([1.0, 0.0, 1.0, 1.0], GroupSelectionResult.KEEP),
    ],
)
def test_dynamic_filter_reads_only_rows_in_the_group_baseline(rewards, expected):
    group = _group(loss_masks=[[1], [1], [1], [0]], exclude_from_baseline=[False, False, False, True])
    group.trajectory_batch["rewards"] = rewards
    group.trajectory_batch["unshaped_rewards"] = rewards
    selection = GroupSelectionPolicy(DynamicSamplingType.FILTER)
    assert selection.evaluate(group) is expected
