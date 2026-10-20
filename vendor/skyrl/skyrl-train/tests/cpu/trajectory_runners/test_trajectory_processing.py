"""
uv run --group dev --extra cpu --isolated pytest tests/cpu/trajectory_runners/test_trajectory_processing.py
"""

import copy

import numpy as np
import pytest
from skyrl_gym.verification import VerificationResult

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.trajectory_runners.base import TrajectoryBatch, TrajectoryID
from skyrl_train.trajectory_runners.trajectory_processing import (
    apply_overlong_filtering,
    concatenate_trajectory_batches,
    get_batch_failure_metrics,
    get_metrics_from_trajectory_batch,
    get_rollout_metrics,
)


@pytest.mark.parametrize(
    "loss_masks,stop_reasons,expected_masks",
    [
        ([[1, 1, 0], [0, 1], [1]], ["stop", "eos", "end_turn"], [[1, 1, 0], [0, 1], [1]]),
        ([[1, 1], [0, 1]], ["length", "length"], [[0, 0], [0, 0]]),
        ([[1, 0], [1, 1], [0, 1]], ["stop", "length", "complete"], [[1, 0], [0, 0], [0, 1]]),
        ([[], [0, 0]], ["length", "length"], [[], [0, 0]]),
        ([], [], []),
        ([[1, 1], [1, 0], [0, 1]], [None, "stop", "length"], [[1, 1], [1, 0], [0, 0]]),
    ],
    ids=["completed", "length", "mixed", "empty-masks", "empty-batch", "unknown-reason"],
)
def test_apply_overlong_filtering_zeros_masks_of_truncated_responses(loss_masks, stop_reasons, expected_masks):
    original_loss_masks = copy.deepcopy(loss_masks)
    original_stop_reasons = list(stop_reasons)
    assert apply_overlong_filtering(loss_masks, stop_reasons) == expected_masks
    assert loss_masks == original_loss_masks
    assert stop_reasons == original_stop_reasons


@pytest.mark.parametrize(
    "num_trials,num_failed,expected_fraction",
    [
        (10, 4, 0.4),
        (64, 0, 0.0),  # a healthy batch still reports the series rather than omitting it
        (64, 64, 1.0),  # the orchestrator-failure path, where every trial is reported failed
        (0, 0, 0.0),  # an empty batch has no denominator to divide by
    ],
)
def test_failed_fraction_is_measured_against_the_whole_batch(num_trials, num_failed, expected_fraction):
    metrics = get_batch_failure_metrics(
        num_trials,
        num_failed_trajectories=num_failed,
        num_failed_instances=num_failed,
        num_masked_trajectories=num_failed,
    )

    assert metrics["generate/failed_trajectory_fraction"] == pytest.approx(expected_fraction)


def test_failure_counts_stay_separate_from_the_fraction():
    # Trajectories fail individually, instances group n_samples_per_prompt of
    # them, and only the infrastructure failures are masked out of the RLOO
    # baseline, so the three counts move independently and none substitutes
    # for another.
    metrics = get_batch_failure_metrics(
        64,
        num_failed_trajectories=40,
        num_failed_instances=8,
        num_masked_trajectories=24,
    )

    assert metrics == {
        "generate/num_trials": 64,
        "generate/num_failed_instances": 8,
        "generate/num_failed_trajectories": 40,
        "generate/num_masked_trajectories": 24,
        "generate/failed_trajectory_fraction": pytest.approx(0.625),
    }


def _generated_group(num_trials: int, num_failed: int, error_type: str = "SandboxError") -> dict:
    failure_metrics = get_batch_failure_metrics(
        num_trials,
        num_failed_trajectories=num_failed,
        num_failed_instances=num_failed,
        num_masked_trajectories=num_failed,
    )
    if num_failed:
        failure_metrics[f"generate/errors/{error_type}"] = num_failed
    return {
        "prompt_token_ids": [[1] for _ in range(num_trials)],
        "response_ids": [[2, 3] for _ in range(num_trials)],
        "rewards": [0.0 for _ in range(num_trials)],
        "loss_masks": [[1, 1] for _ in range(num_trials)],
        "stop_reasons": ["error" if i < num_failed else "stop" for i in range(num_trials)],
        "rollout_logprobs": None,
        "rollout_metrics": failure_metrics,
    }


def test_failure_metrics_survive_concatenation():
    # The fully asynchronous trainer generates one rollout group at a time and reads
    # rollout metrics off the concatenated result only, so a count that does not
    # survive this merge never reaches the tracker. The groups are deliberately
    # unequal: the fraction has to come from the summed totals, not from averaging
    # 0.75 and 0.5 to 0.625.
    merged = concatenate_trajectory_batches(
        [_generated_group(4, 3), _generated_group(2, 1, error_type="ContextLengthExceededError")],
        tis_lcs_alert_threshold=0.005,
    )

    assert merged["rollout_metrics"]["generate/num_trials"] == 6
    assert merged["rollout_metrics"]["generate/num_failed_trajectories"] == 4
    assert merged["rollout_metrics"]["generate/failed_trajectory_fraction"] == pytest.approx(4 / 6)
    assert merged["rollout_metrics"]["generate/errors/SandboxError"] == 3
    assert merged["rollout_metrics"]["generate/errors/ContextLengthExceededError"] == 1


def test_data_sources_survive_async_batch_concatenation():
    first = _generated_group(1, 0)
    first["data_sources"] = ["math"]
    second = _generated_group(2, 0)
    second["data_sources"] = ["tools", None]

    merged = concatenate_trajectory_batches([first, second], tis_lcs_alert_threshold=0.005)

    assert merged["data_sources"] == ["math", "tools", None]


def test_server_error_identity_stays_with_its_row_after_concatenation():
    failed = _generated_group(1, 1)
    failed["server_errors"] = [{"category": "constrained_decoding", "request_id": "request-123", "status_code": 500}]
    merged = concatenate_trajectory_batches([failed, _generated_group(1, 0)], tis_lcs_alert_threshold=0.005)

    assert merged["server_errors"] == [
        {"category": "constrained_decoding", "request_id": "request-123", "status_code": 500},
        None,
    ]


def test_unaligned_logprob_alert_survives_concatenation():
    groups = [_generated_group(1, 0), _generated_group(1, 0)]
    groups[0]["rollout_metrics"].update({"generate/tis/aligned_tokens": 10.0, "generate/tis/exact_match_fraction": 1.0})
    groups[1]["rollout_metrics"].update(
        {
            "generate/tis/aligned_tokens": 10.0,
            "generate/tis/exact_match_fraction": 0.9,
            "generate/tis/unaligned_fraction": 0.1,
            "generate/tis/alignment_fail_count": 1.0,
        }
    )

    merged = concatenate_trajectory_batches(groups, tis_lcs_alert_threshold=0.005)

    assert merged["rollout_metrics"]["generate/tis/unaligned_fraction"] == pytest.approx(0.05)
    assert merged["rollout_metrics"]["generate/tis/lcs_fallback_alert"] == 0.0
    assert merged["rollout_metrics"]["generate/tis/alignment_alert"] == 1.0


def test_full_tito_and_task_rollout_metrics_survive_concatenation():
    groups = [_generated_group(1, 0), _generated_group(1, 0)]
    groups[0]["rollout_metrics"].update(
        {
            "generate/tis/aligned_tokens": 5.0,
            "generate/tis/exact_match_fraction": 1.0,
            "generate/tis/tito_full/attempts": 1.0,
            "generate/tis/tito_full/success_fraction": 1.0,
        }
    )
    groups[1]["rollout_metrics"].update(
        {
            "generate/tis/aligned_tokens": 5.0,
            "generate/tis/exact_match_fraction": 1.0,
            "generate/tis/tito_full/attempts": 1.0,
            "generate/tis/tito_full/success_fraction": 0.0,
            "generate/tis/tito_full/decline_count": 1.0,
            "generate/tis/tito_full/decline/prefix_mismatch": 1.0,
        }
    )
    groups[0]["rollout_metrics"].update(
        {
            "generate/task_rollout/tasks": 1.0,
            "generate/task_rollout/turns": 3.0,
        }
    )
    groups[1]["rollout_metrics"].update(
        {
            "generate/task_rollout/tasks": 1.0,
            "generate/task_rollout/turns": 2.0,
        }
    )

    merged = concatenate_trajectory_batches(groups, tis_lcs_alert_threshold=0.005)

    assert merged["rollout_metrics"]["generate/tis/tito_full/attempts"] == 2.0
    assert merged["rollout_metrics"]["generate/tis/tito_full/success_fraction"] == 0.5
    assert merged["rollout_metrics"]["generate/tis/tito_full/decline_count"] == 1.0
    assert merged["rollout_metrics"]["generate/tis/tito_full/decline/prefix_mismatch"] == 1.0
    assert merged["rollout_metrics"]["generate/tis/alignment_alert"] == 1.0
    assert merged["rollout_metrics"]["generate/task_rollout/tasks"] == 2.0
    assert merged["rollout_metrics"]["generate/task_rollout/turns"] == 5.0


def test_identity_aware_reward_metrics_survive_concatenation():
    groups = [_generated_group(2, 0), _generated_group(2, 0)]
    groups[0]["rollout_metrics"]["generate/reward_shaping/identity_aware/groups"] = 1
    groups[1]["rollout_metrics"]["generate/reward_shaping/identity_aware/groups"] = 1
    groups[1]["rollout_metrics"]["generate/reward_shaping/identity_aware/fallback_groups"] = 1

    merged = concatenate_trajectory_batches(groups, tis_lcs_alert_threshold=0.005)

    assert merged["rollout_metrics"]["generate/reward_shaping/identity_aware/groups"] == 2
    assert merged["rollout_metrics"]["generate/reward_shaping/identity_aware/fallback_groups"] == 1


def test_generators_that_report_no_trials_get_no_failure_series():
    # Only the agentic generator counts trials, and a fabricated zero series for
    # every other generator would read as a measured "nothing failed".
    merged = concatenate_trajectory_batches(
        [{**_generated_group(2, 0), "rollout_metrics": {}}], tis_lcs_alert_threshold=0.005
    )

    assert "generate/num_trials" not in merged["rollout_metrics"]
    assert "generate/failed_trajectory_fraction" not in merged["rollout_metrics"]


@pytest.mark.parametrize("timeout_first", [False, True])
def test_concatenation_preserves_unshaped_rewards_with_incomplete_timeout_output(timeout_first):
    normal = {**_generated_group(1, 0), "rewards": [1.0], "unshaped_rewards": [1.0]}
    timeout = _generated_group(1, 1, error_type="AgentTimeoutError")
    groups = [timeout, normal] if timeout_first else [normal, timeout]

    merged = concatenate_trajectory_batches(groups, tis_lcs_alert_threshold=0.005)

    assert merged["unshaped_rewards"] == ([0.0, 1.0] if timeout_first else [1.0, 0.0])
    assert merged["unshaped_reward_available"] == ([False, True] if timeout_first else [True, False])


def test_partial_failure_metrics_are_rejected_during_concatenation():
    group = _generated_group(2, 1)
    del group["rollout_metrics"]["generate/num_masked_trajectories"]

    with pytest.raises(ValueError, match="generate/num_masked_trajectories"):
        concatenate_trajectory_batches([group], tis_lcs_alert_threshold=0.005)


def test_required_rollout_logprobs_reject_partial_generation_batch():
    with_logprobs = {**_generated_group(2, 0), "rollout_logprobs": [[-0.1, -0.2], [-0.3, -0.4]]}

    with pytest.raises(ValueError, match="rollout_logprobs are required"):
        concatenate_trajectory_batches(
            [with_logprobs, _generated_group(2, 0)],
            require_rollout_logprobs=True,
            tis_lcs_alert_threshold=0.005,
        )


def test_required_rollout_logprobs_allow_fully_excluded_generation_batch():
    trainable = {**_generated_group(2, 0), "rollout_logprobs": [[-0.1, -0.2], [-0.3, -0.4]]}
    excluded = {
        **_generated_group(2, 2),
        "loss_masks": [[0, 0], [0, 0]],
        "exclude_from_baseline": [True, True],
    }

    merged = concatenate_trajectory_batches(
        [trainable, excluded],
        require_rollout_logprobs=True,
        tis_lcs_alert_threshold=0.005,
    )

    assert merged["loss_masks"] == [[1, 1], [1, 1], [0, 0], [0, 0]]
    assert merged["exclude_from_baseline"] == [False, False, True, True]


def test_required_rollout_logprobs_reject_fully_masked_baseline_contributor():
    masked_baseline_contributor = {
        **_generated_group(2, 0),
        "loss_masks": [[0, 0], [0, 0]],
        "exclude_from_baseline": [False, False],
    }

    with pytest.raises(ValueError, match="rollout_logprobs are required"):
        concatenate_trajectory_batches(
            [masked_baseline_contributor],
            require_rollout_logprobs=True,
            tis_lcs_alert_threshold=0.005,
        )


def test_trajectory_batch_concatenation():
    trajectory_batch_1: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "unshaped_rewards": [0.0, 1.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.1, 0.2], [0.3, 0.4]],
        "trajectory_ids": [TrajectoryID("first", 0), TrajectoryID("second", 0)],
        "is_last_step": [True, False],
        "exclude_from_baseline": [False, True],
    }

    trajectory_batch_2: TrajectoryBatch = {
        "prompt_token_ids": [[5, 6, 7], [8]],
        "response_ids": [[5, 6, 7], [8]],
        "rewards": [2.0, 3.0],
        "unshaped_rewards": [1.0, 0.0],
        "loss_masks": [[1, 1, 1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.5, 0.6, 0.7], [0.8]],
        "trajectory_ids": [TrajectoryID("third", 0), TrajectoryID("fourth", 0)],
        "is_last_step": [False, True],
        "exclude_from_baseline": [True, False],
    }

    trajectory_batches = [trajectory_batch_1, trajectory_batch_2]
    concatenated_output = concatenate_trajectory_batches(trajectory_batches, tis_lcs_alert_threshold=0.005)

    assert concatenated_output["prompt_token_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["response_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["rewards"] == [1.0, 2.0, 2.0, 3.0]
    assert concatenated_output["unshaped_rewards"] == [0.0, 1.0, 1.0, 0.0]
    assert concatenated_output["loss_masks"] == [[1, 1], [1, 1], [1, 1, 1], [1]]
    assert concatenated_output["stop_reasons"] == ["stop", "stop", "stop", "stop"]
    assert concatenated_output["rollout_logprobs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7], [0.8]]
    assert [trajectory.instance_id for trajectory in concatenated_output["trajectory_ids"]] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    assert concatenated_output["is_last_step"] == [True, False, False, True]
    assert concatenated_output["exclude_from_baseline"] == [False, True, True, False]

    expected_rollout_metrics = {
        "generate/min_num_tokens": 1,
        "generate/max_num_tokens": 3,
        "generate/avg_num_tokens": 2.0,
        "generate/std_num_tokens": np.std([2, 2, 3, 1]).item(),
        "generate/avg_tokens_non_zero_rewards": 2.0,
        "generate/avg_tokens_zero_rewards": 0,
    }
    assert concatenated_output["rollout_metrics"].keys() == expected_rollout_metrics.keys()
    for key, value in expected_rollout_metrics.items():
        np.testing.assert_allclose(concatenated_output["rollout_metrics"][key], value)


def test_rollout_metrics_include_negative_reward_failures():
    metrics = get_rollout_metrics(
        responses=[[1, 2], list(range(9)), [3, 4, 5], list(range(11))],
        rewards=[1.0, -1.0, 1.0, -1.0],
        verification_results=[
            VerificationResult.verified(float(passed), passed=passed) for passed in [True, False, True, False]
        ],
    )

    assert metrics["generate/avg_tokens_non_zero_rewards"] == pytest.approx(2.5)
    assert metrics["generate/avg_tokens_zero_rewards"] == pytest.approx(10.0)


def test_get_metrics_from_trajectory_batch():
    # Per trajectory rewards, where rewards are List[float]
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    uids = ["a", "b"]
    avg_score, pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch, uids)
    assert avg_score == 1.5
    assert pass_at_n == 1.0

    # Per token rewards, where rewards are List[List[float]], so for pass_at_n we use the last
    # token's reward to signify the trajectory's reward
    trajectory_batch["rewards"] = [[1.0, 0.0], [0.0, 1.0]]
    uids = ["a", "b"]
    avg_score, pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch, uids)
    assert avg_score == 1.0
    assert pass_at_n == 0.5

    trajectory_batch["response_ids"] = [[], [3, 4]]
    trajectory_batch["rewards"] = [[], [0.0, 1.0]]
    avg_score, pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch, uids)
    assert avg_score == 0.5
    assert pass_at_n == 0.5


def test_pass_at_n_uses_unshaped_outcomes():
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [0.2, 0.3, 0.4, 0.5],
        "unshaped_rewards": [0.0, 1.0, 0.0, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop", "stop"],
        "rollout_logprobs": None,
    }

    first_avg_score, first_pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch, ["a", "a", "b", "b"])

    trajectory_batch["rewards"] = [-1.0, 0.0, 4.0, 5.0]
    second_avg_score, second_pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch, ["a", "a", "b", "b"])

    assert first_avg_score == pytest.approx(0.35)
    assert second_avg_score == pytest.approx(2.0)
    assert first_pass_at_n == second_pass_at_n == 0.5


def test_pass_at_n_honors_partial_credit_and_failed_verifiers():
    batch: TrajectoryBatch = {
        "rewards": [0.001, 0.3, 1.0, 1.0, 0.7, 0.8],
        "verification_results": [
            VerificationResult.verified(0.001, passed=False),
            VerificationResult.verified(0.3, passed=False),
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.error("judge unavailable"),
            VerificationResult.verified(0.7),
            None,
        ],
    }
    mean_reward, pass_at_n = get_metrics_from_trajectory_batch(
        batch, ["partial", "partial", "full", "error", "numeric", "harbor"]
    )

    assert mean_reward == pytest.approx(3.801 / 6)
    assert pass_at_n == pytest.approx(3 / 5)


def test_rollout_metrics_skip_unstepped_episode_metrics():
    """Episodes skipped before env.step report empty metrics and must not reach aggregators."""
    metrics = get_rollout_metrics(
        responses=[[1, 2], [3, 4, 5]],
        rewards=[1.0, 0.0],
        env_classes=["aime", "aime"],
        env_metrics=[{"acc": True, "over_evaluation_budget": False, "answered_within_evaluation_budget": True}, {}],
    )

    assert metrics["environment/acc"] == 1.0


def test_environment_rates_use_per_n_contributors_after_filtering_and_repeated_concat():
    n10 = _generated_group(3, 0)
    n10["env_metrics"] = [{"exact_n10": 1.0}, {"exact_n10": 0.0}, {"exact_n10": 1.0}]
    n10["env_classes"] = ["cat_count"] * 3
    n20 = _generated_group(2, 0)
    n20["env_metrics"] = [{"exact_n20": 1.0}, {"exact_n20": 0.0}]
    n20["env_classes"] = ["cat_count"] * 2
    n10_extra = _generated_group(1, 0)
    n10_extra["env_metrics"] = [{"exact_n10": 0.0}]
    n10_extra["env_classes"] = ["cat_count"]

    first = concatenate_trajectory_batches([n10, n20], tis_lcs_alert_threshold=0.005)
    merged = concatenate_trajectory_batches([first, n10_extra], tis_lcs_alert_threshold=0.005)

    assert merged["rollout_metrics"]["environment/exact_n10"] == pytest.approx(0.5)
    assert merged["rollout_metrics"]["environment/exact_n20"] == pytest.approx(0.5)
    filtered = filter_trajectory_batch(merged, [0, 2, 3, 4])
    assert filtered["rollout_metrics"]["environment/exact_n10"] == pytest.approx(1.0)
    assert filtered["rollout_metrics"]["environment/exact_n20"] == pytest.approx(0.5)
