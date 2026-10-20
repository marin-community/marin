"""RayPPOTrainer.postprocess_trajectory_batch reward shaping and reward metrics."""

import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationResult

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.batch_metrics import domain_reward_metrics
from skyrl_train.trajectory_runners.types import TrajectoryBatch
from skyrl_train.rollouts.finalization import propagate_data_sources
from skyrl_train.trajectory_runners.types import TrajectoryID, TaskRequestBatch
from tests.cpu.task_specs import task_input, lowered_task, text_task


def make_trainer(n_samples_per_prompt: int = 1) -> RayPPOTrainer:
    """Postprocessing reads only config and metrics; skip __init__, which starts a Ray trajectory sink."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {"trainer": {"step_wise_training": False}, "generator": {"n_samples_per_prompt": n_samples_per_prompt}}
    )
    trainer.all_metrics = {}
    return trainer


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        # Response-level rewards are credited to the final response token.
        ([1.0, 0.5], [[0.0, 1.0], [0.0, 0.0, 0.5]]),
        # Token-level rewards pass through unchanged.
        ([[0.1, 0.3], [0.2, 0.1, 0.1]], [[0.1, 0.3], [0.2, 0.1, 0.1]]),
    ],
)
def test_postprocess_produces_per_token_rewards(rewards, expected):
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": rewards,
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = make_trainer().postprocess_trajectory_batch(trajectory_batch, ["uid1", "uid2"])

    assert result["rewards"] == expected


def test_pass_at_k_uses_unshaped_outcomes():
    trainer = make_trainer(n_samples_per_prompt=2)
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [0.2, 0.3, 0.4, 0.5],
        "unshaped_rewards": [0.0, 1.0, 0.0, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop", "stop"],
        "rollout_metrics": None,
    }

    trainer.postprocess_trajectory_batch(trajectory_batch, ["a", "a", "b", "b"])

    assert trainer.all_metrics["reward/avg_pass_at_2"] == 0.5
    assert trainer.all_metrics["reward/avg_raw_reward"] == pytest.approx(0.35)


def test_training_reports_normalized_composite_scores_by_agent():
    trainer = make_trainer()
    batch: TrajectoryBatch = {
        "prompt_token_ids": [[0], [0], [0]],
        "loss_masks": [[1], [1], [1]],
        "response_ids": [[1], [2], [3]],
        "rewards": [5.0, 1.0, 0.0],
        "verification_results": [
            VerificationResult.verified(5.0, diagnostics={"agent": "genrm"}, score_min=1.0, score_max=5.0),
            VerificationResult.verified(1.0, diagnostics={"agent": "mcqa"}),
            VerificationResult.error("judge unavailable", diagnostics={"agent": "mcqa"}),
        ],
    }

    trainer.postprocess_trajectory_batch(batch, ["a", "b", "c"])

    assert trainer.all_metrics["reward/avg_raw_reward"] == 2.0
    assert trainer.all_metrics["reward/avg_verifier_score"] == 1.0
    assert trainer.all_metrics["reward/agent/genrm/avg_verifier_score"] == 1.0
    assert trainer.all_metrics["reward/agent/mcqa/avg_verifier_score"] == 1.0


def test_informative_group_fraction_counts_groups_whose_rewards_differ():
    trainer = make_trainer()
    # Group a has reward spread; group b is a tie and carries no advantage signal.
    trainer.postprocess_trajectory_batch(
        {
            "prompt_token_ids": [[0]] * 4,
            "loss_masks": [[1]] * 4,
            "response_ids": [[1], [2], [3], [4]],
            "rewards": [1.0, 0.0, 0.5, 0.5],
        },
        ["a", "a", "b", "b"],
    )
    assert trainer.all_metrics["reward/informative_group_fraction"] == 0.5


def test_domain_reward_metrics_aggregate_and_bound_metric_keys():
    metrics = domain_reward_metrics(["alpha", "alpha", None, "zeta"], [0.0, 1.0, 0.5, 0.0])
    assert metrics == {
        "reward/domain/alpha/avg_raw_reward": 0.5,
        "reward/domain/_missing/avg_raw_reward": 0.5,
        "reward/domain/zeta/avg_raw_reward": 0.0,
    }

    overflow_metrics = domain_reward_metrics(["__other__", *[f"domain-{i:02}" for i in range(40)]], [0.0] + [1.0] * 40)
    assert len(overflow_metrics) == 33
    assert overflow_metrics["reward/domain/__other__/avg_raw_reward"] == 0.0
    assert overflow_metrics["reward/domain_overflow/avg_raw_reward"] == 1.0


def test_domain_reward_metric_names_do_not_merge_distinct_sources():
    metrics = domain_reward_metrics(
        ["a/b", "a_b", None, "unknown", "_missing", "_source_a_2fb", "Math", "math"],
        [0.0, 1.0, 0.25, 0.75, 0.5, 0.6, 0.3, 0.9],
    )

    assert metrics == {
        "reward/domain/_source_a_2fb/avg_raw_reward": 0.0,
        "reward/domain/a_b/avg_raw_reward": 1.0,
        "reward/domain/_missing/avg_raw_reward": 0.25,
        "reward/domain/unknown/avg_raw_reward": 0.75,
        "reward/domain/_source__5fmissing/avg_raw_reward": 0.5,
        "reward/domain/_source__5fsource_5fa_5f2fb/avg_raw_reward": 0.6,
        "reward/domain/_source__4dath/avg_raw_reward": 0.3,
        "reward/domain/math/avg_raw_reward": 0.9,
    }


def test_step_wise_rollout_rows_keep_request_data_sources():
    request = TaskRequestBatch(
        rows=tuple(
            task_input(lowered_task(text_task(uid)), data_source=source)
            for uid, source in (("a", "math"), ("b", "tools"))
        ),
        trajectory_ids=(TrajectoryID("a", 0), TrajectoryID("b", 0)),
    )
    output = {
        "response_ids": [[1], [2], [3]],
        "trajectory_ids": [TrajectoryID("a", 0), TrajectoryID("a", 0), TrajectoryID("b", 0)],
    }

    propagate_data_sources(request, output)

    assert output["data_sources"] == ["math", "math", "tools"]

    output_with_empty_first_trajectory = {
        "response_ids": [[1], [2]],
        "trajectory_ids": [TrajectoryID("b", 0), TrajectoryID("b", 0)],
    }
    propagate_data_sources(request, output_with_empty_first_trajectory)
    assert output_with_empty_first_trajectory["data_sources"] == ["tools", "tools"]


def test_reward_metrics_leave_out_skipped_rollouts():
    trainer = make_trainer(n_samples_per_prompt=2)
    skipped = VerificationResult.skipped("grading is skipped")
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [1.0, 0.0, 0.0, 0.0],
        "verification_results": [
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.verified(0.0, passed=False),
            skipped,
            skipped,
        ],
        "data_sources": ["lean", "lean", "ultra", "ultra"],
        "loss_masks": [[1], [1], [1], [1]],
        "rollout_metrics": None,
    }

    trainer.postprocess_trajectory_batch(trajectory_batch, ["a", "a", "b", "b"])

    assert trainer.all_metrics["reward/avg_raw_reward"] == 0.5
    assert trainer.all_metrics["reward/avg_pass_at_2"] == 1.0
    assert trainer.all_metrics["reward/informative_group_fraction"] == 1.0
    assert trainer.all_metrics["reward/domain/lean/avg_raw_reward"] == 0.5
    assert "reward/domain/ultra/avg_raw_reward" not in trainer.all_metrics


def test_all_skipped_rollouts_record_no_reward_metrics():
    trainer = make_trainer()
    skipped = VerificationResult.skipped("grading is skipped")
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3], [4]],
        "rewards": [0.0, 0.0],
        "verification_results": [skipped, skipped],
        "loss_masks": [[1], [1]],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_trajectory_batch(trajectory_batch, ["a", "b"])

    assert not any(key.startswith("reward/") for key in trainer.all_metrics)
    assert result["rewards"] == [[0.0], [0.0]]
