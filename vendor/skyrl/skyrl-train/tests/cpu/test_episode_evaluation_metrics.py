from dataclasses import replace

import pytest
from rolloutengine.models import ConversationRecord, ModelTurn, RolloutStep, Transition
from skyrl_gym.verification import VerificationResult
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.rollouts.task_projections import episode_statistics
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.utils.trainer_utils import calculate_per_dataset_metrics, cleanup_old_checkpoints


def test_checkpoint_retention_keeps_latest_three_and_selected_best(tmp_path):
    for step in range(1, 7):
        checkpoint = tmp_path / f"global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / "weights").write_text(str(step))
    cleanup_old_checkpoints(str(tmp_path), 3, protected_steps={2})
    assert {path.name for path in tmp_path.iterdir()} == {
        "global_step_2", "global_step_4", "global_step_5", "global_step_6"
    }


def test_masked_verifier_pass_is_raw_only_and_episode_tokens_survive_concatenation():
    def batch(excluded, tokens):
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[7] * 24],
            "loss_masks": [[0] * 24],
            "rewards": [0.0 if excluded else 1.0],
            "unshaped_rewards": [1.0],
            "exclude_from_baseline": [excluded],
            "verification_results": [VerificationResult.verified(1.0, passed=True)],
            "stop_reasons": ["max_turns" if excluded else "stop"],
            "episode_statistics": [
                {"generated_tokens": tokens, "assistant_turns": 128, "tool_calls": 128, "conversations": 128}
            ],
        }

    joined = concatenate_trajectory_batches([batch(True, 3138), batch(False, 4000)], tis_lcs_alert_threshold=0.0)
    assert joined["rollout_metrics"]["generate/avg_num_tokens"] == 3569
    assert joined["rollout_metrics"]["generate/optimized_avg_num_tokens"] == 24
    metrics = _calculate_eval_metrics(joined, ["failed", "healthy"], ["SWE", "SWE"], 1)
    assert metrics["eval/all/pass_at_1"] == 0.5
    assert metrics["eval/all/raw_verifier_pass_at_1"] == 1.0
    assert metrics["eval/all/num_scored"] == 1
    assert metrics["eval/all/verifier_score_coverage"] == 0.5
    assert metrics["eval/all/response_tokens"] == 7138
    assert metrics["eval/all/optimized_response_tokens"] == 48
    assert metrics["eval/all/assistant_turns"] == 256
    assert metrics["eval/SWE/pass_at_1"] == 0.5
    assert metrics["eval/SWE/raw_verifier_pass_at_1"] == 1.0

    failed = calculate_per_dataset_metrics(batch(True, 3138), ["failed"], ["SWE"], 1)
    assert failed["eval/SWE/pass_at_1"] == 0.0
    assert failed["eval/SWE/raw_verifier_pass_at_1"] == 1.0


def test_episode_counters_include_earlier_conversations_and_masked_model_turns():
    turn = ModelTurn(
        message={"role": "assistant", "tool_calls": [{"id": "bash-1"}]},
        prompt_token_ids=(1,),
        response_token_ids=(2, 3),
        logprobs=None,
        stop_reason="tool_calls",
    )
    step = RolloutStep(turn, Transition(done=True), 1, ())
    earlier = ConversationRecord((), (1,), (2, 3), (0, 0), None, (step,))
    last = replace(earlier, steps=(step, step))
    rollout = TrainingRollout(
        task_id="swe",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(2, 3),
        loss_mask=(0, 0),
        logprobs=None,
        grade=GradeResult(Outcome.GRADED, 0.0, passed=False),
        stop_reason="max_turns",
        conversations=(earlier, last),
    )
    assert episode_statistics(rollout) == {
        "generated_tokens": 6,
        "assistant_turns": 3,
        "tool_calls": 3,
        "conversations": 2,
    }
