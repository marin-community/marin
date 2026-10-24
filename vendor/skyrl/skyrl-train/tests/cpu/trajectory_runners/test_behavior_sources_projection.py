import numpy as np
import pytest
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingStep, TrainingTransition, TrainingRollout
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import TrajectoryID, TaskRequestBatch
from tests.cpu.task_specs import lowered_task, task_input, text_task
from tests.cpu.identity_fixtures import Tokenizer, config, rollout


def request(count=1):
    return TaskRequestBatch(
        rows=tuple(task_input(lowered_task(text_task()), metric_class="fixture") for _ in range(count)),
        trajectory_ids=tuple(TrajectoryID("task", index) for index in range(count)),
        sampling_params={"logprobs": True},
    )


def teacher_rollout(response_ids):
    return rollout(response_ids, 1.0, metadata={"behavior_sources": [0, 0, 1, 1]})


def test_projection_stitches_behavior_sources_per_turn():
    batch = WholeTaskProjection(config(), Tokenizer()).project([teacher_rollout([3, 4, 5, 6])], request())
    np.testing.assert_array_equal(batch["behavior_sources"][0], np.array([0, 0, 1, 1], dtype=np.uint8))


def test_projection_omits_the_channel_when_no_turn_records_it():
    batch = WholeTaskProjection(config(), Tokenizer()).project([rollout([3, 4], 1.0)], request())
    assert "behavior_sources" not in batch


def test_concatenation_fills_missing_batches_with_student_sources():
    teacher_batch = WholeTaskProjection(config(), Tokenizer()).project([teacher_rollout([3, 4, 5, 6])], request())
    plain_batch = WholeTaskProjection(config(), Tokenizer()).project([rollout([7, 8], 1.0)], request())
    merged = concatenate_trajectory_batches([teacher_batch, plain_batch], tis_lcs_alert_threshold=0.005)
    np.testing.assert_array_equal(merged["behavior_sources"][0], np.array([0, 0, 1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(merged["behavior_sources"][1], np.array([0, 0], dtype=np.uint8))


def repair_turn_pair():
    """One switch turn (student prefix + teacher suffix) followed by a student turn."""
    switch_turn = ModelTurn(
        message={"role": "assistant", "content": "answer"},
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4, 5, 6),
        logprobs=(-0.1, -0.1, -0.2, -0.2),
        stop_reason="stop",
        metadata={"behavior_sources": [0, 0, 1, 1]},
    )
    resumed_turn = ModelTurn(
        message={"role": "assistant", "content": "second"},
        prompt_token_ids=(),
        response_token_ids=(7, 8),
        logprobs=(-0.3, -0.3),
        stop_reason="stop",
        metadata={"behavior_sources": [0, 0]},
    )
    steps = (
        TrainingStep(switch_turn, TrainingTransition(done=False), 3, (switch_turn.message,)),
        TrainingStep(
            resumed_turn,
            TrainingTransition(done=True),
            5,
            (switch_turn.message, resumed_turn.message),
        ),
    )
    record = TrainingRollout(
        task_id="task",
        messages=(switch_turn.message, resumed_turn.message),
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4, 5, 6, 7, 8),
        loss_mask=(1, 1, 1, 1, 1, 1),
        logprobs=(-0.1, -0.1, -0.2, -0.2, -0.3, -0.3),
        grade=GradeResult(Outcome.GRADED, 1.0),
        stop_reason="stop",
        steps=steps,
    )
    return record


def test_multi_turn_repair_switch_keeps_turn_alignment():
    batch = WholeTaskProjection(config(), Tokenizer()).project([repair_turn_pair()], request())
    np.testing.assert_array_equal(batch["behavior_sources"][0], np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8))


@pytest.mark.parametrize("routed_first", [True, False])
def test_concatenation_preserves_sparse_teacher_routes(routed_first):
    routed = WholeTaskProjection(config(), Tokenizer()).project([teacher_rollout([3, 4, 5, 6])], request())
    routed["teacher_route_keys"] = ["identity"]
    plain = WholeTaskProjection(config(), Tokenizer()).project([rollout([7, 8], 1.0)], request())
    batches = [routed, plain] if routed_first else [plain, routed]
    merged = concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.005)
    assert merged["teacher_route_keys"] == (["identity", None] if routed_first else [None, "identity"])
    assert [len(row) for row in merged["response_ids"]] == ([4, 2] if routed_first else [2, 4])
