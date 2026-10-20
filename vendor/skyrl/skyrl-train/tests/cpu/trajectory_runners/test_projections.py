from dataclasses import replace

import numpy as np
import pytest
from omegaconf import OmegaConf
from rolloutengine.contracts import (
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutFailure,
    RolloutStep,
    Transition,
)
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import TrajectoryID, TaskRequestBatch
from tests.cpu.task_specs import lowered_task, task_input, text_task


class Tokenizer:
    eos_token_id = 99


def config():
    return OmegaConf.create({"apply_overlong_filtering": False, "sampling_params": {"logprobs": True}})


def rollout(response_ids, reward, *, metadata=None, metrics=None, grade=None, token_credit=None):
    score = float(sum(reward) if isinstance(reward, list) else reward)
    turn = ModelTurn(
        message={"role": "assistant", "content": "answer"},
        prompt_token_ids=(1, 2),
        response_token_ids=tuple(response_ids),
        logprobs=(-0.1,) * len(response_ids),
        stop_reason="stop",
        metadata=metadata or {},
    )
    grade = grade or GradeResult(Outcome.GRADED, score)
    transition = Transition(
        done=True,
        reward=score,
        token_rewards=tuple(reward) if isinstance(reward, list) else None,
        token_credit=token_credit,
        grade=grade,
    )
    return RolloutData(
        task_id="task",
        messages=(turn.message,),
        prompt_token_ids=turn.prompt_token_ids,
        response_token_ids=turn.response_token_ids,
        loss_mask=(1,) * len(response_ids),
        logprobs=turn.logprobs,
        grade=grade,
        stop_reason="stop",
        steps=(RolloutStep(turn, transition, len(response_ids) - 1, (turn.message,)),),
        metrics=metrics or {},
    )


def trajectory(*turns):
    end = -1
    steps = []
    for turn in turns:
        end += len(turn.response_token_ids)
        steps.append(replace(turn.steps[0], response_end=end))
    return replace(
        turns[-1],
        prompt_token_ids=turns[0].prompt_token_ids,
        response_token_ids=tuple(token for turn in turns for token in turn.response_token_ids),
        loss_mask=tuple(mask for turn in turns for mask in turn.loss_mask),
        logprobs=tuple(value for turn in turns for value in turn.logprobs),
        steps=tuple(steps),
    )


def request(count=1, *, environments=None):
    return TaskRequestBatch(
        rows=tuple(
            task_input(lowered_task(text_task()), metric_class=(environments or ["fixture"] * count)[index])
            for index in range(count)
        ),
        trajectory_ids=tuple(TrajectoryID("task", index) for index in range(count)),
        sampling_params={"logprobs": True},
    )


def test_whole_projection_preserves_tokens_rewards_and_tool_masks():
    record = trajectory(rollout([3], 0.0), rollout([4], 1.0))
    record = replace(
        record,
        response_token_ids=(3, 90, 4),
        loss_mask=(1, 0, 1),
        logprobs=(-0.1, 0.0, -0.1),
        steps=(record.steps[0], replace(record.steps[1], response_end=2)),
    )
    batch = WholeTaskProjection(config(), Tokenizer()).project([record], request())

    assert batch["prompt_token_ids"] == [[1, 2]]
    assert batch["response_ids"] == [[3, 90, 4]]
    assert batch["rewards"] == [[0.0, 0.0, 1.0]]
    assert batch["loss_masks"] == [[1, 0, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"][0], [-0.1, 0.0, -0.1])
    assert "trajectory_ids" not in batch


@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "stop_reason,response_ids,expected_mask",
    [("stop", [3, 128009], [1, 1]), ("length", [3, Tokenizer.eos_token_id], [0, 0])],
)
def test_overlong_filtering_uses_the_served_stop_reason(projection_type, stop_reason, response_ids, expected_mask):
    cfg = config()
    cfg.apply_overlong_filtering = True
    record = rollout(response_ids, 1.0)
    record = replace(
        record,
        stop_reason=stop_reason,
        steps=(replace(record.steps[0], turn=replace(record.steps[0].turn, stop_reason=stop_reason)),),
    )
    batch = projection_type(cfg, Tokenizer()).project([record], request())
    assert batch["response_ids"] == [response_ids]
    assert batch["loss_masks"] == [expected_mask]


def test_projection_retains_the_verifier_reason_when_harbor_classifies_it_as_a_failure():
    record = replace(
        rollout([3, 4], 0.0, grade=GradeResult(Outcome.INFRA_ERROR, None, error="Request timed out.")),
        failure=RolloutFailure("VerifierRuntimeError"),
    )
    batch = WholeTaskProjection(config(), Tokenizer()).project([record], request())
    assert batch["verification_results"][0].reason == "Request timed out."
    assert batch["exception_types"] == ["VerifierRuntimeError"]


def test_projection_preserves_partial_server_error_diagnostics():
    failed = replace(
        rollout([3], 0.0),
        grade=GradeResult(Outcome.INFRA_ERROR, None, error="Model server rejected generation"),
        failure=RolloutFailure("ModelServerError", {"error_category": "constrained_decoding"}),
    )
    batch = WholeTaskProjection(config(), Tokenizer()).project([failed], request())

    assert batch["loss_masks"] == [[0]]
    assert batch["exclude_from_baseline"] == [True]
    assert batch["server_errors"] == [{"category": "constrained_decoding", "request_id": None, "status_code": None}]


@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
def test_projection_rejects_missing_requested_logprobs_on_a_trainable_turn(projection_type):
    record = rollout([3], 1.0)
    record = replace(
        record, logprobs=None, steps=(replace(record.steps[0], turn=replace(record.steps[0].turn, logprobs=None)),)
    )
    with pytest.raises(RolloutContractError, match="missing requested behavior logprobs"):
        projection_type(config(), Tokenizer()).project([record], request())


@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
def test_projection_preserves_routes_and_fills_missing_rows(projection_type):
    routed = rollout([3, 4], [0.0, 1.0], metadata={"routed_experts": np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)})
    batch = projection_type(config(), Tokenizer()).project([routed, rollout([5], [0.0])], request(2))

    np.testing.assert_array_equal(batch["rollout_routed_experts"][0], [[[1, 2]], [[3, 4]]])
    np.testing.assert_array_equal(batch["rollout_routed_experts"][1], [[[0, 0]]])


@pytest.mark.parametrize("terminal_stop_reason", ["stop", "length"])
def test_step_projection_preserves_served_prompts_group_identity_and_final_step(terminal_stop_reason):
    first = rollout([3], [0.0], grade=GradeResult(Outcome.UNAVAILABLE, None))
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}
        ],
    }
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"}
    first = replace(
        first,
        messages=(tool_message, observation),
        steps=(
            replace(
                first.steps[0],
                turn=replace(first.steps[0].turn, message=tool_message),
                transition=replace(first.steps[0].transition, done=False, observations=(observation,)),
                messages=(tool_message, observation),
            ),
        ),
    )
    second = rollout([4, 5], [0.0, 2.0])
    second = replace(
        second,
        stop_reason=terminal_stop_reason,
        steps=(
            replace(
                second.steps[0],
                turn=replace(second.steps[0].turn, prompt_token_ids=(8, 9), stop_reason=terminal_stop_reason),
            ),
        ),
    )
    cfg = config()
    cfg.apply_overlong_filtering = True
    batch = StepTaskProjection(cfg, Tokenizer()).project([trajectory(first, second)], request())

    assert batch["prompt_token_ids"] == [[1, 2], [8, 9]]
    assert batch["response_ids"] == [[3], [4, 5]]
    assert batch["rewards"] == [[0.0], [0.0, 2.0]]
    assert batch["loss_masks"] == [[1], [0, 0] if terminal_stop_reason == "length" else [1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"][0], [-0.1])
    np.testing.assert_allclose(batch["rollout_logprobs"][1], [-0.1, -0.1])
    assert [(item.instance_id, item.repetition_id, item.step) for item in batch["trajectory_ids"]] == [
        ("task", 0, 0),
        ("task", 0, 1),
    ]
    assert batch["is_last_step"] == [False, True]


@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
def test_projection_preserves_topk_credit_and_span_tags(projection_type):
    record = rollout(
        [3, 4],
        [0.0, 1.0],
        token_credit=(0.1, -0.1),
        metadata={
            "student_topk_indices": [[3, 5], [4, 6]],
            "behavior_topk_logprobs": [[-0.1, -1.1], [-0.2, -1.2]],
            "response_span_tags": [1, 3],
        },
    )
    batch = projection_type(config(), Tokenizer()).project([record], request())

    np.testing.assert_array_equal(batch["student_topk_indices"][0], [[3, 5], [4, 6]])
    np.testing.assert_allclose(batch["behavior_topk_logprobs"][0], [[-0.1, -1.1], [-0.2, -1.2]])
    assert batch["token_level_shaping"] == [[0.1, -0.1]]
    assert batch["response_span_tags"] == [[1, 3]]
    assert batch["unshaped_rewards"] == [1.0]


@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
def test_projection_masks_unavailable_verdict_without_inventing_a_score(projection_type):
    record = rollout([3, 4], 1.0, grade=GradeResult(Outcome.UNAVAILABLE, None, error="Judge unavailable"))
    batch = projection_type(config(), Tokenizer()).project([record], request())

    assert batch["loss_masks"] == [[0, 0]]
    assert batch["exclude_from_baseline"] == [True]
    assert batch["exception_types"] == ["VerifierUnavailable"]
    assert batch["unshaped_rewards"] == [0.0]
    assert batch["unshaped_reward_available"] == [False]
    assert batch["verification_results"][0].score is None


def test_whole_projection_carries_environment_rates_into_async_batch():
    n10_hit = rollout([3], 1.0, metrics={"exact_n10": 1.0})
    n10_miss = rollout([4, 6], 0.4725, grade=GradeResult(Outcome.GRADED, 0.0, passed=False), metrics={"exact_n10": 0.0})
    n20_miss = rollout([5, 7, 8], 0.0, metrics={"exact_n20": 0.0})
    projection = WholeTaskProjection(config(), Tokenizer())
    easy_batch = projection.project([n10_hit, n10_miss], request(2, environments=["cat_count", "cat_count"]))
    hard_batch = projection.project([n20_miss], request(environments=["cat_count"]))

    joined = concatenate_trajectory_batches([easy_batch, hard_batch], tis_lcs_alert_threshold=0.005)

    assert joined["rollout_metrics"]["environment/exact_n10"] == 0.5
    assert joined["rollout_metrics"]["environment/exact_n20"] == 0.0
    assert joined["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 1.0
    assert joined["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 2.5

    filtered = filter_trajectory_batch(joined, [0, 1])
    assert filtered["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 1.0
    assert filtered["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 2.0
    assert "environment/exact_n20" not in filtered["rollout_metrics"]
