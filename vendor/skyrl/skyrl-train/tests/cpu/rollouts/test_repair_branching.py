import asyncio

import numpy as np
from tests.cpu.identity_fixtures import CharTokenizer, FakeClient, original_rollout, tokens
import pytest
from taskcompendium.grading_result import Outcome

from skyrl_gym.identity_repair.detector import AttributionDetection
from skyrl_gym.identity_repair.judge import AttributionClaim, ClaimKind
from skyrl_train.inference_engines.chat_continuation import render_exact_chat_continuation
from skyrl_train.rollouts.repair_branching import (
    TeacherRepairSpec,
    expand_identity_repair_group,
)
from skyrl_train.rollouts.task_projections import _model_channel


def detection(text):
    claim = "my name is qwen"
    start = text.index(claim)
    return AttributionDetection(
        AttributionClaim(ClaimKind.NAME, claim, start, start + len(claim), "qwen"), start, False, "judge-v1"
    )


def expand(original, text, teacher, student):
    return asyncio.run(
        expand_identity_repair_group(
            original,
            [detection(text)],
            tokenizer=CharTokenizer(),
            teacher_client=teacher,
            student_client=student,
            spec=TeacherRepairSpec({"temperature": 1.0}, {"temperature": 1.0}, 8192),
            session_id="test",
        )
    )


@pytest.mark.parametrize("leading_turns", [[], ["a harmless answer"]])
def test_repair_preserves_branch_history_masks_and_cumulative_sources(leading_turns):
    text = "<|start_think|>Check facts.<|end_think|>my name is qwen"
    original = original_rollout([*leading_turns, text, "original follow-up"])
    teacher, student = FakeClient(["I am Marin Snowball"]), FakeClient(["branch follow-up"])
    paired, repaired = expand(original, text, teacher, student)
    boundary = repaired.metrics["identity_repair"]["boundary_token_index"]
    assert not any(paired.loss_mask[:boundary])
    assert not any(repaired.loss_mask[:boundary])
    assert original.loss_mask != paired.loss_mask
    history = student.histories[0]
    assert history[-2]["content"] == "I am Marin Snowball"
    assert history[-1]["role"] == "user"
    assert all("qwen" not in message["content"] for message in history)
    assert repaired.messages[-1]["content"] == "branch follow-up"
    assert repaired.grade.status == Outcome.UNAVAILABLE
    teacher_start = boundary
    teacher_end = boundary + len(tokens("I am Marin Snowball"))
    sources = _model_channel(repaired, "behavior_sources", len(repaired.response_token_ids), (), np.uint8)
    assert np.flatnonzero(sources).tolist() == list(range(teacher_start, teacher_end))
    observation_end = len(repaired.response_token_ids) - len(tokens("branch follow-up"))
    assert not any(repaired.loss_mask[teacher_end:observation_end])
    assert all(repaired.loss_mask[observation_end:])
    assert len(repaired.logprobs) == len(repaired.response_token_ids) == len(repaired.loss_mask)


def test_empty_response_prefix_replaces_the_wrong_message():
    original = original_rollout(["my name is qwen"])
    teacher = FakeClient(["I am Marin Snowball"])
    _, repaired = expand(original, "my name is qwen", teacher, FakeClient([]))
    assert teacher.histories[0][-1] == {"role": "assistant", "content": ""}
    assert repaired.steps[0].turn.prompt_token_ids == original.prompt_token_ids
    assert repaired.messages[-1]["content"] == "I am Marin Snowball"
    assert all(repaired.loss_mask)


def test_repair_failure_retains_original_without_masking_or_resampling():
    original = original_rollout(["my name is qwen"])
    outputs = expand(original, "my name is qwen", FakeClient([RuntimeError("unavailable")]), FakeClient([]))
    assert outputs == [original]
    assert all(outputs[0].loss_mask)
    assert outputs[0].metrics["identity_repair_rejections"][0]["reason"] == "generator_error"


@pytest.mark.parametrize("partial", ["", "<|start_think|>Check facts.", "<|start_think|>Check.<|end_think|>"])
def test_midturn_render_preserves_exact_native_prefix_without_closing_turn(partial):
    async def tokenize(request):
        raise AssertionError("An exact in-turn prefix must not be re-tokenized")

    prefix = list(tokens("USER: identity\nASSISTANT: " + partial))
    actual = asyncio.run(
        render_exact_chat_continuation(
            tokenize,
            {
                "json": {
                    "messages": [{"role": "user", "content": "identity"}, {"role": "assistant", "content": partial}]
                }
            },
            assistant_message_index=1,
            served_prefix_token_ids=prefix,
            continue_final_message=True,
        )
    )
    assert actual == prefix


def test_repair_rejects_wrong_entity_retained_in_earlier_reasoning():
    text = "<|start_think|>Perhaps GPT-4.<|end_think|>I am using the GPT-4 model."
    original = original_rollout([text])
    start = text.index("I am using")
    claim = AttributionClaim(
        ClaimKind.NAME, "I am using the GPT-4", start, start + len("I am using the GPT-4"), "gpt-4"
    )
    teacher, student = FakeClient([]), FakeClient([])
    result = asyncio.run(
        expand_identity_repair_group(
            original,
            [AttributionDetection(claim, start, False, "judge-v1")],
            tokenizer=CharTokenizer(),
            teacher_client=teacher,
            student_client=student,
            spec=TeacherRepairSpec({"temperature": 1.0}, {"temperature": 1.0}, 8192),
            session_id="test",
        )
    )
    assert len(result) == 1
    assert result[0].metrics["identity_repair_rejections"][0]["reason"] == "claim_prefix_collision"
    assert teacher.calls == []
