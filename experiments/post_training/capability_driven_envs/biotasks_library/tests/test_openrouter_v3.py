# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    JsonObjectInFinal,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import StructuredExactSpec

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_openrouter_first_wave_v3 import _score
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import PinnedModel
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import _batches


def _task() -> TaskSpec:
    return TaskSpec(
        id="new-frozen-id",
        context=ConversationInput(events=(TextMessage(role="user", content="Give x."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.JSON,
        answer_format=JsonObjectInFinal(),
        grader=verifyit_package(StructuredExactSpec(expected={"x": 1})).grader,
        source=Source(dataset="test", revision="1", row="1", importer_revision="1"),
    )


def _raw(content: str, status: str = "graded") -> dict:
    return {
        "trial": {"status": status},
        "attempts": [
            {
                "rollout": {
                    "failure": None,
                    "conversations": [
                        {"messages": [{"role": "user", "content": "Give x."}, {"role": "assistant", "content": content}]}
                    ],
                }
            }
        ],
    }


def test_premium_batches_partition_once_without_exceeding_per_batch_reserve() -> None:
    tasks = [_task().model_copy(update={"id": f"item-{index}"}) for index in range(169)]
    model = PinnedModel("anthropic/claude-sonnet-4.6", "anthropic", "Anthropic", 200000)
    batches = _batches(tasks, model, "s3://private/")
    assert [len(chunk) for _, chunk in batches] == [57, 57, 55]
    assert [task.id for _, chunk in batches for task in chunk] == [task.id for task in tasks]
    assert len({uri for uri, _ in batches}) == 3


def test_v3_offline_scoring_accepts_unambiguous_prose_but_not_multiple_answers() -> None:
    assert _score(_task(), _raw('Answer: ```json\n{"x": 1}\n```')) == (1, "graded")
    assert _score(_task(), _raw('{"x": 0}')) == (0, "graded")
    assert _score(_task(), _raw('{"x": 1} or {"x": 2}')) == (0, "submission_failure")
    assert _score(_task(), _raw('{"x": 1}', status="ungraded")) == (None, "ungraded")
