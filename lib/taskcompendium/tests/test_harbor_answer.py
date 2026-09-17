# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned answer task through Harbor's custom-verifier trial lifecycle."""

import dataclasses
import json

import pytest

from taskcompendium.harbor.runner import HarborLaunch, run_trial
from taskcompendium.lowering import HarborTaskBinding, lower_to_harbor
from taskcompendium.models import ExactAnswer, Source, TaskRequirements, TaskSpec
from taskcompendium.rendering import AnswerFormat, Rendering


@pytest.fixture
def specification() -> TaskSpec:
    return TaskSpec(
        id="arithmetic-7-plus-5",
        instructions="What is 7 + 5?",
        verifier=ExactAnswer("12"),
        source=Source("hand-authored", "2026-09-16", "arithmetic-7-plus-5", "1"),
        requirements=TaskRequirements(),
    )


@pytest.mark.parametrize(
    "answer_format,response,reward,status",
    [
        (AnswerFormat.PLAIN, "12", 1.0, "graded"),
        (AnswerFormat.PLAIN, "13", 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"}', 1.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"13"}', 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"', None, "extraction_error"),
    ],
)
async def test_direct_chat_harbor_trial_distinguishes_answer_outcomes(
    tmp_path, specification, answer_format, response, reward, status
):
    binding = HarborTaskBinding()
    rendering = Rendering(answer_format.value, answer_format)
    task = lower_to_harbor(specification, rendering, binding, tmp_path / "task")
    assert not (task / "tests" / "test.sh").exists()
    assert "12" not in (task / "instruction.md").read_text()
    assert "verif" not in (task / "instruction.md").read_text().lower()

    result = await run_trial(
        task, binding, HarborLaunch("replay", agent_kwargs={"response": response}), tmp_path / "trials", "run"
    )

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}


async def test_direct_chat_harbor_trial_records_private_metadata_failure(tmp_path, specification):
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, Rendering("plain", AnswerFormat.PLAIN), binding, tmp_path / "task")
    (task / "rendering.json").write_text("{invalid")

    result = await run_trial(
        task, binding, HarborLaunch("replay", agent_kwargs={"response": "12"}), tmp_path / "trials", "run"
    )

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None
    assert result.verifier_result is None


def test_direct_chat_rejects_unsatisfied_requirements(tmp_path, specification):
    specification = dataclasses.replace(specification, requirements=TaskRequirements(capabilities=("filesystem",)))

    with pytest.raises(ValueError, match="cannot satisfy"):
        lower_to_harbor(specification, Rendering("plain", AnswerFormat.PLAIN), HarborTaskBinding(), tmp_path / "task")


async def test_launch_rejects_binding_changed_after_export(tmp_path, specification):
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, Rendering("plain", AnswerFormat.PLAIN), binding, tmp_path / "task")
    (task / "binding.json").write_text('{"environment":"direct_chat","tools":["terminal"]}')

    with pytest.raises(ValueError, match="direct chat without tools"):
        await run_trial(
            task, binding, HarborLaunch("replay", agent_kwargs={"response": "12"}), tmp_path / "trials", "run"
        )
