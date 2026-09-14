# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay retained answer tasks through Harbor without model inference."""

import json
import shlex
from pathlib import Path

import pytest

from taskcompendium.execution import (
    Chat,
    ChatWithTools,
    HarborExecutionConfig,
    HarnessToolBinding,
    NoEnvironment,
    ShellSimEnvironment,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.gsm8k import IMPORTER_REVISION, import_row
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_answers import import_task as import_answer
from taskcompendium.importers.tasktrove_math import import_task as import_math
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    BoxedLatex,
    FileSubmission,
    JsonPath,
    PlainText,
    Rejected,
    Rendering,
    Source,
    TaskSpecification,
    XmlPath,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _source_task(source: str) -> TaskSpecification:
    if source == "gsm8k":
        retained = json.loads((FIXTURES / "gsm8k.json").read_text())
        row = retained["rows"][0]
        specification = import_row(
            row["data"]["question"],
            row["data"]["answer"],
            Source(retained["dataset"], retained["revision"], row["row"], IMPORTER_REVISION),
        )
    else:
        filename, row, family = {
            "mcq": ("answers/mcq-row-1972.tar.gz", "1972", "qa-short-answer"),
            "exact": ("answers/exact-row-119.tar.gz", "119", "math-answer"),
            "math": ("math/math-row-115.tar.gz", "115", "math-answer"),
        }[source]
        archive = read_archive((FIXTURES / "tasktrove" / filename).read_bytes(), row, family)
        specification = import_math(archive) if source == "math" else import_answer(archive)
    assert not isinstance(specification, Rejected), specification
    return specification


# These answers are fixed validation examples from the retained source rows;
# never derive the replay response from the imported verifier's expected value.
@pytest.mark.parametrize(
    "source,submission,good,bad",
    [
        pytest.param("mcq", AssistantFinal(), "C", "D", id="mcq-plain"),
        pytest.param("mcq", AssistantFinal(JsonPath()), '{"answer":"C"}', '{"answer":"D"}', id="mcq-json"),
        pytest.param("mcq", AssistantFinal(XmlPath()), "<answer>C</answer>", "<answer>D</answer>", id="mcq-xml"),
        pytest.param("exact", AssistantFinal(), "off", "on", id="exact-plain"),
        pytest.param("exact", AssistantFinal(JsonPath()), '{"answer":"off"}', '{"answer":"on"}', id="exact-json"),
        pytest.param("math", AssistantFinal(), "246", "247", id="math-plain"),
        pytest.param("math", AssistantFinal(BoxedLatex()), r"\boxed{246}", r"\boxed{247}", id="math-boxed"),
        pytest.param("math", AssistantFinal(JsonPath()), '{"answer":246}', '{"answer":247}', id="math-json"),
        pytest.param("gsm8k", AssistantFinal(), "72", "73", id="gsm8k-train-0-plain"),
    ],
)
@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0), ("empty", None)])
async def test_real_answer_harbor_replay(tmp_path, source, submission, good, bad, attempt, reward):
    specification = _source_task(source)
    response = {"good": good, "bad": bad, "empty": ""}[attempt]
    task = lower_to_harbor(
        specification,
        (Rendering("answer", submission),),
        HarborExecutionConfig(
            "replay",
            environment_for_requirements(specification.requirements),
            interaction=(
                Chat()
                if isinstance(environment_for_requirements(specification.requirements), NoEnvironment)
                else ChatWithTools(
                    (
                        HarnessToolBinding(
                            "replay",
                            (
                                "shellsim"
                                if isinstance(
                                    environment_for_requirements(specification.requirements), ShellSimEnvironment
                                )
                                else "docker"
                            ),
                        ),
                    )
                )
            ),
        ),
        tmp_path / "task",
        agent_kwargs={"response": response},
    )
    execution = json.loads((task / "execution.json").read_text())

    result = await run_trial(task, execution, tmp_path / "trials", attempt)

    outcome = json.loads((tmp_path / f"trials/{attempt}/verifier/taskcompendium-result.json").read_text())
    assert outcome["reward"] == reward
    if reward is None:
        assert outcome["status"] == "extraction_error"
        assert result.verifier_result is None
    else:
        assert outcome["status"] == "graded"
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}


@pytest.mark.parametrize(
    "source,submission,response",
    [
        pytest.param("mcq", AssistantFinal(JsonPath()), '{"answer":"C"', id="mcq-json-truncated"),
        pytest.param("mcq", AssistantFinal(XmlPath()), "<answer>C", id="mcq-xml-truncated"),
        pytest.param("exact", AssistantFinal(JsonPath()), '{"other":"off"}', id="exact-json-missing-answer"),
        pytest.param("math", AssistantFinal(BoxedLatex()), r"\boxed{246", id="math-boxed-unclosed"),
        pytest.param("math", AssistantFinal(BoxedLatex()), r"\boxed{246} \boxed{247}", id="math-boxed-ambiguous"),
        pytest.param("math", AssistantFinal(JsonPath()), '{"answer":246,"answer":247}', id="math-json-duplicate-answer"),
        pytest.param("math", AssistantFinal(JsonPath()), '{"answer":null}', id="math-json-null-answer"),
    ],
)
async def test_real_answer_harbor_malformed_wrapper_has_no_reward(tmp_path, source, submission, response):
    specification = _source_task(source)
    task = lower_to_harbor(
        specification,
        (Rendering("answer", submission),),
        HarborExecutionConfig(
            "replay",
            environment_for_requirements(specification.requirements),
            interaction=(
                Chat()
                if isinstance(environment_for_requirements(specification.requirements), NoEnvironment)
                else ChatWithTools(
                    (
                        HarnessToolBinding(
                            "replay",
                            (
                                "shellsim"
                                if isinstance(
                                    environment_for_requirements(specification.requirements), ShellSimEnvironment
                                )
                                else "docker"
                            ),
                        ),
                    )
                )
            ),
        ),
        tmp_path / "task",
        agent_kwargs={"response": response},
    )
    execution = json.loads((task / "execution.json").read_text())

    result = await run_trial(task, execution, tmp_path / "trials", "malformed")

    outcome = json.loads((tmp_path / "trials/malformed/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "extraction_error"
    assert outcome["reward"] is None
    assert result.verifier_result is None


@pytest.mark.parametrize("source,good,bad", [("math", "246", "247"), ("mcq", "C", "D")])
@pytest.mark.parametrize("extractor", [PlainText(), JsonPath()])
@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0), ("empty", None), ("missing", None)])
async def test_real_answer_harbor_shellsim_grades_file_not_final_response(
    tmp_path, bridge, source, good, bad, extractor, attempt, reward
):
    specification = _source_task(source)
    answer = good if attempt == "good" else bad
    content = json.dumps({"answer": answer}) if isinstance(extractor, JsonPath) else answer
    commands = [f"printf '%s' {shlex.quote(content)} > answer.txt"]
    if attempt == "empty":
        commands = [": > answer.txt"]
    elif attempt == "missing":
        commands = []
    response = bad if attempt == "good" else good
    task = lower_to_harbor(
        specification,
        (Rendering("file", FileSubmission("/app/answer.txt", extractor)),),
        HarborExecutionConfig(
            "replay",
            ShellSimEnvironment(),
            interaction=(ChatWithTools((HarnessToolBinding("replay", "shellsim"),))),
        ),
        tmp_path / "task",
        agent_kwargs={"commands": commands, "response": response},
        environment_kwargs={"bridge_path": str(Path(bridge).resolve())},
    )
    execution = json.loads((task / "execution.json").read_text())

    result = await run_trial(task, execution, tmp_path / "trials", attempt)

    outcome = json.loads((tmp_path / f"trials/{attempt}/verifier/taskcompendium-result.json").read_text())
    assert outcome["reward"] == reward
    if reward is None:
        assert outcome["status"] == "extraction_error"
        assert result.verifier_result is None
    else:
        assert outcome["status"] == "graded"
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
    transcript = json.loads((tmp_path / f"trials/{attempt}/agent/transcript.json").read_text())
    observations = [entry["content"] for entry in transcript if entry["role"] == "tool"]
    assert all(observation["return_code"] == 0 for observation in observations), observations
