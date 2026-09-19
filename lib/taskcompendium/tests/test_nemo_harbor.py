# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded NeMo imports through Harbor's actual trial lifecycle."""

import json
from html import escape
from pathlib import Path

import pytest

import taskcompendium.grading as grading
from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig, HarborTaskBinding, NoEnvironment
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.nemo import load_code_sample, load_instruction_sample
from taskcompendium.importers.nemo_predicted_action import import_row, replay_action
from taskcompendium.importers.nemo_predicted_action import rendering as action_rendering
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import AssistantFinal, JsonPath, Rejected, Rendering, XmlPath

pytestmark = pytest.mark.harbor_conformance

FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _replay_execution(binding: HarborTaskBinding) -> HarborExecutionConfig:
    return HarborExecutionConfig(binding, HarborLaunchConfig("replay"))


def _attempts() -> dict[str, dict[str, str]]:
    return json.loads((FIXTURES / "attempts.json").read_text())


def _outcome(root: Path, trial_name: str) -> dict:
    return json.loads((root / "trials" / trial_name / "verifier" / "taskcompendium-result.json").read_text())


@pytest.mark.parametrize(
    ("format_id", "submission", "encode"),
    [
        pytest.param("plain", AssistantFinal(), lambda answer: answer, id="plain"),
        pytest.param("json", AssistantFinal(JsonPath()), lambda answer: json.dumps({"answer": answer}), id="json"),
        pytest.param("xml", AssistantFinal(XmlPath()), lambda answer: f"<answer>{answer}</answer>", id="xml"),
    ],
)
@pytest.mark.parametrize(
    ("aggregation", "attempt", "reward"),
    [
        pytest.param("binary", "good", 1.0, id="binary-good"),
        pytest.param("binary", "one_constraint_wrong", 0.0, id="binary-wrong"),
        pytest.param("fraction", "good", 1.0, id="fraction-good"),
        pytest.param("fraction", "one_constraint_wrong", 0.5, id="fraction-wrong"),
    ],
)
async def test_nemo_ifeval_harbor_preserves_aggregation_and_response_format(
    tmp_path, format_id, submission, encode, aggregation, attempt, reward
):
    specification = load_instruction_sample(FIXTURES / "instruction-following-17616.json", aggregation=aggregation)
    assert not isinstance(specification, Rejected)
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (Rendering(f"nemo-ifeval-{format_id}", submission),),
        binding,
        tmp_path / "task",
        reference_execution=_replay_execution(binding),
        agent_kwargs={"response": encode(_attempts()["instruction_following"][attempt])},
    )

    trial_name = f"{aggregation}-{attempt}-{format_id}"
    result = await run_trial(
        task, json.loads((task / "reference-execution.json").read_text()), tmp_path / "trials", trial_name
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    outcome = _outcome(tmp_path, trial_name)
    assert outcome["status"] == "graded"
    assert outcome["reward"] == reward
    assert [constraint["passed"] for constraint in outcome["detail"]["constraints"]] == (
        [True, True] if attempt == "good" else [True, False]
    )


@pytest.mark.docker
@pytest.mark.parametrize(
    ("submission", "encode", "attempt", "reward"),
    [
        pytest.param(AssistantFinal(), lambda response: response, "good", 1.0, id="plain-good"),
        pytest.param(AssistantFinal(), lambda response: response, "wrong", 0.0, id="plain-wrong"),
        pytest.param(AssistantFinal(), lambda response: response, "malformed", 0.0, id="plain-malformed"),
        pytest.param(AssistantFinal(), lambda response: response, "empty", 0.0, id="plain-empty"),
        pytest.param(
            AssistantFinal(JsonPath()), lambda response: json.dumps({"answer": response}), "good", 1.0, id="json-good"
        ),
        pytest.param(
            AssistantFinal(XmlPath()),
            lambda response: f"<answer>{escape(response)}</answer>",
            "good",
            1.0,
            id="xml-good",
        ),
    ],
)
async def test_nemo_code_answer_harbor_uses_private_isolated_grader(
    tmp_path, runtime_image, submission, encode, attempt, reward
):
    specification = load_code_sample(
        FIXTURES / "code-answer-c69268d8bdb4da0685d7b187c88296c1.json", verifier_image=runtime_image
    )
    assert not isinstance(specification, Rejected)
    response = "" if attempt == "empty" else _attempts()["code_answer"][attempt]
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (Rendering("nemo-code-answer", submission),),
        binding,
        tmp_path / "task",
        reference_execution=_replay_execution(binding),
        agent_kwargs={"response": encode(response)},
    )

    result = await run_trial(
        task, json.loads((task / "reference-execution.json").read_text()), tmp_path / "trials", attempt
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    outcome = _outcome(tmp_path, attempt)
    assert outcome["status"] == "graded"
    assert outcome["reward"] == reward
    assert outcome["detail"]["isolation"] == "docker-unprivileged-candidate"


@pytest.mark.docker
@pytest.mark.parametrize(
    ("submission", "response"),
    [
        pytest.param(AssistantFinal(JsonPath()), '{"answer":', id="json"),
        pytest.param(AssistantFinal(XmlPath()), "<answer>", id="xml"),
    ],
)
async def test_nemo_code_answer_harbor_rejects_malformed_structured_wrappers(
    tmp_path, runtime_image, submission, response
):
    specification = load_code_sample(
        FIXTURES / "code-answer-c69268d8bdb4da0685d7b187c88296c1.json", verifier_image=runtime_image
    )
    assert not isinstance(specification, Rejected)
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (Rendering("nemo-code-answer", submission),),
        binding,
        tmp_path / "task",
        reference_execution=_replay_execution(binding),
        agent_kwargs={"response": response},
    )

    result = await run_trial(
        task, json.loads((task / "reference-execution.json").read_text()), tmp_path / "trials", "malformed"
    )

    assert result.verifier_result is None
    outcome = _outcome(tmp_path, "malformed")
    assert outcome["status"] == "extraction_error"
    assert outcome["reward"] is None


@pytest.mark.parametrize("attempt", ["good", "wrong", "malformed"])
async def test_nemo_predicted_action_harbor_distinguishes_correct_wrong_and_malformed_submissions(tmp_path, attempt):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    attempts = json.loads((FIXTURES / "predicted-action.attempts.json").read_text())
    specification = import_row(row, provenance["canonical_json_sha256"])
    assert not isinstance(specification, Rejected)
    protocol = action_rendering(row, provenance["canonical_json_sha256"])
    if attempt == "malformed":
        action = {"tool_calls": [{"id": "malformed", "function": {"name": "unavailable", "arguments": "{}"}}]}
    else:
        action_data = attempts[attempt]
        action = replay_action(action_data["name"], action_data["arguments"])
    reward = 1.0 if attempt == "good" else 0.0
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (protocol,),
        binding,
        tmp_path / "task",
        reference_execution=_replay_execution(binding),
        agent_kwargs={"actions": [action]},
    )

    result = await run_trial(
        task, json.loads((task / "reference-execution.json").read_text()), tmp_path / "trials", attempt
    )

    if attempt == "malformed":
        assert result.verifier_result is None
        assert result.exception_info is not None
        assert not (tmp_path / "trials" / attempt / "verifier" / "taskcompendium-result.json").exists()
        return

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    outcome = _outcome(tmp_path, attempt)
    assert outcome["status"] == "graded"
    assert outcome["reward"] == reward
    transcript = json.loads((tmp_path / "trials" / attempt / "agent" / "transcript.json").read_text())
    assert [entry["role"] for entry in transcript] == ["user", "assistant"]
    assert transcript[-1]["tool_calls"] == action["tool_calls"]


async def test_nemo_predicted_action_harbor_records_verifier_crash_as_infrastructure_failure(tmp_path, monkeypatch):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    attempts = json.loads((FIXTURES / "predicted-action.attempts.json").read_text())
    specification = import_row(row, provenance["canonical_json_sha256"])
    assert not isinstance(specification, Rejected)
    protocol = action_rendering(row, provenance["canonical_json_sha256"])
    action_data = attempts["good"]
    action = replay_action(action_data["name"], action_data["arguments"])
    binding = HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        (protocol,),
        binding,
        tmp_path / "task",
        reference_execution=_replay_execution(binding),
        agent_kwargs={"actions": [action]},
    )

    def crash(*_args, **_kwargs):
        raise RuntimeError("fixture comparator outage")

    monkeypatch.setattr(grading, "compare", crash)
    result = await run_trial(
        task, json.loads((task / "reference-execution.json").read_text()), tmp_path / "trials", "infra"
    )

    assert result.verifier_result is None
    assert result.exception_info is not None
    outcome = _outcome(tmp_path, "infra")
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None
    assert outcome["detail"]["error"] == "RuntimeError: fixture comparator outage"
