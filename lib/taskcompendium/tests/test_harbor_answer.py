# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned answer task through Harbor's custom-verifier trial lifecycle."""

import json
import subprocess
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from harbor.models.task.task import Task

from taskcompendium.grading import (
    ExactAnswerPayload,
    GradeResult,
    Outcome,
    VerifierHandler,
    exact_answer,
    grade_answer,
)
from taskcompendium.harbor.runner import HarborLaunch, run_trial
from taskcompendium.lowering import (
    HarborTaskBinding,
    SelectionPolicy,
    compatible_lowerings,
    lower_to_harbor,
    read_specification,
    select_lowerings,
)
from taskcompendium.models import AnswerFormat, Source, TaskRequirements, TaskSpec, VerifierSpec
from taskcompendium.rendering import Rendering


@dataclass
class ChatEndpoint:
    url: str
    authorizations: list[str | None]
    status: int
    body: bytes


@pytest.fixture
def chat_endpoint():
    endpoint = ChatEndpoint("", [], 200, b'{"choices":[{"message":{"role":"assistant","content":"12"}}]}')

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            endpoint.authorizations.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(endpoint.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(endpoint.body)

        def log_message(self, format, *args):  # noqa: A002 - match BaseHTTPRequestHandler
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def specification() -> TaskSpec:
    return TaskSpec(
        id="arithmetic-7-plus-5",
        instructions="What is 7 + 5?",
        verifier=exact_answer("12"),
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
        requirements=TaskRequirements(),
        permitted_answer_formats=(AnswerFormat.PLAIN, AnswerFormat.JSON),
    )


@pytest.mark.parametrize(
    "answer_format,response,reward,status",
    [
        (AnswerFormat.PLAIN, "12", 1.0, "graded"),
        (AnswerFormat.PLAIN, "13", 0.0, "graded"),
        (AnswerFormat.PLAIN, r"\boxed{12}", 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"}', 1.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"13"}', 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"', None, "extraction_error"),
    ],
)
async def test_direct_chat_harbor_trial_distinguishes_answer_outcomes(
    tmp_path, specification, answer_format, response, reward, status
):
    binding = HarborTaskBinding()
    rendering = Rendering(id=answer_format.value, answer_format=answer_format)
    task = lower_to_harbor(specification, rendering, binding, tmp_path / "task")
    assert Task.is_valid_dir(task, disable_verification=True)
    assert not (task / "tests" / "test.sh").exists()
    assert "12" not in (task / "instruction.md").read_text()
    assert "verif" not in (task / "instruction.md").read_text().lower()
    assert json.loads((task / "specification.json").read_text())["verifier"]["kind"] == "exact_answer"

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


async def test_direct_chat_exact_comparison_uses_pinned_normalization(tmp_path, specification):
    specification = specification.model_copy(update={"verifier": exact_answer("Straße Park")})
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )

    result = await run_trial(
        task,
        binding,
        HarborLaunch("replay", agent_kwargs={"response": "STRASSE   PARK"}),
        tmp_path / "trials",
        "run",
    )

    assert result.verifier_result.rewards == {"reward": 1.0}


async def test_direct_chat_harbor_trial_records_private_metadata_failure(tmp_path, specification):
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )
    (task / "rendering.json").write_text("{invalid")

    result = await run_trial(
        task, binding, HarborLaunch("replay", agent_kwargs={"response": "12"}), tmp_path / "trials", "run"
    )

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None
    assert result.verifier_result is None


def test_direct_chat_rejects_unsatisfied_requirements(tmp_path, specification):
    specification = specification.model_copy(update={"requirements": TaskRequirements(capabilities=("filesystem",))})

    with pytest.raises(ValueError, match="cannot satisfy"):
        lower_to_harbor(
            specification,
            Rendering(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborTaskBinding(),
            tmp_path / "task",
        )


def test_lowering_rejects_answer_format_forbidden_by_task(tmp_path, specification):
    specification = specification.model_copy(
        update={
            "instructions": "Return only the raw C++ program output.",
            "permitted_answer_formats": (AnswerFormat.PLAIN,),
        }
    )
    destination = tmp_path / "task"

    with pytest.raises(ValueError, match="does not permit the 'json' answer format"):
        lower_to_harbor(
            specification,
            Rendering(id="json", answer_format=AnswerFormat.JSON),
            HarborTaskBinding(),
            destination,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    "verifier,message",
    [
        (VerifierSpec(kind="unknown_kind", parameters={}), "Unknown or ambiguous verifier kind"),
        (VerifierSpec(kind="exact_answer", parameters={"expected": 12}), "Invalid 'exact_answer' verifier parameters"),
        (
            VerifierSpec(kind="exact_answer", parameters={"expected": "12", "extra": True}),
            "Invalid 'exact_answer' verifier parameters",
        ),
    ],
)
def test_lowering_rejects_unknown_or_invalid_verifier_before_writing(tmp_path, specification, verifier, message):
    specification = specification.model_copy(update={"verifier": verifier})

    with pytest.raises(ValueError, match=message):
        lower_to_harbor(
            specification,
            Rendering(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborTaskBinding(),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


def test_exported_specification_resolves_verifier_in_fresh_process(tmp_path, specification):
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), HarborTaskBinding(), tmp_path / "task"
    )
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import grade_answer; "
        "from taskcompendium.lowering import read_rendering, read_specification; "
        "root = Path(sys.argv[1]); "
        "result = grade_answer(read_specification(root / 'specification.json'), "
        "read_rendering(root / 'rendering.json'), '12', object()); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )

    completed = subprocess.run([sys.executable, "-c", script, str(task)], capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}


def test_verifier_parameters_cannot_change_exported_or_live_grading(tmp_path, specification):
    parameters = {"expected": "12", "ignore_case": True, "collapse_whitespace": True}
    verifier = VerifierSpec(kind="exact_answer", parameters=parameters)
    specification = specification.model_copy(update={"verifier": verifier})
    parameters["expected"] = "13"
    verifier.parameters["expected"] = "13"

    rendering = Rendering(id="plain", answer_format=AnswerFormat.PLAIN)
    result = grade_answer(specification, rendering, "12", object())
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    task = lower_to_harbor(specification, rendering, HarborTaskBinding(), tmp_path / "task")
    exported = read_specification(task / "specification.json")
    assert grade_answer(exported, rendering, "12", object()).reward == 1.0
    assert grade_answer(exported, rendering, "13", object()).reward == 0.0


def test_registered_grader_receives_verifier_environment(specification, monkeypatch):
    environment = object()

    def grade_probe(payload: ExactAnswerPayload, attempt) -> GradeResult:
        assert attempt.environment is environment
        return GradeResult(Outcome.GRADED, float(payload.expected == "12"))

    handler = VerifierHandler(ExactAnswerPayload, grade_probe)
    entry = SimpleNamespace(name="environment_probe", load=lambda: lambda: handler)
    monkeypatch.setattr("taskcompendium.grading.entry_points", lambda *, group: (entry,))
    specification = specification.model_copy(
        update={"verifier": VerifierSpec(kind="environment_probe", parameters={"expected": "12"})}
    )

    result = grade_answer(specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), "12", environment)

    assert result == GradeResult(Outcome.GRADED, 1.0)


@pytest.mark.parametrize("schema_version", ["0.1", "0.2", "0.3"])
def test_old_task_spec_schema_is_rejected_on_read(tmp_path, specification, schema_version):
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), HarborTaskBinding(), tmp_path / "task"
    )
    path = task / "specification.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=rf"Unsupported TaskSpec schema: {schema_version}"):
        read_specification(path)


def test_selection_exports_only_compatible_answer_format(tmp_path, specification):
    specification = specification.model_copy(update={"permitted_answer_formats": (AnswerFormat.PLAIN,)})
    renderings = (
        Rendering(id="json", answer_format=AnswerFormat.JSON),
        Rendering(id="plain", answer_format=AnswerFormat.PLAIN),
    )
    candidates = compatible_lowerings(specification, renderings, (HarborTaskBinding(),))

    selected = select_lowerings(candidates, SelectionPolicy.FIRST)
    task = lower_to_harbor(specification, selected[0].rendering, selected[0].binding, tmp_path / "task")

    assert len(candidates) == 1
    assert json.loads((task / "rendering.json").read_text())["answer_format"] == "plain"
    assert (
        compatible_lowerings(
            specification.model_copy(update={"requirements": TaskRequirements(capabilities=("filesystem",))}),
            renderings,
            (HarborTaskBinding(),),
        )
        == ()
    )


def test_selection_samples_reproducibly_from_compatible_renderings(specification):
    renderings = (
        Rendering(id="plain", answer_format=AnswerFormat.PLAIN),
        Rendering(id="json", answer_format=AnswerFormat.JSON),
    )
    candidates = compatible_lowerings(specification, renderings, (HarborTaskBinding(),))

    assert select_lowerings(candidates, SelectionPolicy.ALL) == candidates
    assert select_lowerings(candidates, SelectionPolicy.FIRST) == (candidates[0],)
    assert select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=42) == select_lowerings(
        candidates, SelectionPolicy.SAMPLE, rng_key=42
    )
    assert {select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=key)[0] for key in range(16)} == set(candidates)


async def test_launch_rejects_binding_changed_after_export(tmp_path, specification):
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )
    (task / "binding.json").write_text('{"environment":"direct_chat","tools":["terminal"]}')

    with pytest.raises(ValueError, match="direct chat without tools"):
        await run_trial(
            task, binding, HarborLaunch("replay", agent_kwargs={"response": "12"}), tmp_path / "trials", "run"
        )


async def test_chat_trial_resolves_key_at_runtime_without_persisting_it(
    tmp_path, specification, chat_endpoint, monkeypatch
):
    secret = "taskcompendium-local-test-secret"
    monkeypatch.setenv("TASKCOMPENDIUM_TEST_API_KEY", secret)
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )
    launch = HarborLaunch(
        "chat",
        model="fixture-model",
        agent_kwargs={"api_base": chat_endpoint.url, "api_key_env": "TASKCOMPENDIUM_TEST_API_KEY"},
    )

    result = await run_trial(task, binding, launch, tmp_path / "trials", "run")

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert chat_endpoint.authorizations == [f"Bearer {secret}"]
    artifacts = list((tmp_path / "trials/run").rglob("*.json"))
    assert any(path.name == "config.json" for path in artifacts)
    assert any(path.name == "result.json" for path in artifacts)
    assert all(secret not in path.read_text() for path in artifacts)


async def test_chat_launch_rejects_raw_key(tmp_path, specification, chat_endpoint):
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )
    launch = HarborLaunch(
        "chat", model="fixture-model", agent_kwargs={"api_base": chat_endpoint.url, "api_key": "secret"}
    )

    with pytest.raises(ValueError, match="api_key_env"):
        await run_trial(task, binding, launch, tmp_path / "trials", "run")


async def test_chat_http_error_preserves_server_diagnostic(tmp_path, specification, chat_endpoint):
    chat_endpoint.status = 400
    chat_endpoint.body = b'{"error":"model unavailable"}'
    binding = HarborTaskBinding()
    task = lower_to_harbor(
        specification, Rendering(id="plain", answer_format=AnswerFormat.PLAIN), binding, tmp_path / "task"
    )
    launch = HarborLaunch("chat", model="fixture-model", agent_kwargs={"api_base": chat_endpoint.url})

    result = await run_trial(task, binding, launch, tmp_path / "trials", "run")

    assert result.exception_info is not None
    assert "model unavailable" in (tmp_path / "trials/run/result.json").read_text()
