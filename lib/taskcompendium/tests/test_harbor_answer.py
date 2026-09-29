# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned answer task through Harbor's custom-verifier trial lifecycle."""

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from harbor.models.task.task import Task

from taskcompendium.grading import exact_answer, numeric_answer
from taskcompendium.harbor import script_runtime
from taskcompendium.harbor.runner import ChatLaunch, ReplayLaunch, run_trial
from taskcompendium.lowering import (
    DIRECT_CHAT_ENVIRONMENT,
    WORKSPACE_DOCKER_ENVIRONMENT,
    HarborEnvironmentConfig,
    LoweringCandidate,
    SelectionPolicy,
    compatible_lowerings,
    lower_to_harbor,
    read_specification,
    select_lowerings,
)
from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.script import PrivateResource, ScriptVerifier, script_verifier


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
        verifier=numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0),
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
        requirements=TaskRequirements(),
        answer_type=AnswerType.NUMBER,
    )


@pytest.mark.parametrize(
    "answer_format,response,reward,status",
    [
        (AnswerFormat.PLAIN, "12", 1.0, "graded"),
        (AnswerFormat.PLAIN, "12.0", 1.0, "graded"),
        (AnswerFormat.PLAIN, "13", 0.0, "graded"),
        (AnswerFormat.PLAIN, "not a number", 0.0, "graded"),
        (AnswerFormat.PLAIN, r"\boxed{12}", 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"}', 1.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"13"}', 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"', None, "extraction_error"),
    ],
)
async def test_direct_chat_harbor_trial_distinguishes_answer_outcomes(
    tmp_path, specification, answer_format, response, reward, status
):
    environment_config = HarborEnvironmentConfig()
    convention = SubmissionConvention(id=answer_format.value, answer_format=answer_format)
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")
    assert Task.is_valid_dir(task, disable_verification=True)
    assert "12" not in (task / "instruction.md").read_text()

    result = await run_trial(task, environment_config, ReplayLaunch(response=response), tmp_path / "trials", "run")

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}


async def test_script_verifier_uses_extracted_answer_without_exposing_private_files(
    tmp_path, monkeypatch, specification
):
    script = b"#!/usr/bin/env python3\n"
    reference = b"12"
    resources = tuple(
        PrivateResource(
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            embedded_base64=base64.b64encode(content).decode(),
            executable=executable,
        )
        for path, content, executable in (("grade.py", script, True), ("reference.txt", reference, False))
    )
    verifier = ScriptVerifier(
        entrypoint="grade.py",
        timeout_seconds=5,
        runtime_image=f"example/grader@sha256:{'a' * 64}",
        resources=resources,
    )
    specification = specification.model_copy(update={"verifier": script_verifier(verifier)})
    seen_answers = []

    def fake_docker(command, **kwargs):
        mounts = {}
        for index, token in enumerate(command):
            if token == "--mount":
                fields = dict(part.split("=", 1) for part in command[index + 1].split(",") if "=" in part)
                mounts[fields["dst"]] = fields["src"]
        assert set(mounts) == {"/app", "/tests", "/verifier"}
        assert not (tmp_path / "task-plain/environment/grade.py").exists()
        answer = json.loads((Path(mounts["/verifier"]) / "submission.json").read_text())["answer"]
        seen_answers.append(answer)
        expected = (Path(mounts["/tests"]) / "reference.txt").read_text()
        (Path(mounts["/verifier"]) / "result.json").write_text(
            json.dumps({"status": "scored", "reward": float(answer == expected)})
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script_runtime.subprocess, "run", fake_docker)
    for name, answer_format, response, reward in (
        ("plain", AnswerFormat.PLAIN, "12", 1.0),
        ("json", AnswerFormat.JSON, '{"answer":"12"}', 1.0),
        ("wrong", AnswerFormat.PLAIN, "13", 0.0),
    ):
        task = lower_to_harbor(
            specification,
            SubmissionConvention(id=name, answer_format=answer_format),
            HarborEnvironmentConfig(),
            tmp_path / f"task-{name}",
        )
        assert reference not in (task / "instruction.md").read_bytes()
        result = await run_trial(
            task, HarborEnvironmentConfig(), ReplayLaunch(response=response), tmp_path / "trials", name
        )
        assert result.verifier_result.rewards == {"reward": reward}
    assert seen_answers == ["12", "12", "13"]

    staged_reference = tmp_path / "task-wrong/private_resources/reference.txt"
    staged_reference.write_text("tampered")
    result = await run_trial(
        tmp_path / "task-wrong", HarborEnvironmentConfig(), ReplayLaunch(response="12"), tmp_path / "trials", "tampered"
    )
    outcome = json.loads((tmp_path / "trials/tampered/verifier/taskcompendium-result.json").read_text())
    assert result.verifier_result is None
    assert outcome["status"] == "invalid_task"
    assert outcome["reward"] is None
    assert seen_answers == ["12", "12", "13"]

    staged_reference.unlink()
    result = await run_trial(
        tmp_path / "task-wrong", HarborEnvironmentConfig(), ReplayLaunch(response="12"), tmp_path / "trials", "missing"
    )
    outcome = json.loads((tmp_path / "trials/missing/verifier/taskcompendium-result.json").read_text())
    assert result.verifier_result is None
    assert outcome["status"] == "invalid_task"
    assert outcome["reward"] is None
    assert seen_answers == ["12", "12", "13"]


async def test_direct_chat_exact_comparison_uses_pinned_normalization(tmp_path, specification):
    specification = specification.model_copy(
        update={"verifier": exact_answer("Straße Park"), "answer_type": AnswerType.TEXT}
    )
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        environment_config,
        tmp_path / "task",
    )

    result = await run_trial(
        task,
        environment_config,
        ReplayLaunch(response="STRASSE   PARK"),
        tmp_path / "trials",
        "run",
    )

    assert result.verifier_result.rewards == {"reward": 1.0}


@pytest.mark.parametrize(
    "response,reward",
    [("12.05", 1.0), ("12.2", 0.0)],
)
def test_numeric_answer_uses_explicit_tolerance(specification, response, reward):
    specification = specification.model_copy(
        update={"verifier": numeric_answer(12.0, tolerance_abs=0.1, tolerance_rel=0.0)}
    )
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)

    result = grade_answer(specification, convention, response, object())

    assert (result.status, result.reward) == ("graded", reward)


async def test_direct_chat_harbor_trial_records_private_metadata_failure(tmp_path, specification):
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        environment_config,
        tmp_path / "task",
    )
    (task / "submission_convention.json").write_text("{invalid")

    result = await run_trial(task, environment_config, ReplayLaunch(response="12"), tmp_path / "trials", "run")

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None
    assert result.verifier_result is None


def test_direct_chat_rejects_unsatisfied_requirements(tmp_path, specification):
    specification = specification.model_copy(update={"requirements": TaskRequirements(capabilities=("filesystem",))})

    with pytest.raises(ValueError, match="cannot satisfy"):
        lower_to_harbor(
            specification,
            SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborEnvironmentConfig(),
            tmp_path / "task",
        )


@pytest.mark.parametrize(
    "verifier,message",
    [
        (
            VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json='{"expected": 12}'),
            "Invalid 'exact_answer' verifier parameters",
        ),
        (
            VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json='{"expected": "12", "extra": true}'),
            "Invalid 'exact_answer' verifier parameters",
        ),
    ],
)
def test_lowering_rejects_invalid_verifier_before_writing(tmp_path, specification, verifier, message):
    specification = specification.model_copy(update={"verifier": verifier})

    with pytest.raises(ValueError, match=message):
        lower_to_harbor(
            specification,
            SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborEnvironmentConfig(),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


def test_exported_specification_resolves_verifier_in_fresh_process(tmp_path, specification):
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.verifier_registry import grade_answer; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "result = grade_answer(read_specification(root / 'specification.json'), "
        "read_submission_convention(root / 'submission_convention.json'), '12', object()); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )

    completed = subprocess.run([sys.executable, "-c", script, str(task)], capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}


def test_old_verifier_schema_is_rejected_on_read(tmp_path, specification):
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    path = task / "specification.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = "0.1"
    payload["verifier"] = {"expected": "12", "ignore_case": True, "ignore_whitespace": True}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=r"Unsupported TaskSpec schema: 0\.1"):
        read_specification(path)


def test_file_result_cannot_use_text_submission_convention(tmp_path, specification):
    specification = specification.model_copy(update={"answer_type": AnswerType.FILE})
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)

    assert compatible_lowerings(specification, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(ValueError, match="cannot carry 'file'"):
        lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_workspace_state_lowering_requires_a_snapshot_capable_environment(tmp_path, specification):
    script = b"#!/usr/bin/env python3\n"
    resource = PrivateResource(
        path="grade.py",
        sha256=hashlib.sha256(script).hexdigest(),
        embedded_base64=base64.b64encode(script).decode(),
        executable=True,
    )
    verifier = ScriptVerifier(
        entrypoint="grade.py",
        timeout_seconds=5,
        runtime_image=f"example/grader@sha256:{'a' * 64}",
        resources=(resource,),
    )
    specification = specification.model_copy(
        update={
            "answer_type": AnswerType.WORKSPACE_STATE,
            "requirements": TaskRequirements(capabilities=("filesystem",)),
            "verifier": script_verifier(verifier),
        }
    )
    convention = SubmissionConvention(id="workspace", answer_format=AnswerFormat.WORKSPACE)
    docker_config = HarborEnvironmentConfig(
        environment=WORKSPACE_DOCKER_ENVIRONMENT,
        docker_image=f"example/agent@sha256:{'b' * 64}",
        tools=("filesystem",),
    )

    assert compatible_lowerings(specification, (convention,), (HarborEnvironmentConfig(), docker_config)) == (
        LoweringCandidate(convention, docker_config),
    )
    task = lower_to_harbor(specification, convention, docker_config, tmp_path / "task")
    assert Task.is_valid_dir(task, disable_verification=True)


def test_selection_policies_use_compatible_conventions(specification):
    conventions = (
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        SubmissionConvention(id="json", answer_format=AnswerFormat.JSON),
    )
    candidates = compatible_lowerings(specification, conventions, (HarborEnvironmentConfig(),))

    assert select_lowerings(candidates, SelectionPolicy.ALL) == candidates
    assert select_lowerings(candidates, SelectionPolicy.FIRST) == (candidates[0],)
    repeated = [select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=42) for _ in range(10)]
    assert all(selection == repeated[0] for selection in repeated)
    assert {select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=key)[0] for key in range(16)} == set(candidates)
    assert select_lowerings(candidates, SelectionPolicy.FIRST, required_environment=DIRECT_CHAT_ENVIRONMENT) == (
        candidates[0],
    )
    with pytest.raises(ValueError, match="No compatible lowerings for environment 'shellsim'"):
        select_lowerings(candidates, SelectionPolicy.FIRST, required_environment="shellsim")


async def test_chat_trial_resolves_key_at_runtime_without_persisting_it(
    tmp_path, specification, chat_endpoint, monkeypatch
):
    secret = "taskcompendium-local-test-secret"
    monkeypatch.setenv("TASKCOMPENDIUM_TEST_API_KEY", secret)
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        environment_config,
        tmp_path / "task",
    )
    launch = ChatLaunch(model="fixture-model", api_base=chat_endpoint.url, api_key_env="TASKCOMPENDIUM_TEST_API_KEY")

    result = await run_trial(task, environment_config, launch, tmp_path / "trials", "run")

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert chat_endpoint.authorizations == [f"Bearer {secret}"]
    artifacts = list((tmp_path / "trials/run").rglob("*.json"))
    assert any(path.name == "config.json" for path in artifacts)
    assert any(path.name == "result.json" for path in artifacts)
    assert all(secret not in path.read_text() for path in artifacts)


async def test_chat_http_error_preserves_server_diagnostic(tmp_path, specification, chat_endpoint):
    chat_endpoint.status = 400
    chat_endpoint.body = b'{"error":"model unavailable"}'
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        environment_config,
        tmp_path / "task",
    )
    launch = ChatLaunch(model="fixture-model", api_base=chat_endpoint.url)

    result = await run_trial(task, environment_config, launch, tmp_path / "trials", "run")

    assert result.exception_info is not None
    assert "model unavailable" in (tmp_path / "trials/run/result.json").read_text()
