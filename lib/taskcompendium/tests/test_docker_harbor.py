# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""File submissions through Harbor's real lifecycle with CPU Docker I/O."""

import asyncio
import base64
import gzip
import io
import json
import os
import re
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import UnixStreamServer
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit

import pytest
from harbor.agents.base import BaseAgent
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.trial.trial import Trial
from shellbox.machine import Command, ExitReason, InvalidWorkspaceFile, MachineSpec, Result

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.harbor.docker import DockerWorkspaceEnvironment
from taskcompendium.harbor.runner import run_trial
from taskcompendium.lowering import DOCKER_ENVIRONMENT, HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import JsonFile, TextFile

IMAGE = "python:3.12-slim-bullseye@sha256:411fa4dcfdce7e7a3057c45662beba9dcd4fa36b2e50a2bfcd6c9333e59bf0db"


class WorkspaceAgent(BaseAgent):
    """Fixture agent writes one final file and never emits a chat transcript."""

    def __init__(self, *args, command: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.command = command

    @staticmethod
    def name() -> str:
        return "workspace-fixture"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        diagnostics = await environment.exec(
            "printf worker-diagnostic > /logs/agent/worker.log; printf worker-artifact > /logs/artifacts/worker.txt"
        )
        if diagnostics.return_code != 0:
            raise RuntimeError(diagnostics.stderr or diagnostics.stdout)
        result = await environment.exec(self.command)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout)


class LocalDirectoryMachine:
    """Test-only machine with real files and shell execution, without a Docker CLI."""

    started: asyncio.Event

    def __init__(self, root: Path, specification: MachineSpec):
        self.root = root
        self.specification = specification
        self.running = True
        root.mkdir()

    def _path(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    async def run(self, command: Command) -> Result:
        if not self.running:
            raise RuntimeError("Local test machine is stopped")
        program = command.argv[-1]
        if program == "fixture-block":
            self._path(self.specification.workdir).joinpath("answer.txt").write_text("12")
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.running = False
                raise
        program = re.sub(r"/(workspace|logs)(?=/|$|[\s\"';])", lambda match: str(self.root / match.group(1)), program)
        process = await asyncio.create_subprocess_exec(
            *command.argv[:-1],
            program,
            cwd=self._path(command.cwd or self.specification.workdir),
            env={**os.environ, **command.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(command.stdin), command.timeout)
        except BaseException:
            process.kill()
            await process.wait()
            self.running = False
            raise
        return Result(
            process.returncode,
            stdout[: command.output_limit_bytes],
            stderr[: command.output_limit_bytes],
            len(stdout) > command.output_limit_bytes,
            len(stderr) > command.output_limit_bytes,
            ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        destination = self._path(target)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    async def download(self, source: str, target: Path) -> None:
        path = self._path(source)
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    async def read_file(self, path: str, max_bytes: int) -> bytes | None:
        candidate = self._path(self.specification.workdir) / path
        if not candidate.exists():
            return None
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > max_bytes:
            raise InvalidWorkspaceFile("Local test candidate must be one bounded regular file")
        with candidate.open("rb") as stream:
            content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise InvalidWorkspaceFile("Local test candidate exceeds its byte limit")
        return content

    async def close(self) -> None:
        paths = [str(path.relative_to(self.root)) for path in self.root.rglob("*")]
        (self.root.parent / "closed-paths.json").write_text(json.dumps(paths))
        shutil.rmtree(self.root)
        self.running = False


class LocalDirectoryFactory:
    """Supply the test-only machine through the production MachineFactory seam."""

    def __init__(self, root: Path):
        self.root = root

    async def create(self, specification: MachineSpec) -> LocalDirectoryMachine:
        return LocalDirectoryMachine(self.root, specification)


class LocalDirectoryEnvironment(DockerWorkspaceEnvironment):
    """An actual Harbor environment using a test-only local machine factory."""

    def __init__(self, *args, local_root: str, **kwargs):
        super().__init__(*args, machine_factory=LocalDirectoryFactory(Path(local_root)), **kwargs)


class DockerEngineHandler(BaseHTTPRequestHandler):
    """Fake daemon serves host-side metadata and exactly one file archive."""

    root: Path
    archive_started: Event
    exec_started: Event
    archive_release: Event | None = None
    archive_fault: str | None = None

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler contract
        pass

    def _container(self) -> Path:
        identifier = urlsplit(self.path).path.split("/")[3]
        definition = json.loads((self.root / f"{identifier}.json").read_text())
        return self.root / definition["project"]

    def _metadata(self) -> tuple[Path, dict] | None:
        container = self._container()
        path = parse_qs(urlsplit(self.path).query)["path"][0]
        target = container / path.lstrip("/")
        try:
            info = target.lstat()
        except FileNotFoundError:
            return None
        mode = info.st_mode & 0o7777
        if stat.S_ISDIR(info.st_mode):
            mode |= 1 << 31
        elif stat.S_ISLNK(info.st_mode):
            mode |= 1 << 27
        elif stat.S_ISFIFO(info.st_mode):
            mode |= 1 << 25
        elif not stat.S_ISREG(info.st_mode):
            mode |= 1 << 19
        return target, {
            "name": target.name,
            "size": info.st_size,
            "mode": mode,
            "linkTarget": os.readlink(target) if target.is_symlink() else "",
        }

    def _respond(
        self, status: int, metadata: dict | None = None, payload: bytes = b"", encoding: str | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        if encoding is not None:
            self.send_header("Content-Encoding", encoding)
        if metadata is not None:
            self.send_header("X-Docker-Container-Path-Stat", base64.b64encode(json.dumps(metadata).encode()).decode())
        self.end_headers()
        if self.command != "HEAD":
            if self.command == "GET" and urlsplit(self.path).path.endswith("/archive"):
                self.archive_started.set()
                if self.archive_release is not None:
                    assert self.archive_release.wait(timeout=10), "CPU test did not release archive transfer"
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                # A cancelled client has closed its connection before the fake daemon writes.
                return

    def do_HEAD(self):
        entry = self._metadata()
        self._respond(404 if entry is None else 200, None if entry is None else entry[1])

    def do_GET(self):
        container = self._container()
        if urlsplit(self.path).path.endswith("/json"):
            state = {
                "Running": (container / "running").exists(),
                "Paused": (container / "paused").exists(),
                "Restarting": False,
            }
            if self.archive_fault == "state-mismatch":
                state["Running"] = not state["Running"]
            payload = json.dumps({"State": state}).encode()
            if self.archive_fault == "state-overflow":
                payload += b" " * (128 * 1024)
            self._respond(200, payload=payload)
            return
        if (container / "running").exists() and not (container / "paused").exists():
            self._respond(409)
            return
        entry = self._metadata()
        if entry is None:
            self._respond(404)
            return
        target, metadata = entry
        if self.archive_fault == "http-error":
            self._respond(500)
            return
        if self.archive_fault == "header-overflow":
            metadata["name"] = "a" * 10000
        with io.BytesIO() as payload:
            with tarfile.open(fileobj=payload, mode="w:") as archive:
                archive.add(target, arcname=target.name, recursive=False)
            content = payload.getvalue()
            if self.archive_fault == "archive-overflow":
                content += b"0" * (128 * 1024)
            encoding = None
            if self.archive_fault == "force-gzip" or "gzip" in self.headers.get("Accept-Encoding", ""):
                content = gzip.compress(content)
                encoding = "gzip"
            self._respond(200, metadata, content, encoding)

    def do_POST(self):
        if self.path == "/fixture-ready":
            self.exec_started.set()
            self._respond(204)
            return
        container = self._container()
        operation = urlsplit(self.path).path.rsplit("/", 1)[1]
        with (container / "collection-events.jsonl").open("a") as events:
            events.write(json.dumps({"operation": operation}) + "\n")
        if operation == "pause":
            (container / "paused").touch()
        elif operation == "unpause":
            if self.archive_fault == "unpause-error":
                self._respond(500)
                return
            (container / "paused").unlink(missing_ok=True)
        else:
            self._respond(404)
            return
        self._respond(204)


@pytest.fixture
def docker_boundary(tmp_path, monkeypatch):
    binary = tmp_path / "bin/docker"
    binary.parent.mkdir()
    fake = Path(__file__).parents[2] / "shellbox/tests/fixtures/docker_fake.py"
    binary.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake))} "$@"\n')
    binary.chmod(0o755)
    root = tmp_path / "containers"
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("TASKCOMPENDIUM_FAKE_DOCKER_ROOT", str(root))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    with tempfile.TemporaryDirectory(prefix="tc-engine-", dir="/tmp") as directory:
        socket = Path(directory) / "engine.sock"
        monkeypatch.setenv("TASKCOMPENDIUM_FAKE_DOCKER_SOCKET", str(socket))
        monkeypatch.setattr(DockerEngineHandler, "archive_started", Event(), raising=False)
        monkeypatch.setattr(DockerEngineHandler, "exec_started", Event(), raising=False)
        handler = type("BoundDockerEngineHandler", (DockerEngineHandler,), {"root": root})
        with UnixStreamServer(str(socket), handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield root
            finally:
                server.shutdown()
                thread.join()


@pytest.fixture
def specification():
    return TaskSpec(
        id="file-sum",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user", content="Read input.txt, add the two integers, and write only the result to answer.txt."
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            docker_image=IMAGE, working_directory="/workspace", capabilities=("filesystem", "shell")
        ),
        answer_type=AnswerType.NUMBER,
        verifier=exact_answer("12"),
        source=Source(dataset="synthetic", revision="1", row="file-sum", importer_revision="1"),
        resources=ResourceGroups(
            all=(
                TaskResource(
                    path="input.txt", source=InlineFile(content_base64=base64.b64encode(b"5 7\n").decode("ascii"))
                ),
            ),
            worker=(
                TaskResource(
                    path="worker.txt",
                    source=InlineFile(content_base64=base64.b64encode(b"worker-visible").decode("ascii")),
                ),
            ),
            verifier=(
                TaskResource(
                    path="private.txt",
                    source=InlineFile(content_base64=base64.b64encode(b"private-verifier-sentinel").decode("ascii")),
                ),
            ),
            oracle=(
                TaskResource(
                    path="private.txt",
                    source=InlineFile(content_base64=base64.b64encode(b"private-oracle-sentinel").decode("ascii")),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "operation,reward,status",
    [
        ("sum", 1.0, "graded"),
        ("wrong", 0.0, "graded"),
        ("missing", 0.0, "submission_failure"),
        ("symlink", 0.0, "submission_failure"),
        ("fifo", 0.0, "submission_failure"),
        ("workspace-symlink", 0.0, "submission_failure"),
        ("invalid-utf8", 0.0, "submission_failure"),
        ("oversize", 0.0, "submission_failure"),
    ],
)
async def test_harbor_file_trial_grades_without_transcript_and_keeps_private_inputs_host_side(
    tmp_path, docker_boundary, specification, operation, reward, status
):
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="text-file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    commands = {
        "sum": 'read first second < input.txt; printf %s "$((first + second))" > answer.txt',
        "wrong": "printf 13 > answer.txt",
        "missing": "true",
        "symlink": "ln -s input.txt answer.txt",
        "fifo": "mkfifo answer.txt",
        "workspace-symlink": "cd ..; mv workspace real-workspace; ln -s real-workspace workspace",
        "invalid-utf8": 'printf "\\377" > answer.txt',
        "oversize": 'python3 -c \'from pathlib import Path; Path("answer.txt").write_text("a" * 1025)\'',
    }
    agent = AgentConfig(
        import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": commands[operation]}, model_name="fixture-model"
    )
    result = await run_trial(task, config, agent, tmp_path / "trials", "run", trial_timeout=30)
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    assert result.config.agent == agent
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert not (tmp_path / "trials/run/agent/submission.json").exists()
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    containers = list(docker_boundary.glob("closed/*/workspace"))
    assert len(containers) == 1
    assert not (containers[0].parent / "paused").exists()
    collection = [
        json.loads(line) for line in (containers[0].parent / "collection-events.jsonl").read_text().splitlines()
    ]
    assert collection == [{"operation": "pause"}, {"operation": "unpause"}]
    assert (containers[0] / "worker.txt").read_text() == "worker-visible"
    assert not (containers[0] / "private.txt").exists()
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    assert not any(
        "private-" in path.read_text(errors="replace")
        for path in containers[0].iterdir()
        if path.is_file() and not path.is_symlink()
    )


@pytest.mark.parametrize(
    "value,reward,status",
    [('{"count":12}', 1.0, "graded"), ('{"count":13}', 0.0, "graded"), ("NaN", 0.0, "submission_failure")],
)
async def test_harbor_json_file_uses_private_structured_grader(
    tmp_path, docker_boundary, specification, value, reward, status
):
    specification = specification.model_copy(
        update={"answer_type": AnswerType.STATE, "verifier": structured_exact({"count": 12})}
    )
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(specification, JsonFile(id="json-file", path="answer.json"), config, tmp_path / "task")
    agent = AgentConfig(
        import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": f"printf %s {shlex.quote(value)} > answer.json"}
    )
    result = await run_trial(task, config, agent, tmp_path / "trials", "run")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    assert json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())["status"] == status


@pytest.mark.parametrize("tamper", ["image", "undeclared_file", "symlink", "mtime"])
async def test_harbor_docker_tampered_package_fails_before_runtime(tmp_path, specification, tamper):
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    if tamper == "image":
        path = task / "task.toml"
        path.write_text(path.read_text().replace("allow_internet = false", "allow_internet = true"))
    elif tamper == "undeclared_file":
        (task / "environment/Dockerfile").write_text("FROM scratch\n")
    elif tamper == "mtime":
        path = task / "specification.json"
        wire = json.loads(path.read_text())
        wire["resources"]["all"][0]["mtime_ns"] = 1700000000123456789
        path.write_text(json.dumps(wire))
    else:
        (task / "environment/escape").symlink_to(task / "specification.json")
    with pytest.raises((ValueError, NotImplementedError)):
        await run_trial(task, config, AgentConfig(name="nop"), tmp_path / "trials", "run")
    assert not (tmp_path / "trials").exists()


@pytest.mark.parametrize(
    "feature",
    ["capability", "setup", "environment", "provider", "private-runtime", "mtime", "final-tools", "workspace-state"],
)
def test_unsupported_semantics_have_no_file_lowering_or_partial_export(tmp_path, specification, feature):
    wire = specification.model_dump(mode="json")
    if feature == "capability":
        wire["environment_requirements"]["capabilities"].append("network")
    elif feature == "setup":
        wire["environment_requirements"]["setup_commands"] = ["touch setup.txt"]
    elif feature == "environment":
        wire["environment_requirements"]["environment_variables"] = {"TASK_MODE": "required"}
    elif feature == "provider":
        wire["environment_requirements"]["tool_providers"] = {
            "provider": {"action_interface": "v1", "initial_state": {}}
        }
    elif feature == "private-runtime":
        wire["verifier"]["environment_requirements"]["working_directory"] = "/private-grader"
    elif feature == "mtime":
        wire["resources"]["worker"][0]["mtime_ns"] = 1700000000123456789
    elif feature == "final-tools":
        wire["final_tools"] = [{"name": "required-action", "parameters": {"type": "object"}}]
    else:
        wire["answer_type"] = "workspace_state"
    task = TaskSpec.model_validate(wire)
    convention = TextFile(id="file", path="answer.txt")
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    assert compatible_lowerings(task, (convention,), (config,)) == ()
    with pytest.raises((NotImplementedError, ValueError)):
        lower_to_harbor(task, convention, config, tmp_path / "task")
    assert not (tmp_path / "task").exists()


async def test_harbor_docker_logstuff_workspace_collects_without_worker_python(
    tmp_path, docker_boundary, specification, monkeypatch
):
    monkeypatch.setenv("TASKCOMPENDIUM_FAKE_NO_WORKER_PYTHON", "1")
    requirements = specification.environment_requirements.model_copy(update={"working_directory": "/logstuff"})
    specification = specification.model_copy(update={"environment_requirements": requirements})
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": "printf 12 > answer.txt"})
    result = await run_trial(task, config, agent, tmp_path / "trials", "run")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert next(docker_boundary.glob("closed/*/logstuff/answer.txt")).read_text() == "12"


async def test_harbor_nonroot_image_can_use_restrictive_public_resources(
    tmp_path, docker_boundary, specification, monkeypatch
):
    monkeypatch.setenv("TASKCOMPENDIUM_FAKE_IMAGE_UID", "1000")
    monkeypatch.setenv("TASKCOMPENDIUM_FAKE_IMAGE_GID", "1001")
    specification = specification.model_copy(
        update={
            "resources": ResourceGroups(
                all=(
                    TaskResource(
                        path="input.txt",
                        source=InlineFile(content_base64=base64.b64encode(b"5 7\n").decode("ascii")),
                        mode="0400",
                    ),
                ),
                worker=(
                    TaskResource(
                        path="scripts/solve.sh",
                        source=InlineFile(
                            content_base64=base64.b64encode(
                                b'#!/bin/bash\nread first second < input.txt\nprintf %s "$((first + second))"\n'
                            ).decode("ascii")
                        ),
                        mode="0500",
                    ),
                ),
            )
        }
    )
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    agent = AgentConfig(
        import_path=f"{__name__}:WorkspaceAgent",
        kwargs={"command": "test -r input.txt && test ! -w input.txt && ./scripts/solve.sh > answer.txt"},
    )
    result = await run_trial(task, config, agent, tmp_path / "trials", "run")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    workspace = next(docker_boundary.glob("closed/*/workspace"))
    assert (workspace / "answer.txt").read_text() == "12"
    assert stat.S_IMODE((workspace / "input.txt").stat().st_mode) == 0o400
    assert stat.S_IMODE((workspace / "scripts/solve.sh").stat().st_mode) == 0o500


@pytest.mark.parametrize("mode", ["0000", "0100"])
def test_harbor_unreadable_public_resource_mode_fails_before_export(tmp_path, specification, mode):
    specification = specification.model_copy(
        update={
            "resources": ResourceGroups(
                worker=(
                    TaskResource(
                        path="input.txt",
                        source=InlineFile(content_base64=base64.b64encode(b"5 7\n").decode("ascii")),
                        mode=mode,
                    ),
                )
            )
        }
    )
    with pytest.raises(NotImplementedError):
        lower_to_harbor(
            specification,
            TextFile(id="file", path="answer.txt"),
            HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize(
    "fault,status,reward",
    [
        ("http-error", "infra_error", None),
        ("unpause-error", "infra_error", None),
        ("state-overflow", "infra_error", None),
        ("state-mismatch", "infra_error", None),
        ("header-overflow", "infra_error", None),
        ("force-gzip", "infra_error", None),
        ("archive-overflow", "submission_failure", 0.0),
    ],
)
async def test_harbor_archive_fault_unpauses_and_cleans_trial(
    tmp_path, docker_boundary, specification, monkeypatch, fault, status, reward
):
    monkeypatch.setattr(DockerEngineHandler, "archive_fault", fault)
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": "printf 12 > answer.txt"})
    result = await run_trial(task, config, agent, tmp_path / "trials", "run")
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
        assert result.exception_info is not None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    events = [json.loads(line) for line in (docker_boundary / "events.jsonl").read_text().splitlines()]
    assert events[-1]["command"][0] == "rm"


async def test_harbor_cancelled_archive_unpauses_and_cleans_trial(tmp_path, docker_boundary, specification, monkeypatch):
    release = Event()
    monkeypatch.setattr(DockerEngineHandler, "archive_release", release)
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": "printf 12 > answer.txt"})
    attempt = asyncio.create_task(run_trial(task, config, agent, tmp_path / "trials", "run"))
    try:
        assert await asyncio.to_thread(DockerEngineHandler.archive_started.wait, 10)
        attempt.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    events = [json.loads(line) for line in (docker_boundary / "events.jsonl").read_text().splitlines()]
    assert events[-1]["command"][0] == "rm"


@pytest.mark.parametrize(
    "shape", ["nested-workdir", "nested-result", "log-workdir", "root-alias-workdir", "log-alias-workdir"]
)
def test_harbor_unsupported_file_paths_fail_before_export(tmp_path, specification, shape):
    if shape == "nested-workdir":
        requirements = specification.environment_requirements.model_copy(
            update={"working_directory": "/tasks/workspace"}
        )
        specification = specification.model_copy(update={"environment_requirements": requirements})
    elif shape == "log-workdir":
        requirements = specification.environment_requirements.model_copy(update={"working_directory": "/logs/agent"})
        specification = specification.model_copy(update={"environment_requirements": requirements})
    elif shape in {"root-alias-workdir", "log-alias-workdir"}:
        workdir = "/.." if shape == "root-alias-workdir" else "//logs"
        requirements = specification.environment_requirements.model_copy(update={"working_directory": workdir})
        specification = specification.model_copy(update={"environment_requirements": requirements})
    path = "results/answer.txt" if shape == "nested-result" else "answer.txt"
    with pytest.raises((NotImplementedError, ValueError)):
        lower_to_harbor(
            specification,
            TextFile(id="file", path=path),
            HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


async def test_harbor_remote_transport_fails_before_trial_creation(tmp_path, specification, monkeypatch):
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(
        specification, TextFile(id="file", path="answer.txt", max_bytes=1024), config, tmp_path / "task"
    )
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "tcp://example.invalid:2375")
    with pytest.raises(NotImplementedError):
        await run_trial(task, config, AgentConfig(name="nop"), tmp_path / "trials", "run")
    assert not (tmp_path / "trials").exists()


@pytest.mark.parametrize("finish", ["timeout", "cancel", "stop-error"])
async def test_harbor_interrupted_agent_retains_diagnostics_before_explicit_close(
    tmp_path, docker_boundary, specification, monkeypatch, finish
):
    if finish == "stop-error":
        monkeypatch.setenv("TASKCOMPENDIUM_FAKE_STOP_FAILURE", "1")
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(specification, TextFile(id="file", path="answer.txt"), config, tmp_path / "task")
    agent = AgentConfig(
        import_path=f"{__name__}:WorkspaceAgent",
        kwargs={"command": "fixture-block"},
        override_timeout_sec=2,
    )
    pending = asyncio.create_task(run_trial(task, config, agent, tmp_path / "trials", "run"))
    assert await asyncio.to_thread(DockerEngineHandler.exec_started.wait, 10)
    if finish == "cancel":
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        result = json.loads((tmp_path / "trials/run/result.json").read_text())
        assert result["exception_info"]["exception_type"] == "CancelledError"
        assert result["verifier_result"] is None
    else:
        result = await pending
        assert result.exception_info is not None
        if finish == "timeout":
            assert result.exception_info.exception_type == "AgentTimeoutError"
            assert result.verifier_result.rewards == {"reward": 1.0}
        else:
            assert result.verifier_result is None
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    workspace = next(docker_boundary.glob("closed/*/workspace"))
    assert (workspace / "answer.txt").read_text() == "12"
    assert not (workspace / "private.txt").exists()
    if finish == "timeout":
        assert not (workspace.parent / "running").exists()
        assert not (workspace.parent / "collection-events.jsonl").exists()


@pytest.mark.parametrize("finish", ["failure", "cancel"])
async def test_harbor_failed_or_cancelled_start_removes_worker_before_agent(
    tmp_path, docker_boundary, specification, monkeypatch, finish
):
    monkeypatch.setenv(
        "TASKCOMPENDIUM_FAKE_START_FAILURE" if finish == "failure" else "TASKCOMPENDIUM_FAKE_START_BLOCK", "1"
    )
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(specification, TextFile(id="file", path="answer.txt"), config, tmp_path / "task")
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": "printf 12 > answer.txt"})
    pending = asyncio.create_task(run_trial(task, config, agent, tmp_path / "trials", "run"))
    if finish == "cancel":
        assert await asyncio.to_thread(DockerEngineHandler.exec_started.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        result = await pending
        assert result.exception_info is not None
        assert result.verifier_result is None
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    workspace = next(docker_boundary.glob("closed/*/workspace"))
    assert not (workspace / "answer.txt").exists()
    assert not (workspace.parent / "logs/agent/worker.log").exists()
    assert not (tmp_path / "trials/run/verifier/taskcompendium-result.json").exists()


@pytest.mark.parametrize("finish", ["graded", "error", "cancel"])
async def test_actual_harbor_trial_uses_local_factory_and_retains_host_outputs(
    tmp_path, specification, monkeypatch, finish
):
    binary = tmp_path / "bin/docker"
    binary.parent.mkdir()
    unexpected_docker = tmp_path / "unexpected-docker"
    binary.write_text(f"#!/bin/sh\nprintf unexpected > {shlex.quote(str(unexpected_docker))}\nexit 1\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    started = asyncio.Event()
    monkeypatch.setattr(LocalDirectoryMachine, "started", started, raising=False)
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(specification, TextFile(id="file", path="answer.txt"), config, tmp_path / "task")
    commands = {
        "graded": 'test ! -e private.txt; read first second < input.txt; printf %s "$((first + second))" > answer.txt',
        "error": "exit 42",
        "cancel": "fixture-block",
    }
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": commands[finish]})
    trial = await Trial.create(
        TrialConfig.model_validate(
            {
                "task": {"path": str(task)},
                "trials_dir": str(tmp_path / "trials"),
                "trial_name": "run",
                "environment": {
                    "import_path": f"{__name__}:LocalDirectoryEnvironment",
                    "kwargs": {"local_root": str(tmp_path / "machine")},
                },
                "agent": agent.model_dump(),
                "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
            }
        )
    )
    pending = asyncio.create_task(trial.run())
    if finish == "cancel":
        await asyncio.wait_for(started.wait(), timeout=10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        result = await pending
        if finish == "graded":
            assert result.exception_info is None, result.exception_info
            assert result.verifier_result.rewards == {"reward": 1.0}
        else:
            assert result.exception_info is not None
            assert result.verifier_result is None
    assert not unexpected_docker.exists()
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    assert not (tmp_path / "machine").exists()
    paths = json.loads((tmp_path / "closed-paths.json").read_text())
    assert "workspace/input.txt" in paths
    assert "workspace/worker.txt" in paths
    assert "workspace/private.txt" not in paths
    assert not any(path.startswith("logs/verifier") for path in paths)
    assert (tmp_path / "trials/run/result.json").exists()


@pytest.mark.parametrize("failure", ["image-workspace-symlink", "diagnostic-directory"])
async def test_harbor_invalid_startup_stops_before_agent_and_preserves_image_target(
    tmp_path, docker_boundary, specification, monkeypatch, failure
):
    monkeypatch.setenv(
        (
            "TASKCOMPENDIUM_FAKE_IMAGE_WORKSPACE_SYMLINK"
            if failure == "image-workspace-symlink"
            else "TASKCOMPENDIUM_FAKE_LOG_FAILURE"
        ),
        "1",
    )
    config = HarborEnvironmentConfig(environment=DOCKER_ENVIRONMENT)
    task = lower_to_harbor(specification, TextFile(id="file", path="answer.txt"), config, tmp_path / "task")
    agent = AgentConfig(import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": "printf 12 > answer.txt"})
    result = await run_trial(task, config, agent, tmp_path / "trials", "run")
    assert result.exception_info is not None
    assert result.verifier_result is None
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "logs/agent/worker.log").exists()
    assert not (closed / "workspace/answer.txt").exists()
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
    if failure == "image-workspace-symlink":
        assert [path.name for path in (closed / "image-target").iterdir()] == ["sentinel"]
        assert (closed / "image-target/sentinel").read_text() == "image-original"
        assert not (closed / "owners.json").exists()
    assert not (tmp_path / "trials/run/verifier/taskcompendium-result.json").exists()
