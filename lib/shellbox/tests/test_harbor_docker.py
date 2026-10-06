# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor lifecycle and bounded terminal collection over CPU machine boundaries."""

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

try:
    from harbor.agents.base import BaseAgent
    from harbor.models.trial.config import AgentConfig, TrialConfig
    from harbor.models.verifier.result import VerifierResult
    from harbor.trial.trial import Trial
    from harbor.verifier.base import BaseVerifier
except ModuleNotFoundError as error:
    if error.name != "harbor":
        raise
    pytest.skip("Harbor optional dependency is unavailable", allow_module_level=True)
from shellbox.backends.docker.environment import DockerEnvironment
from shellbox.backends.docker.terminal import DockerControlPlane
from shellbox.machine import Command, ExitReason, InvalidWorkspaceFile, MachineSpec, Result, TerminalFileReader

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


class LocalDirectoryEnvironment(DockerEnvironment):
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
    fake = Path(__file__).parent / "fixtures/docker_fake.py"
    binary.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake))} "$@"\n')
    binary.chmod(0o755)
    root = tmp_path / "containers"
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("SHELLBOX_FAKE_DOCKER_ROOT", str(root))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    with tempfile.TemporaryDirectory(prefix="shellbox-engine-", dir="/tmp") as directory:
        socket = Path(directory) / "engine.sock"
        monkeypatch.setenv("SHELLBOX_FAKE_DOCKER_SOCKET", str(socket))
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


class FileVerifier(BaseVerifier):
    """Test-only host verifier reads a bounded candidate through TerminalFileReader."""

    async def verify(self) -> VerifierResult:
        machine = self.environment.machine
        assert isinstance(machine, TerminalFileReader)
        try:
            candidate = await machine.read_file("answer.txt", 1024)
        except InvalidWorkspaceFile:
            candidate = None
        return VerifierResult(rewards={"reward": float(candidate == b"12")})


async def trial_for(
    tmp_path, command, *, local=False, workdir="/workspace", timeout=None, input_mode=0o644, public_symlink=False
):
    task = tmp_path / "task"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (task / "instruction.md").write_text("Read input.txt and write the sum to answer.txt.")
    (environment / "input.txt").write_text("5 7\n")
    (environment / "input.txt").chmod(input_mode)
    (environment / "worker.txt").write_text("worker-visible")
    tests = task / "tests"
    tests.mkdir()
    (tests / "private.txt").write_text("host-private-sentinel")
    if public_symlink:
        (environment / "input.txt").unlink()
        (environment / "input.txt").symlink_to(tests / "private.txt")
    (task / "task.toml").write_text(
        f'version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\nworkdir = "{workdir}"\nallow_internet = false\n'
    )
    if local:
        environment_config = {
            "import_path": f"{__name__}:LocalDirectoryEnvironment",
            "kwargs": {"local_root": str(tmp_path / "machine")},
        }
    else:
        control = DockerControlPlane(os.environ["SHELLBOX_FAKE_DOCKER_SOCKET"], "1.51")
        environment_config = {
            "import_path": "shellbox.backends.docker.environment:DockerEnvironment",
            "kwargs": {"archive_socket": control.socket, "archive_api_version": control.api_version},
        }
    agent = AgentConfig(
        import_path=f"{__name__}:WorkspaceAgent", kwargs={"command": command}, override_timeout_sec=timeout
    )
    return await Trial.create(
        TrialConfig.model_validate(
            {
                "task": {"path": str(task)},
                "trials_dir": str(tmp_path / "trials"),
                "trial_name": "run",
                "environment": environment_config,
                "agent": agent.model_dump(),
                "verifier": {"import_path": f"{__name__}:FileVerifier"},
            }
        )
    )


@pytest.mark.parametrize(
    "operation,reward",
    [
        ("sum", 1.0),
        ("wrong", 0.0),
        ("missing", 0.0),
        ("symlink", 0.0),
        ("fifo", 0.0),
        ("workspace-symlink", 0.0),
        ("oversize", 0.0),
    ],
)
async def test_docker_trial_reads_bounded_file_and_keeps_private_inputs_host_side(
    tmp_path, docker_boundary, operation, reward
):
    commands = {
        "sum": 'read first second < input.txt; printf %s "$((first + second))" > answer.txt',
        "wrong": "printf 13 > answer.txt",
        "missing": "true",
        "symlink": "ln -s input.txt answer.txt",
        "fifo": "mkfifo answer.txt",
        "workspace-symlink": "cd ..; mv workspace real-workspace; ln -s real-workspace workspace",
        "oversize": 'python3 -c \'from pathlib import Path; Path("answer.txt").write_text("a" * 1025)\'',
    }
    trial = await trial_for(tmp_path, commands[operation])
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "paused").exists()
    events = [json.loads(line) for line in (closed / "collection-events.jsonl").read_text().splitlines()]
    assert events == [{"operation": "pause"}, {"operation": "unpause"}]
    assert not list(closed.rglob("private.txt"))
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]


@pytest.mark.parametrize(
    "fault", ["http-error", "unpause-error", "state-overflow", "state-mismatch", "header-overflow", "force-gzip"]
)
async def test_docker_protocol_failure_is_ungraded_and_closes_machine(tmp_path, docker_boundary, monkeypatch, fault):
    monkeypatch.setattr(DockerEngineHandler, "archive_fault", fault)
    trial = await trial_for(tmp_path, "printf 12 > answer.txt")
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info is not None
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]


async def test_docker_archive_cancellation_unpauses_and_closes_machine(tmp_path, docker_boundary, monkeypatch):
    release = Event()
    monkeypatch.setattr(DockerEngineHandler, "archive_release", release)
    trial = await trial_for(tmp_path, "printf 12 > answer.txt")
    attempt = asyncio.create_task(trial.run())
    try:
        assert await asyncio.to_thread(DockerEngineHandler.archive_started.wait, 10)
        attempt.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "paused").exists()
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]


@pytest.mark.parametrize("finish", ["timeout", "cancel", "stop-error"])
async def test_interrupted_command_retains_diagnostics_until_close(tmp_path, docker_boundary, monkeypatch, finish):
    if finish == "stop-error":
        monkeypatch.setenv("SHELLBOX_FAKE_STOP_FAILURE", "1")
    trial = await trial_for(tmp_path, "fixture-block", timeout=2)
    pending = asyncio.create_task(trial.run())
    assert await asyncio.to_thread(DockerEngineHandler.exec_started.wait, 10)
    if finish == "cancel":
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        result = await pending
        assert result.exception_info is not None
        if finish == "timeout":
            assert result.verifier_result.rewards == {"reward": 1.0}
        else:
            assert result.verifier_result is None
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    closed = next(docker_boundary.glob("closed/*"))
    assert (closed / "workspace/answer.txt").read_text() == "12"
    assert not list(closed.rglob("private.txt"))
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]


@pytest.mark.parametrize("finish", ["graded", "error", "cancel"])
async def test_harbor_trial_uses_local_directory_machine_without_docker(tmp_path, monkeypatch, finish):
    binary = tmp_path / "bin/docker"
    binary.parent.mkdir()
    unexpected = tmp_path / "unexpected-docker"
    binary.write_text(f"#!/bin/sh\nprintf unexpected > {shlex.quote(str(unexpected))}\nexit 1\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    started = asyncio.Event()
    monkeypatch.setattr(LocalDirectoryMachine, "started", started, raising=False)
    commands = {
        "graded": 'read a b < input.txt; printf %s "$((a + b))" > answer.txt',
        "error": "exit 42",
        "cancel": "fixture-block",
    }
    trial = await trial_for(tmp_path, commands[finish], local=True)
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
    assert not unexpected.exists()
    assert (tmp_path / "trials/run/agent/worker.log").read_text() == "worker-diagnostic"
    assert (tmp_path / "trials/run/artifacts/worker.txt").read_text() == "worker-artifact"
    assert not (tmp_path / "machine").exists()
    paths = json.loads((tmp_path / "closed-paths.json").read_text())
    assert "workspace/input.txt" in paths
    assert "workspace/worker.txt" in paths
    assert not any(path.startswith("logs/verifier") or path.endswith("private.txt") for path in paths)


@pytest.mark.parametrize("failure", ["image-workspace-symlink", "diagnostic-directory"])
async def test_failed_startup_preserves_image_target_and_stops_before_agent(
    tmp_path, docker_boundary, monkeypatch, failure
):
    monkeypatch.setenv(
        "SHELLBOX_FAKE_IMAGE_WORKSPACE_SYMLINK" if failure == "image-workspace-symlink" else "SHELLBOX_FAKE_LOG_FAILURE",
        "1",
    )
    trial = await trial_for(tmp_path, "printf 12 > answer.txt")
    result = await trial.run()
    assert result.exception_info is not None
    assert result.verifier_result is None
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "logs/agent/worker.log").exists()
    assert not (closed / "workspace/answer.txt").exists()
    if failure == "image-workspace-symlink":
        assert [path.name for path in (closed / "image-target").iterdir()] == ["sentinel"]
        assert (closed / "image-target/sentinel").read_text() == "image-original"
        assert not (closed / "owners.json").exists()


async def test_nonroot_image_reads_restrictive_public_file_without_widening_mode(tmp_path, docker_boundary, monkeypatch):
    monkeypatch.setenv("SHELLBOX_FAKE_IMAGE_UID", "1000")
    monkeypatch.setenv("SHELLBOX_FAKE_IMAGE_GID", "1001")
    trial = await trial_for(
        tmp_path,
        'test -r input.txt && test ! -w input.txt && read a b < input.txt && printf %s "$((a + b))" > answer.txt',
        input_mode=0o400,
    )
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    workspace = next(docker_boundary.glob("closed/*/workspace"))
    assert stat.S_IMODE((workspace / "input.txt").stat().st_mode) == 0o400


@pytest.mark.parametrize("configuration", [{"public_symlink": True}, {"workdir": "/.."}])
async def test_unsafe_workspace_is_rejected_before_machine_creation(tmp_path, docker_boundary, configuration):
    with pytest.raises(ValueError):
        await trial_for(tmp_path, "cat input.txt > answer.txt", **configuration)
    assert not list(docker_boundary.glob("harbor-machine-*"))
    assert not (docker_boundary / "events.jsonl").exists()


@pytest.mark.parametrize("finish", ["failure", "cancel"])
async def test_failed_or_cancelled_create_removes_machine_before_agent(tmp_path, docker_boundary, monkeypatch, finish):
    monkeypatch.setenv("SHELLBOX_FAKE_START_FAILURE" if finish == "failure" else "SHELLBOX_FAKE_START_BLOCK", "1")
    trial = await trial_for(tmp_path, "printf 12 > answer.txt")
    pending = asyncio.create_task(trial.run())
    if finish == "cancel":
        assert await asyncio.to_thread(DockerEngineHandler.exec_started.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        result = await pending
        assert result.exception_info is not None
        assert result.verifier_result is None
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "workspace/answer.txt").exists()
    assert not (closed / "logs/agent/worker.log").exists()
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]


async def test_oversized_archive_is_candidate_failure_and_releases_machine(tmp_path, docker_boundary, monkeypatch):
    monkeypatch.setattr(DockerEngineHandler, "archive_fault", "archive-overflow")
    trial = await trial_for(tmp_path, "printf 12 > answer.txt")
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 0.0}
    closed = next(docker_boundary.glob("closed/*"))
    assert not (closed / "paused").exists()
    assert not [path for path in docker_boundary.glob("harbor-machine-*") if path.is_dir()]
