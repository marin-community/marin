"""Real subprocess execution at the Machine boundary for trusted test programs.

This fixture has no container isolation. Container security requires the Docker tests.
"""

import asyncio
import json
import os
import shutil
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import pytest_asyncio
import psutil
from rolloutengine.models import ModelTurn, MachineRuntimeSpec
from rolloutengine.engine import RolloutEngine
from shellbox.machine import Backend, Command, ExitReason, Result
from skyrl_gym.rollout_specs import rollout_spec
from skyrl_gym.source_task import source_task
from taskcompendium.models import CommandSemantics, Source

from skyrl_gym.task_factories import episode_factories, episode_harnesses
from skyrl_gym.nemotron_tasks import NemotronTaskSession


@pytest.fixture
def task_lowering():
    def make(task, name, *, max_turns=3):
        # The process fixture substitutes for native image execution, without isolation.
        task = task.model_copy(
            update={
                "environment_requirements": task.environment_requirements.model_copy(
                    update={
                        "command_semantics": CommandSemantics.LINUX_PROCESS,
                        "docker_image": "fixture@sha256:" + "0" * 64,
                    }
                )
            }
        )
        return rollout_spec(
            task,
            {
                "task_session": name,
                "max_turns": max_turns,
                "model_turn_timeout": None,
                "command_timeout": 1,
                "tool_turn_timeout": 5,
                "total_turn_timeout": None,
                "attempt_timeout": None,
                "verifier_timeout": 5,
                "cleanup_timeout": 5,
                "max_total_tokens": None,
            },
            task_machine=MachineRuntimeSpec(
                backend="local", network="deny", cpus=None, memory_mb=None, storage_mb=None,
                gpus=0, user=None, startup_timeout=None, cleanup_timeout=None,
            ),
        )

    return make


@pytest.fixture
def model_turn():
    def make(text, *, stop_reason="stop", token_count=2, metadata=None, message=None):
        return ModelTurn(
            {"role": "assistant", "content": text} if message is None else message,
            (10, 11),
            tuple(range(20, 20 + token_count)),
            (-0.5,) * token_count,
            stop_reason,
            text=text,
            metadata={} if metadata is None else metadata,
        )

    return make


@pytest.fixture
def python_tool_turn(model_turn):
    def make(code, *, arguments=None):
        return model_turn(
            "",
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "stateful_python_code_exec",
                            "arguments": json.dumps({"code": code}) if arguments is None else arguments,
                        },
                    }
                ],
            },
        )

    return make


@pytest.fixture
def retrieval_service():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            query = request["query"]
            status = 400 if query == "rejected" else 500 if query == "retry" and len(requests) == 1 else 200
            result = [] if query == "missing" else [[{"document": {"contents": "Paris is the capital of France."}}]]
            body = b"not json" if query == "malformed" else json.dumps({"result": result}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/retrieve", requests
        finally:
            server.shutdown()
            thread.join()


class LocalProcessMachine:
    def __init__(self, root: Path):
        self.root = root
        self.closed = False
        (root / "workspace").mkdir()

    def path(self, value: str) -> Path:
        if Path(value).is_relative_to(self.root):
            return Path(value)
        return self.root / value.lstrip("/")

    async def run(self, command: Command) -> Result:
        if self.closed:
            raise RuntimeError("Machine is closed")
        argv = [str(self.path(value)) if value.startswith("/") else value for value in command.argv]
        if argv[0] == "python":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.path(command.cwd or "/workspace"),
            env={**os.environ, **command.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(command.stdin), timeout=command.timeout)
        except (asyncio.CancelledError, TimeoutError):
            descendants = psutil.Process(process.pid).children(recursive=True)
            for descendant in descendants:
                try:
                    descendant.kill()
                except psutil.NoSuchProcess:
                    pass
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            await self.close()
            raise
        limit = command.output_limit_bytes
        return Result(
            process.returncode,
            stdout[:limit],
            stderr[:limit],
            len(stdout) > limit,
            len(stderr) > limit,
            ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        destination = self.path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    async def download(self, source: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path(source), target)

    async def close(self) -> None:
        if self.closed:
            return
        for file in (self.root / "tmp").glob("skyrl-*/pid"):
            try:
                os.killpg(int(file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.closed = True


@pytest_asyncio.fixture
async def machine():
    # macOS's default temporary root exceeds the Unix-domain socket path limit.
    with TemporaryDirectory(prefix="sb-test-", dir="/tmp") as directory:
        machine = LocalProcessMachine(Path(directory))
        try:
            yield machine
        finally:
            await machine.close()


@pytest_asyncio.fixture
async def nemotron_session(machine, task_lowering):
    sessions = []

    async def make(agent, record, config=None, request=None, *, environment=None):
        task = source_task(
            [{"role": "user", "content": "Public task question"}],
            {
                "extra_info": {
                    "nemotron_ultra": {
                        "route": "task_session",
                        "agent": agent,
                        "record_json": json.dumps(record),
                        "request_json": json.dumps({} if request is None else request),
                    }
                }
            },
            {} if config is None else config,
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            environment=environment,
        )
        session = NemotronTaskSession(task_lowering(task, "nemotron_ultra", max_turns=50), machine)
        sessions.append(session)
        await session.prepare()
        return session

    try:
        yield make
    finally:
        for session in sessions:
            await session.close()


@pytest.fixture
def lean_compiler(machine, monkeypatch):
    """Replace the external compiler process while retaining the real Shellbox command boundary."""
    project = machine.path("/lean/project")
    project.mkdir(parents=True)
    binary = machine.path("/bin/lake")
    binary.parent.mkdir()
    binary.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, re, signal, sys\n"
        "proof = pathlib.Path(sys.argv[-1]).read_text()\n"
        "if 'hang_compiler' in proof: signal.pause()\n"
        "if 'truncated_output' in proof: print('x' * 70000)\n"
        "if 'sorry' in proof: print('warning: declaration uses sorry')\n"
        "if re.search(r'(?m)^[ \\t]+(?:theorem|lemma|example)\\b', proof):\n"
        "    print('error: declaration is not a tactic'); sys.exit(1)\n"
        "if 'bad_tactic' in proof:\n"
        "    print('error: unknown tactic bad_tactic'); sys.exit(1)\n"
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    return str(project)


@pytest_asyncio.fixture
async def rollout_session(task_lowering):
    """Run scripted model responses through the public engine and real SQLite processes."""
    with TemporaryDirectory(prefix="sb-rollout-", dir="/tmp") as directory:
        machines = []

        class Factory:
            backend = Backend.DOCKER

            def check_environment(self, _requirements, _spec):
                # This local fixture supports the declared test environments.
                pass

            async def create(self, spec):
                root = Path(directory) / str(len(machines))
                root.mkdir()
                machine = LocalProcessMachine(root)
                machines.append(machine)
                return machine

        async def run(name, responses, extras, config=None, *, max_turns=3):
            responses = iter(responses)

            async def model(request):
                response = next(responses)
                message = {"role": "assistant", "content": response} if isinstance(response, str) else response
                text = message.get("content") or ""
                prefix = () if request.continuation is None else request.continuation.served_token_ids
                prompt = (*prefix, 90, 91) if prefix else (10, 11)
                return ModelTurn(message, prompt, (20, 21), (-0.5, -0.5), "stop", text=text)

            task = source_task(
                [{"role": "user", "content": "Public task question"}],
                extras,
                {} if config is None else config,
                Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            )
            spec = task_lowering(task, name, max_turns=max_turns)
            engine = RolloutEngine({"local": Factory()}, episode_harnesses(spec, episode_factories()))
            return await engine.run_task(spec, model)

        try:
            yield run
        finally:
            for machine in machines:
                await machine.close()
