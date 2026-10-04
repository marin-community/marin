# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare private repair tasks from bounded, pinned source snapshots."""

import argparse
import ast
import asyncio
import difflib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from rigging.filesystem.storage_path import prefix_join
from rolloutengine.grading import _grade_rollout
from rolloutengine.machines import _install_files, _task_machine
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory
from shellbox.machine import Command, ExitReason, MachineFactory
from taskcompendium.environment import (
    EnvironmentAsset,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    RegistryImage,
    ShellVerifierSpec,
)
from taskcompendium.grading import Outcome
from taskcompendium.importers.swe import PATCH_PATH, SWEInstance, swe_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.parquet import write_tasks
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.glm import GLM_MODEL, resolve_glm_base_url
from experiments.post_training.russell_rsi.corpus import CommitRecord
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import SourceSnapshot, source_group_id, source_path

WORKSPACE = "/workspace"
PRIVATE_CASES = "/tmp/taskcompendium/cases.json"
RUNNER_PATH = "/tmp/taskcompendium/runner.py"
GUARD_PATH = "/tmp/taskcompendium/guard.py"
RESULT_PREFIX = "RSI_RESULT="
WHEEL_ROOT = "/opt/rsi-wheels"
MAX_WHEEL_BYTES = 50_000_000
MAX_WHEEL_FILES = 50
MAX_GENERATION_BYTES = 256_000
DEPENDENCY_SETUP_TIMEOUT = 300


class InvalidRepair(ValueError):
    """Generated task code or source scope cannot enter acceptance."""


class ObservationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    probe_python: str = Field(min_length=1, max_length=12_000)
    expected_json: JsonValue


class GeneratedRepair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    problem_statement: str = Field(min_length=1)
    cases: tuple[ObservationCase, ...] = Field(min_length=1, max_length=8)
    editable_paths: tuple[str, ...] = Field(min_length=1)


PROBE_RUNNER = """import json, os, sys
probe = sys.stdin.read(12001)
if len(probe) > 12000:
    raise ValueError("Probe exceeds its character budget")
write, dumps = os.write, json.dumps
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
sys.stdout = sys.stderr
namespace = {"__name__": "__rsi_probe__"}
exec(compile(probe, "<private-probe>", "exec"), namespace)
payload = dumps(namespace["observation"], sort_keys=True, separators=(",", ":"), allow_nan=False)
write(1, payload.encode())
"""


RUNNER = (
    f"import json, os, pathlib, selectors, signal, subprocess, sys, time\nCHILD = {PROBE_RUNNER!r}\n"
    + """MAX_OUTPUT = 8192
CASE_TIMEOUT = 10
private = pathlib.Path(__file__).parent
if os.geteuid() != 0:
    raise RuntimeError("The grader supervisor requires root in its isolated machine")
os.chmod(private, 0o700)
cases = json.loads((private / "cases.json").read_text())

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def observation(probe):
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", CHILD], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/workspace",
        user=65534, group=65534, extra_groups=[], start_new_session=True, close_fds=True,
    )
    output, diagnostics = bytearray(), bytearray()
    error = None
    deadline = time.monotonic() + CASE_TIMEOUT
    try:
        process.stdin.write(probe.encode())
        process.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, output)
            selector.register(process.stderr, selectors.EVENT_READ, diagnostics)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = "timeout"
                    break
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    key.data.extend(chunk)
                    if len(output) + len(diagnostics) > MAX_OUTPUT:
                        error = "output_limit"
                        break
                if error:
                    break
        if error:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            error = "timeout"
        if error:
            return None, error, diagnostics[:1024].decode(errors="replace")
        if process.returncode != 0:
            return None, "execution_error", diagnostics[:1024].decode(errors="replace")
        try:
            value = json.loads(output)
            canonical(value)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return None, "invalid_observation", diagnostics[:1024].decode(errors="replace")
        return value, None, diagnostics[:1024].decode(errors="replace")
    finally:
        # No descendant may retain a grading output pipe after this case.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()
        process.stderr.close()

observations, case_errors, case_diagnostics = [], [], []
failures = errors = 0
for case in cases:
    value, error, diagnostic = observation(case["probe_python"])
    observations.append(value)
    case_errors.append(error)
    case_diagnostics.append(diagnostic)
    if error is not None:
        errors += 1
    elif canonical(value) != canonical(case["expected_json"]):
        failures += 1
metrics = {"tests": len(cases), "failures": failures, "errors": errors,
           "observations": observations, "case_errors": case_errors, "case_diagnostics": case_diagnostics}
print("RSI_RESULT=" + json.dumps(metrics, sort_keys=True, allow_nan=False))
sys.exit(0 if cases and failures == errors == 0 else 1)
"""
)


def validate_probe(code: str) -> None:
    """Reject scoring predicates and broad catches in generated observation code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise InvalidRepair(f"Private probe has invalid Python: {error}") from error
    assigned = False
    forbidden_catches = {
        "Exception",
        "BaseException",
        "ImportError",
        "ModuleNotFoundError",
        "SystemExit",
        "KeyboardInterrupt",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assert, ast.Compare)):
            raise InvalidRepair("Private probes must return raw observations, not scoring predicates")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] in ("unittest", "pytest") for name in names):
                raise InvalidRepair("Private probes cannot import test scorers")
        if isinstance(node, ast.Try) and node.handlers:
            if any(
                isinstance(item, (ast.Import, ast.ImportFrom)) for statement in node.body for item in ast.walk(statement)
            ):
                raise InvalidRepair("Private probes cannot catch import or setup errors")
            if not any(isinstance(item, ast.Call) for statement in node.body for item in ast.walk(statement)):
                raise InvalidRepair("Named exception catches must surround a source call")
        if isinstance(node, ast.ExceptHandler):
            catches = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if any(not isinstance(catch, (ast.Name, ast.Attribute)) for catch in catches):
                raise InvalidRepair("Private probes may catch only named source exceptions")
            if any((catch.id if isinstance(catch, ast.Name) else catch.attr) in forbidden_catches for catch in catches):
                raise InvalidRepair("Private probes cannot hide import, setup, or process errors")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned |= any(isinstance(target, ast.Name) and target.id == "observation" for target in targets)
    if not assigned:
        raise InvalidRepair("Private probe must assign its raw result to observation")


def build_task(
    snapshot: SourceSnapshot,
    repair: GeneratedRepair,
    *,
    image: str,
    timeout: float,
    dependency_wheels: Path | None = None,
    dependency_wheels_uri: str | None = None,
) -> TaskSpec:
    """Expose only the parent tree and keep the verifier in a fresh machine."""
    changed = {
        path
        for path in set(snapshot.parent_files) | set(snapshot.reference_files)
        if snapshot.parent_files.get(path) != snapshot.reference_files.get(path)
    }
    allowed = tuple(sorted(source_path(path) for path in changed))
    if not allowed or any(path in snapshot.license_paths for path in allowed):
        raise InvalidRepair("Reference changes must be source files")
    if any(
        not path.endswith(".py")
        or "tests" in PurePosixPath(path).parts
        or PurePosixPath(path).name.startswith("test_")
        or PurePosixPath(path).name.endswith("_test.py")
        for path in allowed
    ):
        raise InvalidRepair("This seed accepts Python source edits only")
    if not set(repair.editable_paths) <= set(allowed):
        raise InvalidRepair("Generated editable paths are outside the source change scope")
    for case in repair.cases:
        validate_probe(case.probe_python)
        try:
            json.dumps(case.expected_json, allow_nan=False)
        except ValueError as error:
            raise InvalidRepair("Expected observation must be finite JSON") from error
    if snapshot.commit_sha in repair.problem_statement:
        raise InvalidRepair("Problem statement contains the reference commit")
    for path in changed:
        lines = difflib.ndiff(
            snapshot.parent_files.get(path, "").splitlines(), snapshot.reference_files.get(path, "").splitlines()
        )
        for line in lines:
            added = line[2:].strip()
            if line.startswith("+ ") and len(added) >= 24 and not added.startswith(("#", "def ", "class ", '"')):
                if added in repair.problem_statement:
                    raise InvalidRepair("Problem statement contains reference implementation text")
    wheels = []
    wheel_hashes = {}
    if dependency_wheels is not None:
        if dependency_wheels_uri is None:
            raise ValueError("Wheel dependencies require an immutable artifact URI")
        paths = sorted(dependency_wheels.iterdir())
        if not paths or len(paths) > MAX_WHEEL_FILES or sum(path.stat().st_size for path in paths) > MAX_WHEEL_BYTES:
            raise ValueError("Wheel bundle exceeds the dependency budget or is empty")
        for path in paths:
            if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
                raise ValueError("Dependency bundle requires regular wheel files")
            content = path.read_bytes()
            wheels.append(
                EnvironmentAsset(
                    path=f"{WHEEL_ROOT}/{path.name}",
                    uri=prefix_join(dependency_wheels_uri, path.name),
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                )
            )
            wheel_hashes[path.name] = wheels[-1].sha256
    wheel_setup = (
        ()
        if not wheels
        else (
            EnvironmentCommand(
                argv=(
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    WHEEL_ROOT,
                    *(wheel.path for wheel in wheels),
                ),
                timeout=DEPENDENCY_SETUP_TIMEOUT,
            ),
        )
    )
    environment = EnvironmentSpec(
        kind=EnvironmentKind.DOCKER,
        image=RegistryImage(reference=image),
        workdir=WORKSPACE,
        network=False,
        memory_mb=1024,
        cpus=1,
        files=tuple(
            EnvironmentFile(path=f"{WORKSPACE}/{path}", content=text.encode())
            for path, text in sorted(snapshot.parent_files.items())
        ),
        assets=tuple(wheels),
        setup=(
            *wheel_setup,
            EnvironmentCommand(
                argv=(
                    "sh",
                    "-c",
                    "git init -q && git add -A -f && "
                    "git -c user.name=task -c user.email=task@example.invalid commit -qm baseline",
                ),
                timeout=timeout,
            ),
        ),
    )
    guard = (
        "import pathlib, subprocess\n"
        f"allowed = set({allowed!r})\n"
        "subprocess.run(['git', 'add', '-A', '-f'], check=True)\n"
        "changed = subprocess.check_output(['git','diff','--cached','--name-only','-z','HEAD']).decode().split('\\0')\n"
        "assert set(filter(None, changed)) <= allowed, 'Changes outside editable source paths'\n"
        "modes = subprocess.check_output(['git','ls-files','-s']).decode().splitlines()\n"
        "assert all(line.startswith(('100644 ', '100755 ')) for line in modes), 'Nonregular file'\n"
    )
    guard += "assert not any(path.is_symlink() for path in pathlib.Path('/workspace').rglob('*')), 'Symlink in source'\n"
    eval_script = f"set -eu\npython -I {GUARD_PATH}\n" f"python -I {RUNNER_PATH}\n"
    task = swe_task(
        SWEInstance(
            instance_id="rsi-"
            + hashlib.sha256(f"{snapshot.repository}:{source_group_id(snapshot)}".encode()).hexdigest()[:24],
            problem_statement=repair.problem_statement + "\nEditable source paths: " + ", ".join(allowed) + ".",
            eval_script=eval_script,
            image_name=image,
        ),
        source=Source(
            dataset="russell-rsi",
            revision=snapshot.parent_sha,
            row=f"{snapshot.repository}:{snapshot.parent_sha}:{source_group_id(snapshot)}",
            importer_revision="russell-rsi-v2",
        ),
        environment=environment,
        verifier_timeout=timeout,
    )
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    verifier = verifier.model_copy(
        update={
            "collect": (
                EnvironmentCommand(
                    argv=(
                        "sh",
                        "-c",
                        'mkdir -p "$(dirname "$1")" && git add -A -f -- . '
                        "':!**/__pycache__/**' ':!*.pyc' && git diff --cached --binary > \"$1\"",
                        "collect-patch",
                        PATCH_PATH,
                    ),
                    timeout=timeout,
                ),
            ),
            "files": (
                *verifier.files,
                EnvironmentFile(
                    path=PRIVATE_CASES, content=json.dumps([case.model_dump() for case in repair.cases]).encode()
                ),
                EnvironmentFile(path=RUNNER_PATH, content=RUNNER.encode()),
                EnvironmentFile(path=GUARD_PATH, content=guard.encode()),
            ),
        }
    )
    return task.model_copy(
        update={
            "verifier": task.verifier.model_copy(update={"parameters_json": verifier.model_dump_json()}),
            "metadata": {
                "split": snapshot.split,
                "repository": snapshot.repository,
                "license_paths": list(snapshot.license_paths),
                "dependency_wheel_sha256": wheel_hashes,
            },
        }
    )


class VerifierReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    tests: int = Field(ge=1, le=8)
    failures: int = Field(ge=0)
    errors: int = Field(ge=0)
    observations: tuple[JsonValue, ...]
    case_errors: tuple[str | None, ...]
    case_diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    parent: VerifierReport | None
    reference: VerifierReport | None


async def run_verifier(
    files: dict[str, str], verifier: ShellVerifierSpec, *, environment: EnvironmentSpec, factory: MachineFactory
) -> VerifierReport | None:
    """Collect metrics with the same image and dependency setup as patch admission."""
    source_files = tuple(
        EnvironmentFile(path=f"{WORKSPACE}/{path}", content=text.encode()) for path, text in sorted(files.items())
    )
    trial_environment = environment.model_copy(update={"files": source_files})
    async with _task_machine(trial_environment, {EnvironmentKind.DOCKER: factory}) as machine:
        assert machine is not None
        await _install_files(machine, verifier.files)
        result = await machine.run(Command(argv=("python", "-I", RUNNER_PATH), cwd=WORKSPACE, timeout=verifier.timeout))
        if result.reason != ExitReason.EXITED or result.exit_code not in (0, 1):
            return None
        lines = [line for line in result.stdout.decode(errors="replace").splitlines() if line.startswith(RESULT_PREFIX)]
        if len(lines) != 1:
            return None
        try:
            report = VerifierReport.model_validate_json(lines[0][len(RESULT_PREFIX) :])
        except ValidationError:
            return None
        return report


async def accept_candidate(task: TaskSpec, snapshot: SourceSnapshot, *, factory: MachineFactory) -> AcceptanceResult:
    """Accept a behavioral parent failure and a successful reference result."""
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    outcomes = []
    for files in (snapshot.parent_files, snapshot.reference_files):
        outcomes.append(
            [await run_verifier(files, verifier, environment=task.environment, factory=factory) for _ in range(2)]
        )
    parent, reference = outcomes[0][0], outcomes[1][0]
    stable = all(
        first is not None
        and second is not None
        and first.model_dump_json(exclude={"case_diagnostics"}) == second.model_dump_json(exclude={"case_diagnostics"})
        for first, second in outcomes
    )
    accepted = bool(
        stable
        and parent
        and reference
        and parent.failures > 0
        and parent.errors == 0
        and reference.tests == parent.tests
        and reference.failures == reference.errors == 0
    )
    return AcceptanceResult(accepted, parent, reference)


CONTROL_REWARDS = {
    "noop": 0.0,
    "reference": 1.0,
    "outside_source": 0.0,
    "source_exit": 0.0,
    "stdout_success": 0.0,
    "fd_forge": 0.0,
    "multiple_json": 0.0,
    "expected_file_read": 0.0,
    "unittest_monkeypatch": 0.0,
    "private_file_tamper": 0.0,
}
CONTROL_SOURCE = {
    "source_exit": "import os; os._exit(0)\n",
    "stdout_success": 'import os; print(\'RSI_RESULT={"tests":1,"failures":0,"errors":0}\', flush=True); os._exit(0)\n',
    "fd_forge": (
        "import os\nfor fd in range(1, 64):\n"
        '    try: os.write(fd, b\'{"tests":1,"failures":0,"errors":0}\')\n'
        "    except OSError: pass\nos._exit(0)\n"
    ),
    "multiple_json": "import os; os.write(1, b'null\\nnull'); os._exit(0)\n",
    "expected_file_read": (
        "import os\n"
        "with open('/tmp/taskcompendium/cases.json') as private_file: print(private_file.read())\n"
        "os._exit(0)\n"
    ),
    "unittest_monkeypatch": (
        "import unittest\n"
        "unittest.TestResult.wasSuccessful = lambda self: True\n"
        "unittest.TextTestRunner.run = lambda self, test: type('Fake', (), "
        "{'testsRun': 1, 'failures': [], 'errors': [], 'skipped': [], 'wasSuccessful': lambda self: True})()\n"
    ),
    "private_file_tamper": (
        "import os\n"
        "with open('/tmp/taskcompendium/cases.json', 'w') as private_file: private_file.write('[]')\n"
        "os._exit(0)\n"
    ),
}


def attacked_source(content: str, attack: str) -> str:
    tree = ast.parse(content)
    future_end = max(
        (
            node.end_lineno or node.lineno
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ),
        default=0,
    )
    lines = content.splitlines(keepends=True)
    lines.insert(future_end, attack)
    return "".join(lines)


async def patch_controls(task: TaskSpec, snapshot: SourceSnapshot, *, factory: MachineFactory) -> dict[str, dict]:
    """Replay repair and attack controls through patch collection and the fresh grader."""
    factories = {EnvironmentKind.DOCKER: factory}
    outcomes = {}
    for control in CONTROL_REWARDS:
        async with _task_machine(task.environment, factories) as machine:
            assert machine is not None
            files = snapshot.parent_files if control in ("noop", "unittest_monkeypatch") else snapshot.reference_files
            if control != "noop":
                with tempfile.TemporaryDirectory() as directory:
                    for index, (path, content) in enumerate(files.items()):
                        if control in CONTROL_SOURCE and snapshot.parent_files.get(path) != snapshot.reference_files.get(
                            path
                        ):
                            content = attacked_source(content, CONTROL_SOURCE[control])
                        local = Path(directory) / str(index)
                        local.write_text(content)
                        await machine.upload(local, f"{WORKSPACE}/{path}")
                    for path in set(snapshot.parent_files) - set(files):
                        removed = await machine.run(Command(argv=("rm", f"{WORKSPACE}/{path}"), timeout=30))
                        if removed.exit_code != 0:
                            raise RuntimeError("Cannot prepare source deletion")
                    if control == "outside_source":
                        local = Path(directory) / "outside"
                        local.write_text("outside source scope\n")
                        await machine.upload(local, f"{WORKSPACE}/outside.txt")
            grade = await _grade_rollout(
                task,
                SubmissionConvention(id="acceptance", answer_format=AnswerFormat.PLAIN),
                ({"role": "assistant", "content": "Completed."},),
                machine,
                factories,
            )
            outcomes[control] = {"status": grade.status.value, "reward": grade.reward, "diagnostics": grade.diagnostics}
    return outcomes


def controls_pass(outcomes: dict[str, dict]) -> bool:
    return all(
        outcomes[name]["status"] == Outcome.GRADED.value and outcomes[name]["reward"] == reward
        for name, reward in CONTROL_REWARDS.items()
    )


class InvalidGeneration(ValueError):
    """The provider response cannot enter task construction."""


class SnapshotBudgetExceeded(ValueError):
    """The pinned source exceeds the teacher input budget."""


def generation_request(snapshot: SourceSnapshot, *, failure_summary: str, max_tokens: int) -> dict:
    """Build the exact teacher request for generation and resume checks."""
    source_json = snapshot.model_dump_json()
    if len(source_json.encode()) > MAX_GENERATION_BYTES:
        raise SnapshotBudgetExceeded("Source exceeds the teacher input byte budget")
    instruction = (
        "Create a repository repair task from these pinned trees. Return only JSON with problem_statement, "
        "cases, and editable_paths. Each case is {probe_python, expected_json}. Describe behavior, not patch steps "
        "or the reference implementation. A probe runs as an unprivileged fresh Python process and assigns a raw "
        "JSON-compatible result to observation. The trusted parent compares this result to expected_json. "
        "Only probe code and inputs enter the child. Never put expected values, assertions, pass/fail checks, "
        "comparisons to expected results, or scorer calls in probe_python. Import and setup errors are not "
        "behavioral observations. Catch only a named source exception when that exception is the broken behavior, "
        "and record its name as raw observation. Do not catch ImportError, Exception, or BaseException. "
        "Do not inspect source text, hashes, Git history, or exact code. Use only stdlib and the supplied source. "
        "Do not include tests, the commit SHA, reference code, or the patch in the problem statement. "
        "The source root is /workspace. Private probes have no source file: do not use __file__ for source paths. "
        "Add the actual package root to sys.path: /workspace/backend, /workspace/src, or "
        "/workspace/lib/<package>/src as shown by the supplied paths. "
        "Import actual repository modules, never synthetic substitutes. "
        "Only listed source files exist. Do not assume an omitted __init__.py re-exports names: import from the "
        "specific supplied module. If a module is new and its package is absent in the parent, check the top-level "
        "package with importlib.util.find_spec before checking a nested module, then return a raw absence value. "
        "All dependencies must be actual installed packages. "
        "For new APIs, call them through the real module and record a named AttributeError if absent. "
        "Do not monkeypatch the scorer or result output. Keep 2 to 4 independent meaningful cases with bounded "
        "JSON output. Each probe is self-contained valid Python with at most 12000 characters. "
        "editable_paths must be a nonempty subset of the actual changed source paths. "
        "The response must match this JSON schema: " + json.dumps(GeneratedRepair.model_json_schema()) + ". "
        "The abstract fixed-dev failure summary guides task selection only: " + failure_summary
    )
    return {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": source_json},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "prompt_cache_key": f"russell-rsi-{snapshot.repository}-{source_group_id(snapshot)}",
        },
    }


async def generate_repair(
    snapshot: SourceSnapshot,
    *,
    relay_job: str,
    token_env: str,
    failure_summary: str,
    max_tokens: int,
    artifact: Path,
) -> GeneratedRepair:
    """Reuse an identical request or store the full GLM response before parsing."""
    request = generation_request(snapshot, failure_summary=failure_summary, max_tokens=max_tokens)
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if artifact.exists():
        stored = json.loads(artifact.read_text())
        if stored["request_sha256"] != fingerprint or stored["request"] != request or stored["relay_job"] != relay_job:
            raise ValueError("Stored generation has a different request identity")
        raw_response = stored["response"]
    else:
        async with AsyncOpenAI(base_url=resolve_glm_base_url(relay_job), api_key=os.environ[token_env]) as client:
            response = await client.chat.completions.create(**request)
        raw_response = response.model_dump(mode="json")
        artifact.write_text(
            json.dumps(
                {"request": request, "request_sha256": fingerprint, "relay_job": relay_job, "response": raw_response}
            )
            + "\n"
        )
    choices = raw_response.get("choices", [])
    content = choices[0]["message"].get("content") if choices else None
    if content is None:
        raise InvalidGeneration("GLM returned no task")
    return GeneratedRepair.model_validate_json(content)


async def generate_candidates(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.snapshots.open() as stream:
        for line in stream:
            snapshot = SourceSnapshot.model_validate_json(line)
            if snapshot.split == "test":
                continue
            if processed >= args.max_candidates:
                break
            processed += 1
            directory = args.output / source_group_id(snapshot)
            directory.mkdir(exist_ok=True)
            source = directory / "snapshot.json"
            if source.exists() and SourceSnapshot.model_validate_json(source.read_text()) != snapshot:
                raise ValueError("Stored generation has a different source snapshot")
            artifact = directory / "generation.json"
            if (directory / "repair.json").exists() and not artifact.exists():
                raise ValueError("Stored repair has no request identity")
            try:
                repair = await generate_repair(
                    snapshot,
                    relay_job=args.relay_job,
                    token_env=args.token_env,
                    failure_summary=args.failure_summary,
                    max_tokens=args.max_tokens,
                    artifact=artifact,
                )
            except (ValidationError, InvalidGeneration, SnapshotBudgetExceeded) as error:
                (directory / "generation_rejected.json").write_text(
                    json.dumps(
                        {
                            "rejected": True,
                            "reason": str(error),
                            "stage": (
                                "source_input_budget"
                                if isinstance(error, SnapshotBudgetExceeded)
                                else "generated_schema"
                            ),
                        }
                    )
                    + "\n"
                )
                if not source.exists():
                    source.write_text(snapshot.model_dump_json() + "\n")
                continue
            existing = directory / "repair.json"
            if existing.exists() and GeneratedRepair.model_validate_json(existing.read_text()) != repair:
                raise ValueError("Stored repair differs from the generation response")
            if not source.exists():
                source.write_text(snapshot.model_dump_json() + "\n")
            if not existing.exists():
                existing.write_text(repair.model_dump_json() + "\n")


def source_partition(snapshot: SourceSnapshot, record: CommitRecord) -> str:
    """Require the pinned inventory identity and its source-family split."""
    if (
        record.sha != snapshot.commit_sha
        or snapshot.repository not in record.repositories
        or record.parents != [snapshot.parent_sha]
        or not record.family
    ):
        raise ValueError("Snapshot differs from the inventory provenance")
    if record.split not in ("train", "dev", "test") or snapshot.split != record.split:
        raise ValueError("Snapshot differs from the source-family split")
    return record.split


def write_partitions(
    output: Path, tasks: list[tuple[SourceSnapshot, TaskSpec]], inventory: dict[str, CommitRecord]
) -> None:
    """Write separate train and development tasks. Keep test sources sealed."""
    partitions = {"train": [], "dev": []}
    for snapshot, task in tasks:
        record = inventory[snapshot.commit_sha]
        split = source_partition(snapshot, record)
        if split == "test":
            continue
        metadata = {**task.metadata, "split": split, "family": record.family}
        partitions[split].append(task.model_copy(update={"metadata": metadata}))
    for split, filename in (("train", "train.parquet"), ("dev", "development.parquet")):
        write_tasks(str(output / filename), partitions[split])


@dataclass(frozen=True)
class DependencyWheels:
    path: Path
    uri: str


def repository_wheels(bundles: Path, repository: str) -> DependencyWheels | None:
    """Select the recorded wheel bundle for this source repository."""
    manifest = json.loads((bundles / "manifest.json").read_text())
    relative = manifest["repository_wheels"][repository]
    if relative is None:
        return None
    wheels = (bundles / relative).resolve()
    if not wheels.is_relative_to(bundles.resolve()):
        raise ValueError("Wheel bundle is outside its recorded artifact root")
    return DependencyWheels(wheels, prefix_join(manifest["base_uri"], relative))


async def accept_candidates(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    accepted = []
    with args.inventory.open() as stream:
        records = [CommitRecord(**json.loads(line)) for line in stream]
    inventory = {record.sha: record for record in records}
    if len(inventory) != len(records):
        raise ValueError("Inventory contains duplicate commit identities")
    families = {}
    for record in records:
        if record.family in families and families[record.family] != record.split:
            raise ValueError("Inventory family occurs in different splits")
        families[record.family] = record.split
    if args.backend == "docker":
        factory = DockerMachineFactory(skopeo=args.skopeo, image_cache=args.image_cache)
    else:
        factory = QemuMachineFactory(Acceleration.TCG, prepared_registry_bundles={args.image: args.prepared_bundle})
    candidates = [
        directory
        for directory in sorted(args.candidates.iterdir())
        if directory.is_dir() and (directory / "repair.json").exists()
    ]
    for directory in candidates[: args.max_candidates]:
        snapshot = SourceSnapshot.model_validate_json((directory / "snapshot.json").read_text())
        split = source_partition(snapshot, inventory[snapshot.commit_sha])
        if split == "test":
            (directory / "acceptance.json").write_text(
                json.dumps({"accepted": False, "sealed": True, "split": "test"}) + "\n"
            )
            continue
        try:
            repair = GeneratedRepair.model_validate_json((directory / "repair.json").read_text())
            dependencies = repository_wheels(args.dependency_bundles, snapshot.repository)
            task = build_task(
                snapshot,
                repair,
                image=args.image,
                timeout=args.timeout,
                dependency_wheels=dependencies.path if dependencies is not None else None,
                dependency_wheels_uri=dependencies.uri if dependencies is not None else None,
            )
        except (ValidationError, InvalidRepair) as error:
            (directory / "acceptance.json").write_text(
                json.dumps({"accepted": False, "stage": "task_validation", "reason": str(error)}) + "\n"
            )
            continue
        result = await accept_candidate(task, snapshot, factory=factory)
        evidence = {
            "accepted": False,
            "stage": "behavioral_acceptance",
            "behavioral_acceptance": result.accepted,
            "split": split,
            "family": inventory[snapshot.commit_sha].family,
            "parent": result.parent.model_dump(mode="json") if result.parent is not None else None,
            "reference": result.reference.model_dump(mode="json") if result.reference is not None else None,
            "image": args.image,
            "patch_controls": {},
        }
        acceptance = directory / "acceptance.json"
        acceptance.write_text(json.dumps(evidence) + "\n")
        controls = await patch_controls(task, snapshot, factory=factory) if result.accepted else {}
        admitted = result.accepted and controls_pass(controls)
        evidence.update(accepted=admitted, stage="completed", patch_controls=controls)
        acceptance.write_text(json.dumps(evidence) + "\n")
        if admitted:
            accepted.append((snapshot, task))
    write_partitions(args.output, accepted, inventory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--snapshots", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--relay-job", required=True)
    generate.add_argument("--token-env", default=GLM_TOKEN_ENV)
    generate.add_argument("--failure-summary", required=True)
    generate.add_argument("--max-candidates", type=int, required=True)
    generate.add_argument("--max-tokens", type=int, required=True)
    accept = commands.add_parser("accept")
    accept.add_argument("--inventory", type=Path, required=True)
    accept.add_argument("--candidates", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    accept.add_argument("--dependency-bundles", type=Path, required=True)
    accept.add_argument("--image", required=True, help="Pinned Python task image.")
    accept.add_argument("--backend", choices=("docker", "qemu"), required=True)
    accept.add_argument("--prepared-bundle", type=Path, help="Verified QEMU bundle for the pinned task image.")
    accept.add_argument("--skopeo", type=Path)
    accept.add_argument("--image-cache", type=Path)
    accept.add_argument("--max-candidates", type=int, required=True)
    accept.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.max_candidates <= 0:
        parser.error("max-candidates must be positive")
    if args.command == "accept":
        if args.backend == "qemu" and args.prepared_bundle is None:
            parser.error("qemu requires prepared-bundle")
        if args.backend == "docker" and (args.skopeo is None or args.image_cache is None):
            parser.error("docker requires skopeo and image-cache")
    asyncio.run(generate_candidates(args) if args.command == "generate" else accept_candidates(args))


if __name__ == "__main__":
    main()
