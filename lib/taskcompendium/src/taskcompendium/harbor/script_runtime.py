# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run private script graders in a bounded, disposable OCI container."""

import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.harbor.workspace import MAX_WORKSPACE_BYTES, validate_workspace_snapshot
from taskcompendium.models import AnswerType
from taskcompendium.submission import WORKSPACE_ROOT
from taskcompendium.verifiers.script import (
    NetworkPolicy,
    ResourceResolver,
    ScriptVerifier,
    materialize_private_resources,
)

APP_DIR = WORKSPACE_ROOT
TESTS_DIR = "/tests"
VERIFIER_DIR = "/verifier"
SUBMISSION_FILE = "submission.json"
RESULT_FILE = "result.json"
MAX_SUBMISSION_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_RUNTIME_SECONDS = 900
MEMORY_LIMIT = "512m"
CPU_LIMIT = "1"
PID_LIMIT = "64"
TMPFS_LIMIT = "64m"
DOCKER_CLEANUP_TIMEOUT = 10


@dataclass(frozen=True)
class ScriptSubmission:
    """The normalized answer and submission metadata given to a script grader."""

    protocol_version: int
    answer_type: AnswerType
    convention_id: str
    answer: str | None


def _copy_workspace(source: Path, destination: Path) -> None:
    """Copy regular files without following agent-controlled symlinks."""
    validate_workspace_snapshot(source)
    destination.mkdir(mode=0o777)
    total_bytes = 0
    pending = [(source, destination)]
    while pending:
        current_source, current_destination = pending.pop()
        for entry in current_source.iterdir():
            source_stat = entry.lstat()
            target = current_destination / entry.name
            if stat.S_ISDIR(source_stat.st_mode):
                target.mkdir(mode=0o777)
                target.chmod(0o777)
                pending.append((entry, target))
            elif stat.S_ISREG(source_stat.st_mode):
                total_bytes += source_stat.st_size
                if total_bytes > MAX_WORKSPACE_BYTES:
                    raise ValueError("Workspace snapshot exceeds the runtime size limit")
                descriptor = os.open(entry, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as input_file, target.open("xb") as output_file:
                    if not stat.S_ISREG(os.fstat(input_file.fileno()).st_mode):
                        raise ValueError("Workspace snapshot contains a non-file entry")
                    shutil.copyfileobj(input_file, output_file)
                executable = bool(source_stat.st_mode & 0o111)
                target.chmod(0o777 if executable else 0o666)
            else:
                raise ValueError("Workspace snapshot contains a symlink or special file")
    destination.chmod(0o777)


def _read_result(path: Path) -> GradeResult:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return GradeResult(Outcome.INVALID_TASK, None, "Script grader did not write a result")
    except OSError as error:
        return GradeResult(Outcome.INVALID_TASK, None, f"Cannot read script grader result: {error}")
    try:
        with os.fdopen(descriptor, "rb") as result_file:
            if not stat.S_ISREG(os.fstat(result_file.fileno()).st_mode):
                raise ValueError("Script grader result is not a regular file")
            payload = result_file.read(MAX_RESULT_BYTES + 1)
        if len(payload) > MAX_RESULT_BYTES:
            raise ValueError("Script grader result exceeds the size limit")
        result = json.loads(payload)
        if not isinstance(result, dict) or result.get("status") not in {"scored", "invalid_task", "infra_error"}:
            raise ValueError("Script grader result has an invalid status")
        if set(result) - {"status", "reward", "error"}:
            raise ValueError("Script grader result has unknown fields")
        error = result.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("Script grader result error must be text")
        if result["status"] == "scored":
            reward = result.get("reward")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or not 0 <= reward <= 1
            ):
                raise ValueError("Scored script grader result requires a reward between zero and one")
            return GradeResult(Outcome.GRADED, float(reward), error)
        if "reward" in result:
            raise ValueError("Unscored script grader result cannot include a reward")
        status = Outcome.INVALID_TASK if result["status"] == "invalid_task" else Outcome.INFRA_ERROR
        return GradeResult(status, None, error)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return GradeResult(Outcome.INVALID_TASK, None, str(error))


def _docker_command(config: ScriptVerifier, root: Path, container_name: str, docker_binary: str) -> list[str]:
    network = "none" if config.network_policy == NetworkPolicy.DISABLED else "bridge"
    return [
        docker_binary,
        "run",
        "--rm",
        "--name",
        container_name,
        "--pull=never",
        "--network",
        network,
        "--log-driver=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        PID_LIMIT,
        "--memory",
        MEMORY_LIMIT,
        "--memory-swap",
        MEMORY_LIMIT,
        "--cpus",
        CPU_LIMIT,
        "--user",
        "65534:65534",
        "--tmpfs",
        f"/tmp:rw,nosuid,noexec,size={TMPFS_LIMIT}",
        "--mount",
        f"type=bind,src={root / 'app'},dst={APP_DIR}",
        "--mount",
        f"type=bind,src={root / 'tests'},dst={TESTS_DIR},readonly",
        "--mount",
        f"type=bind,src={root / 'verifier'},dst={VERIFIER_DIR}",
        config.runtime_image,
        f"{TESTS_DIR}/{config.entrypoint}",
        *config.args,
    ]


def run_script_verifier(
    config: ScriptVerifier,
    submission: ScriptSubmission,
    workspace_snapshot: Path,
    verifier_dir: Path,
    resolve_uri: ResourceResolver | None = None,
    *,
    docker_binary: str = "docker",
) -> GradeResult:
    """Grade a frozen workspace with an isolated script and return its verdict."""
    try:
        record = asdict(submission)
        if record["protocol_version"] != 1 or isinstance(record["protocol_version"], bool):
            raise ValueError("Unsupported submission protocol")
        if record["answer_type"] not in {"text", "number", "file", "workspace_state", "native_action"}:
            raise ValueError("Submission record has an invalid answer type")
        if not isinstance(record["convention_id"], str) or not record["convention_id"]:
            raise ValueError("Submission record requires a convention ID")
        if record["answer"] is not None and not isinstance(record["answer"], str):
            raise ValueError("Submission answer must be text or null")
        submission_bytes = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
        if len(submission_bytes) > MAX_SUBMISSION_BYTES:
            raise ValueError("Submission record exceeds the runtime size limit")
    except (TypeError, ValueError) as error:
        return GradeResult(Outcome.INVALID_TASK, None, str(error))

    with tempfile.TemporaryDirectory(prefix="taskcompendium-script-") as temporary:
        root = Path(temporary)
        app = root / "app"
        tests = root / "tests"
        verifier = root / "verifier"
        try:
            _copy_workspace(workspace_snapshot, app)
            materialize_private_resources(config.resources, tests, resolve_uri=resolve_uri)
            entrypoint = tests / config.entrypoint
            if not entrypoint.is_file() or entrypoint.is_symlink():
                raise ValueError("Script entrypoint is not a pinned private file")
            for directory in (tests, *[path for path in tests.rglob("*") if path.is_dir()]):
                directory.chmod(0o555)
            for resource in tests.rglob("*"):
                if resource.is_file():
                    resource.chmod(0o555 if resource.stat().st_mode & 0o111 else 0o444)
            verifier.mkdir(mode=0o777)
            verifier.chmod(0o777)
            (verifier / SUBMISSION_FILE).write_bytes(submission_bytes)
            (verifier / SUBMISSION_FILE).chmod(0o444)
        except (OSError, TypeError, ValueError) as error:
            return GradeResult(Outcome.INVALID_TASK, None, str(error))

        container_name = f"taskcompendium-verifier-{uuid.uuid4().hex}"
        command = _docker_command(config, root, container_name, docker_binary)
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=min(config.timeout_seconds, MAX_RUNTIME_SECONDS),
                check=False,
            )
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    [docker_binary, "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=DOCKER_CLEANUP_TIMEOUT,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                return GradeResult(Outcome.INFRA_ERROR, None, f"Script grader timed out and cleanup failed: {error}")
            return GradeResult(Outcome.INFRA_ERROR, None, "Script grader exceeded its timeout")
        except OSError as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"Docker runtime unavailable: {error}")

        result_path = verifier / RESULT_FILE
        result = _read_result(result_path)
        if completed.returncode == 125 and not result_path.exists():
            return GradeResult(Outcome.INFRA_ERROR, None, "Docker could not start the script grader")
        if completed.returncode != 0 and result.status == Outcome.INVALID_TASK and result.error is None:
            return GradeResult(Outcome.INVALID_TASK, None, "Script grader exited without a valid result")
        verifier_dir.mkdir(parents=True, exist_ok=True)
        protocol_status = "scored" if result.status == Outcome.GRADED else result.status.value
        result_record: dict[str, str | float] = {"status": protocol_status}
        if result.reward is not None:
            result_record["reward"] = result.reward
        if result.error is not None:
            result_record["error"] = result.error
        (verifier_dir / RESULT_FILE).write_text(json.dumps(result_record) + "\n")
        return result
