# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert native Harbor cohort rows into executable TaskSpecs the rollout serves.

marin #12167 removes the Harbor runtime from SkyRL: the rollout serves only
executable TaskSpecs. A cohort row's grading lives in its own package harness —
Harbor's verifier uploads the package's ``tests/`` tree to ``/tests`` in the
task's environment, chmods the script, runs it as the environment's user, and
reads the reward the harness wrote under ``/logs/verifier``: a structured
``verdict.json`` when the package carries ``tests/verifier.toml``, otherwise
``reward.json`` (JSON) or ``reward.txt`` (number). The conversion keeps the
harness bytes and the exact Harbor execution command, and re-expresses the
verifier steps as the sandbox grader the rollout engine executes:

- ``collect`` commands run on the agent's machine as root, each under the
  grader's verifier deadline: the ``tests/`` tree is restored to ``/tests``
  with its original modes, the task's ``[[verifier.collect]]`` blocks run
  next, then Harbor's exact execution command runs with the task's verifier
  env in the native reward-file mode — the script runs unchanged even when
  its tests fail, its exact exit code persists to
  ``/logs/verifier/exit-code.txt`` beside the reward file it wrote, and the
  reward file arbitrates the score. Harbor bounds each collect hook at its own
  60-second default and only warns when one fails; here a failing or hanging
  collect is a grading failure at the verifier deadline — the conservative
  direction for graders that must fail closed. A harness that crashes before
  scoring leaves no reward file, and the missing file is a grading failure.
- ``artifacts`` copy the test stdout and the reward file to the grading
  machine.
- ``reward`` reads the structured verdict when ``verifier.toml`` is present;
  the Harbor reward-file fallback maps ``reward.json`` to JSON and
  ``reward.txt`` to a number, and a missing file stays a grading failure.
- ``environment`` pins the task's own image or build so the grading machine can
  transport the reward; the actor machine keeps the workspace state the harness
  grades.

Everything else stays byte-identical: context, resources, integrity contract.
Anything the conversion cannot express faithfully is a typed rejection, never a
silent rewrite: a separate verifier container, a non-root verifier user, a
multi-step reward strategy, a Windows environment. The declared execution
limits lift from the source task configuration exactly as the native import
does, refusing conflicts with an already-measured population.

The module never imports the vendored Harbor: the controller and runtime
environments have none (the base-environment contract), so the task
configuration is read with the same raw schema the native import uses, and the
parity test pins the read against Harbor's own parser where Harbor is
installed.
"""

import gzip
import hashlib
import io
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from taskcompendium.harbor.execution import native_execution_limits, task_with_declared_execution_requirements
from taskcompendium.harbor.native_package import validate_native_package
from taskcompendium.harbor.root_delivery import HOME_RESTORE_CHUNK_BYTES, archive_chunk_commands
from taskcompendium.models import (
    ArtifactKind,
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    FileReward,
    MissingArtifactPolicy,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.staging import StagedFile, write_archive

HARBOR_TEST_SCRIPT = "tests/test.sh"
HARBOR_VERIFIER_CONFIG = "tests/verifier.toml"
HARBOR_TEST_STDOUT = "/logs/verifier/test-stdout.txt"
HARBOR_TEST_EXIT = "/logs/verifier/exit-code.txt"
HARBOR_VERDICT = "/logs/verifier/verdict.json"
HARBOR_REWARD_JSON = "/logs/verifier/reward.json"
HARBOR_REWARD_TEXT = "/logs/verifier/reward.txt"
COPIED_TEST_STDOUT = "/source-test-output.txt"
COPIED_TEST_EXIT = "/source-exit-code.txt"
COPIED_VERDICT = "/source-verdict.json"
COPIED_REWARD_JSON = "/source-reward.json"
COPIED_REWARD_TEXT = "/source-reward.txt"
CONVERSION_PROTOCOL = "harbor-cohort-taskspec-v1"
HARBOR_VERIFIER_TIMEOUT = 600.0
_ROOT_USERS = frozenset({"root", "0"})


class CohortConversionRejected(ValueError):
    """A cohort row the conversion cannot express faithfully; re-admission needs a typed variant."""


def _require_root_user(user: object, what: str) -> None:
    """Collect commands run as root; a declared non-root user cannot be honored."""
    if user is not None and str(user).lower() not in _ROOT_USERS:
        raise CohortConversionRejected(f"{what} declares user {user!r}; the conversion runs its commands as root")


def _task_environment(data: Mapping[str, Any]) -> dict[str, Any]:
    """The raw ``[environment]`` table with Harbor's own defaults and normalization."""
    environment = data.get("environment") or {}
    os_name = environment.get("os")
    return {
        "docker_image": environment.get("docker_image"),
        "os": (os_name or "linux").lower() if isinstance(os_name, str) else "linux",
        "workdir": environment.get("workdir"),
        "env": dict(environment.get("env") or {}),
        "setup": environment.get("setup"),
        "cpus": environment.get("cpus"),
        "memory_mb": environment.get("memory_mb"),
        "storage_mb": environment.get("storage_mb"),
    }


def _task_verifier(data: Mapping[str, Any]) -> dict[str, Any]:
    """The raw ``[verifier]`` table with Harbor's own defaults."""
    verifier = data.get("verifier") or {}
    environment_mode = verifier.get("environment_mode")
    has_environment = verifier.get("environment") is not None
    return {
        "env": dict(verifier.get("env") or {}),
        "user": verifier.get("user"),
        "timeout_sec": (
            verifier.get("timeout_sec") if verifier.get("timeout_sec") is not None else HARBOR_VERIFIER_TIMEOUT
        ),
        "environment_mode": environment_mode or ("separate" if has_environment else "shared"),
        "collect": list(verifier.get("collect") or []),
    }


def _needs_chmod(resource: TaskResource) -> bool:
    mode = resource.mode
    return mode is None or not (int(mode, 8) & 0o111)


def _staged_tests_tree(task: TaskSpec) -> tuple[list[StagedFile], bool]:
    """The package's tests/ tree staged for /tests, and whether it carries a verifier config."""
    files = [resource for resource in task.resources.verifier if resource.path.startswith("tests/")]
    has_verifier_config = any(resource.path == HARBOR_VERIFIER_CONFIG for resource in files)
    staged = [
        StagedFile(
            resource.path.removeprefix("tests/"),
            resource_bytes(resource),
            0o644 if resource.mode is None else int(resource.mode, 8),
            resource.mtime_ns,
        )
        for resource in files
    ]
    return staged, has_verifier_config


def _restore_commands(task: TaskSpec, staged: list[StagedFile]) -> tuple[VerifierCommand, ...]:
    """Chunk the tests/ archive onto the agent machine and extract it under /tests."""
    with TemporaryDirectory(prefix="cohort-conversion-") as directory:
        archive = Path(directory) / "tests.tar"
        write_archive(archive, staged, [])
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as writer:
            writer.write(archive.read_bytes())
    roots = sorted({file.path.partition("/")[0] for file in staged})
    script = "\n".join(
        (
            "set -euo pipefail",
            *(f"rm -rf -- {PurePosixPath('/tests') / root}" for root in roots),
            "tar -C /tests -xzf /tests/.capture.tar.gz",
            "rm -f /tests/.capture.tar.gz",
        )
    )
    return (
        *archive_chunk_commands(compressed.getvalue(), "/tests", HOME_RESTORE_CHUNK_BYTES),
        VerifierCommand(argv=("bash", "-c", script), cwd="/"),
    )


def _reward_files(has_verifier_config: bool) -> tuple[tuple[VerifierArtifact, ...], FileReward, VerifierCommand]:
    """Copy Harbor's reward artifacts and read them with the same precedence.

    The argv guards the artifact's arrival: a grading machine without the
    copied reward file fails loudly instead of scoring anything. The reward
    artifacts themselves copy with the skip-if-missing policy — the fallback
    path's reward.json and reward.txt are alternatives (the harness writes one
    of them), and a crashed harness leaves the guard to fail the grading. The
    test script's own exit code is copied beside the reward for diagnostics,
    exactly as Harbor records it in the control detail.
    """
    skipped = MissingArtifactPolicy.SKIP
    exit_artifact = (
        VerifierArtifact(source=HARBOR_TEST_EXIT, target=COPIED_TEST_EXIT, kind=ArtifactKind.FILE, missing=skipped),
    )
    if has_verifier_config:
        artifacts = (
            VerifierArtifact(
                source=HARBOR_TEST_STDOUT, target=COPIED_TEST_STDOUT, kind=ArtifactKind.FILE, missing=skipped
            ),
            VerifierArtifact(source=HARBOR_VERDICT, target=COPIED_VERDICT, kind=ArtifactKind.FILE, missing=skipped),
            *exit_artifact,
        )
        reward = FileReward(files=(RewardFile(path=COPIED_VERDICT, format=RewardFileFormat.VERDICT_JSON),))
        return artifacts, reward, VerifierCommand(argv=("test", "-f", COPIED_VERDICT), cwd="/")
    artifacts = (
        VerifierArtifact(source=HARBOR_TEST_STDOUT, target=COPIED_TEST_STDOUT, kind=ArtifactKind.FILE, missing=skipped),
        VerifierArtifact(source=HARBOR_REWARD_JSON, target=COPIED_REWARD_JSON, kind=ArtifactKind.FILE, missing=skipped),
        VerifierArtifact(source=HARBOR_REWARD_TEXT, target=COPIED_REWARD_TEXT, kind=ArtifactKind.FILE, missing=skipped),
        *exit_artifact,
    )
    reward = FileReward(
        files=(
            RewardFile(path=COPIED_REWARD_JSON, format=RewardFileFormat.JSON),
            RewardFile(path=COPIED_REWARD_TEXT, format=RewardFileFormat.NUMBER),
        )
    )
    guard = VerifierCommand(
        argv=("bash", "-c", f"test -f {COPIED_REWARD_JSON} || test -f {COPIED_REWARD_TEXT}"),
        cwd="/",
    )
    return artifacts, reward, guard


@dataclass(frozen=True)
class CohortConversion:
    """One converted row: the executable TaskSpec and its per-task conversion ledger entry."""

    task: TaskSpec
    input_task_sha256: str
    output_task_sha256: str
    changed_fields: tuple[str, ...]


def convert_cohort_task(task: TaskSpec, *, input_task_sha256: str) -> CohortConversion:
    """Rewrite one native Harbor cohort row as an executable TaskSpec, or reject it typed."""
    try:
        validate_native_package(task)
    except ValueError as error:
        raise CohortConversionRejected(f"{task.id}: not a native Harbor package: {error}") from error
    package = {resource.path: resource for resource in task.resources.verifier}
    data = tomllib.loads(resource_bytes(package["task.toml"]).decode())
    environment = _task_environment(data)
    verifier = _task_verifier(data)
    if data.get("steps"):
        raise CohortConversionRejected(f"{task.id}: multi-step tasks keep their Harbor trial strategy")
    if environment["os"] == "windows":
        raise CohortConversionRejected(f"{task.id}: Windows environments stay behind the Harbor runtime")
    if verifier["environment_mode"] == "separate":
        raise CohortConversionRejected(f"{task.id}: a separate verifier container stays behind the Harbor runtime")
    _require_root_user(verifier["user"], f"{task.id}: verifier")
    for block in verifier["collect"]:
        _require_root_user(block.get("user"), f"{task.id}: verifier.collect")

    if environment["docker_image"]:
        image, build = environment["docker_image"], None
    else:
        image, build = None, DockerBuildContext(
            files=tuple(
                resource.model_copy(update={"path": path.removeprefix("environment/")})
                for path, resource in package.items()
                if path.startswith("environment/")
            )
        )

    staged, has_verifier_config = _staged_tests_tree(task)
    script_resource = package[HARBOR_TEST_SCRIPT]
    chmod_commands: tuple[VerifierCommand, ...] = ()
    if _needs_chmod(script_resource):
        chmod_commands = (VerifierCommand(argv=("chmod", "+x", "/tests/test.sh"), cwd="/"),)
    verifier_env = dict(verifier["env"])
    collect_blocks = tuple(
        VerifierCommand(
            argv=("bash", "-c", block["command"]),
            cwd=environment["workdir"] or "/",
            env=verifier_env,
        )
        for block in verifier["collect"]
    )
    workdir = environment["workdir"] or "/"
    # Native reward-file result mode: the verifier log directory is recreated
    # clean (the cohort images may carry a recorded trial's stale verdict —
    # grading it would be a false pass), the script runs unchanged even when
    # its tests fail, its exact exit code persists beside the reward file it
    # wrote, and the reward file arbitrates the score (the recorded Harbor bug
    # control exits nonzero by design while writing a scored zero).
    execute = VerifierCommand(
        argv=(
            "bash",
            "-c",
            f"rm -rf /logs/verifier && mkdir -p /logs/verifier && "
            f"( ( /tests/test.sh ) > {HARBOR_TEST_STDOUT} 2>&1 || status=$?; "
            f'printf "%s\\n" "${{status-0}}" > {HARBOR_TEST_EXIT} ); exit 0',
        ),
        cwd=workdir,
        env=verifier_env,
    )
    artifacts, reward, guard = _reward_files(has_verifier_config)
    grader = ScriptGrader(
        argv=guard.argv,
        cwd=guard.cwd,
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=image,
            docker_build=build,
            working_directory="/",
        ),
        answer_path=None,
        collect=(
            *_restore_commands(task, staged),
            *chmod_commands,
            *collect_blocks,
            execute,
        ),
        artifacts=artifacts,
        reward=reward,
        timeout=verifier["timeout_sec"],
    )
    converted = task_with_declared_execution_requirements(
        task.model_copy(
            update={
                "environment_requirements": EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    docker_image=image,
                    docker_build=build,
                    working_directory=environment["workdir"],
                    environment_variables=dict(environment["env"]),
                    setup_commands=(environment["setup"]["command"],) if environment["setup"] else (),
                ),
                "grader": grader,
                "tags": tuple(tag for tag in task.tags if tag != "harbor"),
                "labels": {**task.labels, "harbor_cohort_conversion": CONVERSION_PROTOCOL},
            }
        ),
        native_execution_limits(data),
    )
    output_sha = hashlib.sha256(converted.model_dump_json().encode()).hexdigest()
    return CohortConversion(
        task=converted,
        input_task_sha256=input_task_sha256,
        output_task_sha256=output_sha,
        changed_fields=(
            "environment_requirements",
            "execution_requirements",
            "grader",
            "tags",
            "labels",
        ),
    )
