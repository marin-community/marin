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
silent rewrite: a separate verifier container that declares its own image, a
non-root verifier user, a multi-step reward strategy, a Windows environment.
A declared-separate verifier with NO image of its own is the sandbox grader
itself — a fresh grading machine from the task's own image — so it converts
(coordinator ruling 2026-11-28, FineEnvs). The declared execution
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
HARBOR_EVAL_OUTPUT = "/logs/verifier/eval-output.txt"
COPIED_TEST_STDOUT = "/source-test-output.txt"
COPIED_TEST_EXIT = "/source-exit-code.txt"
COPIED_VERDICT = "/source-verdict.json"
COPIED_REWARD_JSON = "/source-reward.json"
COPIED_REWARD_TEXT = "/source-reward.txt"
COPIED_EVAL_OUTPUT = "/source-eval-output.txt"
REWARD_PATH = "/converted-verdict.json"
REWARD_JSON_PATH = "/converted-reward.json"
REWARD_TEXT_PATH = "/converted-reward.txt"
REWARD_DIVERGENCE_PATH = "/converted-reward-divergence.txt"
CONVERSION_PROTOCOL = "harbor-cohort-taskspec-v1"
HARBOR_VERIFIER_TIMEOUT = 600.0
_ROOT_USERS = frozenset({"root", "0"})

# Close the tamper race (coordinator ruling 2026-11-26: harden before OpenSWE
# admission): kill every process the sandbox still carries that is neither this
# command's own ancestry nor the sandbox's service infrastructure, before the
# verifier runs and before the artifact copies. Daytona provides no confirmed
# scoped descendant teardown, so an agent process that outlived its command
# could otherwise rewrite the reward file while the artifacts copy. Best effort
# by design: a kill failure must never fail grading, and the sandbox's own
# services (the Daytona agent and toolbox) survive by cmdline identity.
KILL_AGENT_LEFTOVERS_SCRIPT = r"""ppid_of() { awk '/^PPid:/{print $2}' "/proc/$1/status" 2>/dev/null; }
set +e
self=$$
ancestors=" "
walk=$self
while [ -n "$walk" ] && [ "$walk" != "1" ] && [ "$walk" != "0" ]; do
  ancestors="$ancestors$walk "
  walk=$(ppid_of "$walk")
done
for entry in /proc/[0-9]*; do
  pid=${entry#/proc/}
  case "$ancestors" in *" $pid "*) continue ;; esac
  [ "$pid" = "1" ] && continue
  cmdline=$(cat "$entry/cmdline" 2>/dev/null | tr '\0' ' ')
  [ -n "$cmdline" ] || continue
  case "$cmdline" in *daytona*|*toolbox*|*shellbox*) continue ;; esac
  kill -KILL "$pid" 2>/dev/null
done
exit 0
"""


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
    environment = verifier.get("environment")
    environment_mode = verifier.get("environment_mode")
    has_environment = environment is not None
    return {
        "env": dict(verifier.get("env") or {}),
        "user": verifier.get("user"),
        "timeout_sec": (
            verifier.get("timeout_sec") if verifier.get("timeout_sec") is not None else HARBOR_VERIFIER_TIMEOUT
        ),
        "environment_mode": environment_mode or ("separate" if has_environment else "shared"),
        "verifier_image": (environment or {}).get("docker_image"),
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


# The OpenSWE fail-closed marker rule recomputed from the copied eval output
# (coordinator ruling 2026-11-26): a planted or rewritten reward that disagrees
# with the output's own verdict scores zero and leaves the divergence beside
# it. Runs on the grading machine over the copied artifacts; the eval output is
# the exact bytes the verifier captured (the profile's verifier cats it into the
# stdout as well, so the stdout is the fallback source).
OPENSWE_RECOMPUTE_SCRIPT = r"""if [ -f /source-reward.txt ]; then
  out=/source-eval-output.txt
  [ -f "$out" ] || out=/source-test-output.txt
  total=$(grep -c '^OPENSWE_EXIT_CODE=' "$out" || true)
  zero=$(grep -c '^OPENSWE_EXIT_CODE=0$' "$out" || true)
  expected=0
  if [ "$total" -eq 1 ] && [ "$zero" -eq 1 ]; then expected=1; fi
  recorded=$(cat /source-reward.txt 2>/dev/null)
  if [ "$recorded" != "$expected" ]; then
    printf 'recorded=%s recomputed=%s\n' "${recorded-<missing>}" "$expected" > /converted-reward-divergence.txt
    echo 0 > /converted-reward.txt
  fi
fi
"""


def _reward_files(
    has_verifier_config: bool, recompute: str | None = None
) -> tuple[tuple[VerifierArtifact, ...], FileReward, VerifierCommand]:
    """Copy Harbor's reward artifacts and read them at the grader's own reward paths.

    The grading runtime clears the reward-file paths before the grader command
    runs (the reward is the grader command's output), so the argv copies the
    arrived artifact to the reward path — the root-delivery graders' exact
    shape — and the FileReward reads it there. The reward artifacts copy with
    the skip-if-missing policy: the fallback path's reward.json and reward.txt
    are alternatives the harness chooses between, and a crashed harness leaves
    the copy (and so the grading) failed. The test script's own exit code is
    copied beside the reward for diagnostics, exactly as Harbor records it in
    the control detail.

    ``recompute`` hardens the fallback's text path: the grader's argv re-derives
    the expected verdict from the copied eval output and scores zero on any
    disagreement with the copied reward file, so a planted or rewritten reward
    cannot score (the divergence lands beside the reward as evidence). A family
    without a verified derivation rule keeps the plain copy; the mechanism is
    per-family opt-in.
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
        reward = FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.VERDICT_JSON),))
        return artifacts, reward, VerifierCommand(argv=("cp", COPIED_VERDICT, REWARD_PATH), cwd="/")
    artifacts = (
        VerifierArtifact(source=HARBOR_TEST_STDOUT, target=COPIED_TEST_STDOUT, kind=ArtifactKind.FILE, missing=skipped),
        VerifierArtifact(source=HARBOR_REWARD_JSON, target=COPIED_REWARD_JSON, kind=ArtifactKind.FILE, missing=skipped),
        VerifierArtifact(source=HARBOR_REWARD_TEXT, target=COPIED_REWARD_TEXT, kind=ArtifactKind.FILE, missing=skipped),
        VerifierArtifact(source=HARBOR_EVAL_OUTPUT, target=COPIED_EVAL_OUTPUT, kind=ArtifactKind.FILE, missing=skipped),
        *exit_artifact,
    )
    reward = FileReward(
        files=(
            RewardFile(path=REWARD_JSON_PATH, format=RewardFileFormat.JSON),
            RewardFile(path=REWARD_TEXT_PATH, format=RewardFileFormat.NUMBER),
        )
    )
    recompute_text = recompute or ":"
    argv = (
        "bash",
        "-c",
        f"if cp {COPIED_REWARD_JSON} {REWARD_JSON_PATH} 2>/dev/null; then :; else\n"
        f"cp {COPIED_REWARD_TEXT} {REWARD_TEXT_PATH}\n"
        f"{recompute_text}\n"
        f"fi",
    )
    return artifacts, reward, VerifierCommand(argv=argv, cwd="/")


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
    if verifier["environment_mode"] == "separate" and verifier["verifier_image"]:
        # Coordinator ruling (2026-11-28, FineEnvs): a declared-separate verifier with NO
        # [verifier.environment] image means a fresh grading machine from the task's own
        # image — exactly the sandbox grader below, so it converts. A declared-separate
        # verifier WITH its own distinct image stays behind the Harbor runtime.
        raise CohortConversionRejected(
            f"{task.id}: a separate verifier container with its own image stays behind the Harbor runtime"
        )
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
    # The OpenSWE profile's fail-closed verifier carries its own marker rule; its
    # copied eval output re-derives the verdict at the reward path (the grader
    # hardening: a planted reward that disagrees scores zero).
    recompute = (
        OPENSWE_RECOMPUTE_SCRIPT
        if any(file.path == "tests/eval_script.sh" for file in task.resources.verifier)
        else None
    )
    artifacts, reward, guard = _reward_files(has_verifier_config, recompute)
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
            VerifierCommand(argv=("bash", "-c", KILL_AGENT_LEFTOVERS_SCRIPT), cwd="/"),
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
