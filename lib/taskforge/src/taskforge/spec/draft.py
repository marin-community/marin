# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs and lower it for RolloutEngine.

These helpers own the serialized details a builder should not repeat: the
grader model, resource groups, and the capability requirements of a task
machine. ``assemble`` adds the checks TaskSpec itself does not make and returns
a spec that survives a JSON round trip.

A ``TaskSpec`` holds the semantic task only. How it runs (machine backend,
limits, user, deadlines, harness) is the ``TaskRolloutSpec`` RolloutEngine executes, and
``lower`` alone builds one, choosing each machine's backend for the host. A
lowered spec is therefore bound to the host whose factories it names, and its
task may name the software differently from the ``TaskSpec``: on a host with
the local backend, ``GRADER_BASE_IMAGE`` becomes ``LOCAL_CPYTHON``; on one
without it a package machine has no backend (``host_requirements``).

Every Taskforge spec names ``SHELL_HARNESS``, RolloutEngine's shell harness, and every Taskforge engine
comes from ``rollout_engine``, which registers that harness with its command timeout: the timeout of one
shell command belongs to the harness, not to the spec.

Only ``_presentation`` builds the presented context, concrete tools and answer
format, and only ``lower`` builds a ``TaskRolloutSpec``, so moving either is a
single-site change.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import JsonValue
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerFormat,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    FunctionDefinition,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.submission import require_submission_compatibility
from verifyit.candidate import IN_PROCESS_MODES
from verifyit.spec import JudgeSpec, Spec, mode_of

from taskforge.sandbox.factories import MachineHost, container_backend, grading_environment
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec.decision import SOLVER_TURN_STATE, DecisionGrader, StateSource, task_package
from taskforge.spec.judge import (
    JUDGE_MODELS,
    JUDGE_QUESTION,
    JUDGE_SPEC_PATH,
    JudgeRubricSpec,
    JudgeStrength,
    judge_spec_json,
)
from taskforge.spec.rubric import Rubric

SHELL_CAPABILITY = "shell"
SHELL_HARNESS = "shellbox"
"""The name every Taskforge ``TaskRolloutSpec`` selects its harness by, and ``rollout_engine`` registers
RolloutEngine's ``ShellHarness`` under."""
NO_MACHINE = EnvironmentRequirements()
"""The environment of a task without a machine: the solver answers in the conversation, with no tools."""
FILESYSTEM_CAPABILITY = "filesystem"
SHELLSIM_CPYTHON = PackageSpec(pypi_packages=())
"""The software of a task machine without an image: the ports catalog's CPython and no PyPI packages.

ShellSim installs a catalog's CPython runtime whenever ``pypi_packages`` is set, even to no packages;
a bare ``PackageSpec()`` has the built-in shell tools and no ``python3``. The catalog
``sandbox.factories.SHELLSIM_CATALOG`` provides CPython 3.13 with its standard library.
"""

LOCAL_CPYTHON = PackageSpec(pypi_packages=())
LOCAL_READ_ONLY_ROOTS = (
    "/bin",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/opt",
    "/run/systemd/resolve",
    "/sbin",
    "/sys",
    "/usr",
)
"""The host directories the local backend mounts read-only, shellbox's ``SYSTEM_DIRECTORIES``.

A machine that must write under one of them, a task shipping files to ``/opt`` for instance, cannot
run on the local backend, and ``host_requirements`` keeps its image. ``tests/spec/test_draft.py``
holds this tuple equal to shellbox's on hosts where the local backend imports.
"""
"""``GRADER_BASE_IMAGE``'s software on the local backend: a uv-managed CPython 3.12 and no PyPI packages.

shellbox's local factory resolves the empty requirement list into an empty hash lock, then installs the
interpreter (``shellbox.backends.local.python_environment.DEFAULT_PYTHON_VERSION``) and a venv from it
under the factory's package cache when the first such machine starts, and mounts both read-only as
the machine's ``HostImage``, the venv's ``bin`` first on ``PATH``. Later machines on the host reuse
the build. ``sh``, ``setsid``, ``tar`` and the other commands come from the host's system directories.
"""


type Reward = StdoutReward | ExitCodeReward | FileReward

ANSWER_PATH = "/app/answer.txt"
"""Where a script grader finds the extracted answer of a text, number, JSON or native-action task."""
GRADER_RESOURCE_ROOT = "/tests"
"""Where RolloutEngine installs ``resources.verifier`` on the verifier machine."""
PYTHON_GRADER_ARGV = ("python3", f"{GRADER_RESOURCE_ROOT}/grade.py")
PYTHON_GRADER_FILES = frozenset({"grade.py", "config.json"})
GRADER_VERDICT_PATH = "/logs/verifier/reward.json"
"""Where a Taskforge grader program writes its verdict (``spec.verdict``)."""
MACHINE_ANSWERS = frozenset({AnswerType.FILE, AnswerType.WORKSPACE_STATE})
"""Answers the grader reads from the task machine's files rather than from the conversation."""


@dataclass(frozen=True)
class Resources:
    """Machine limits. ``None`` leaves the limit to the machine factory."""

    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0


@dataclass(frozen=True)
class MachineSettings:
    """A ``MachineRuntimeSpec`` without its backend; ``lower`` picks the backend for this host."""

    network: NetworkPolicy
    resources: Resources
    user: str | None
    startup_timeout: float | None
    cleanup_timeout: float | None


@dataclass(frozen=True)
class _Presentation:
    """The TaskSpec fields that describe how the task is presented."""

    context: ConversationInput
    final_tools: tuple[FunctionDefinition, ...]
    answer_format: AnswerFormat


def file(path: str, content: str, mode: int = 0o644) -> TaskResource:
    """A UTF-8 text file at ``path`` relative to the machine root.

    ``"workspace/numbers.txt"`` lands at ``/workspace/numbers.txt``; an absolute
    path fails TaskResource validation.
    """
    return TaskResource(path=path, source=inline_resource(path, content.encode()).source, mode=f"{mode:o}")


def requirements(
    *,
    image: str | None,
    packages: Sequence[str] = (),
    setup: Sequence[str] = (),
    workdir: str = "/workspace",
    env: Mapping[str, str] | None = None,
) -> EnvironmentRequirements:
    """The task machine's contents, all visible to the agent.

    ``image`` is a digest-pinned registry reference (an ``ImageBuilder`` output), whose commands
    run as Linux processes, or ``None`` for ShellSim's simulated shell and built-in filesystem with
    the catalog's CPython (``SHELLSIM_CPYTHON``). ``packages`` instead names PyPI requirements
    (``"numpy"``, ``"pandas==2.2.3"``) for a machine without an image: Linux processes with CPython
    3.12 and those packages, installed from wheels when the machine starts, with no image to build
    (``host_requirements`` says how each host provides them). ``setup`` commands run with
    ``sh -c`` as user ``0`` under the machine's startup timeout; a builder that
    needs a readiness check puts a wait loop there. Agent-visible files go to
    ``assemble(files=)``.

    Raises:
        ValueError: both ``image`` and ``packages`` are given, or a package is not one requirement
            (``check_packages``).
    """
    if image is not None and packages:
        raise ValueError("A task machine names an image or packages, not both")
    check_packages(packages)
    if packages:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, PackageSpec(pypi_packages=tuple(packages))
    elif image is None:
        semantics, package_spec = CommandSemantics.SHELL_SIMULATOR, SHELLSIM_CPYTHON
    else:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, None
    return EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        command_semantics=semantics,
        docker_image=image,
        package_spec=package_spec,
        working_directory=workdir,
        setup_commands=tuple(setup),
        environment_variables=dict(env or {}),
    )


def check_packages(packages: Sequence[str]) -> None:
    """Raise ``ValueError`` unless each entry is one PyPI requirement without pip options."""
    malformed = [entry for entry in packages if not entry.strip() or entry.startswith("-") or {"\n", "\r"} & set(entry)]
    if malformed:
        raise ValueError(f"Packages are PyPI requirements, one per entry, without pip options: {malformed}")


def machine(
    *,
    startup_timeout: float | None,
    network: bool = False,
    resources: Resources = Resources(),
    user: str | None = None,
    cleanup_timeout: float | None = None,
) -> MachineSettings:
    """Deployment settings for the task machine or a shell grader's machine.

    ``user`` is the user model commands (or the shell grader) run as; ``None``
    keeps the image's user.
    """
    return MachineSettings(
        network=NetworkPolicy.ALLOW if network else NetworkPolicy.DENY,
        resources=resources,
        user=user,
        startup_timeout=startup_timeout,
        cleanup_timeout=cleanup_timeout,
    )


def rollout_limits(
    *,
    max_turns: int,
    model_turn_timeout: float | None,
    tool_turn_timeout: float | None,
    total_turn_timeout: float | None,
    attempt_timeout: float | None,
    verifier_timeout: float | None,
    cleanup_timeout: float,
) -> RolloutLimits:
    """The turn budget and deadlines of one attempt; the shell command timeout is the harness's (``rollout_engine``)."""
    return RolloutLimits(
        max_turns=max_turns,
        # No per-attempt token budget on trials: a trial is bounded by its deadlines and its turn budget,
        # and a cap here would end an attempt the evidence needs whole.
        max_total_tokens=None,
        model_turn_timeout=model_turn_timeout,
        tool_turn_timeout=tool_turn_timeout,
        total_turn_timeout=total_turn_timeout,
        attempt_timeout=attempt_timeout,
        verifier_timeout=verifier_timeout,
        cleanup_timeout=cleanup_timeout,
    )


def rollout_engine(factories: Mapping[str, MachineFactory], *, command_timeout: float | None) -> RolloutEngine:
    """A ``RolloutEngine`` on ``factories`` with RolloutEngine's shell harness registered as ``SHELL_HARNESS``.

    ``command_timeout`` bounds each shell command of a turn; ``None`` leaves it to the turn deadlines.
    ``RolloutEngine.check`` requires it to be shorter than a spec's ``tool_turn_timeout`` when both are set.
    """
    return RolloutEngine(factories, {SHELL_HARNESS: ShellHarness(command_timeout=command_timeout)})


def check(spec: TaskRolloutSpec, factories: Mapping[str, MachineFactory]) -> None:
    """Reject ``spec`` as ``RolloutEngine.check`` does on ``factories`` under the shell harness.

    The check is the engine's own, before any machine starts. The harness it checks against has no command
    timeout; the engine that runs ``spec`` checks its own again.

    Raises:
        ValueError: RolloutEngine rejects the spec.
        NotImplementedError: RolloutEngine cannot run the spec.
    """
    rollout_engine(factories, command_timeout=None).check(spec)


def reward_file(
    path: str, file_format: RewardFileFormat, key: str = "reward", pass_above: float | None = None
) -> FileReward:
    """Read the score from one reward file the grader writes."""
    return FileReward(files=(RewardFile(path=path, format=file_format, key=key),), pass_above=pass_above)


VERDICT_REWARD = reward_file(GRADER_VERDICT_PATH, RewardFileFormat.JSON)
"""How a Taskforge grader program reports: one JSON object ``{"reward": r, "detail": {"checks": [...]}}``
at ``GRADER_VERDICT_PATH`` (``spec.verdict.verdict_json``). TaskCompendium's ``FileReward`` makes ``detail``
the ``GradeResult.detail``."""


def grader_environment(task_image: str | None, *, env: Mapping[str, str] | None = None) -> EnvironmentRequirements:
    """The verifier machine of a script grader: the task image, or Taskforge's grader base without one.

    A Docker task grades in a fresh machine from its own image, so the grader sees the task's tools. A
    ShellSim task, or a task without a machine, grades in ``GRADER_BASE_IMAGE`` (CPython 3.12 with
    the standard library, ``sh``, ``setsid`` and ``tar``, as root in ``/app``).
    """
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=GRADER_BASE_IMAGE if task_image is None else task_image,
        environment_variables=dict(env or {}),
    )


def script_grader(
    argv: Sequence[str],
    reward: Reward,
    *,
    environment: EnvironmentRequirements,
    answer_path: str | None,
    timeout: float,
    files: Sequence[TaskResource] = (),
    cwd: str = "/app",
    env: Mapping[str, str] | None = None,
    collect: Sequence[VerifierCommand] = (),
    artifacts: Sequence[VerifierArtifact] = (),
) -> GraderPackage:
    """A grader command run in a separate verifier machine built from ``environment``.

    ``environment`` names a digest-pinned image (``grader_environment``) or, on Iris, a packages
    lock that RolloutEngine builds on its local backend. The verifier machine receives ``files``
    under ``/tests``, the task's agent-visible files at ``/``, the captured ``output_paths`` at
    their own absolute paths, the conversation as JSON chat messages at
    ``/tests/conversation.json``, and the extracted answer at ``answer_path``, which is ``None``
    for a file or workspace-state answer. ``collect`` commands run as root in the task machine and
    ``artifacts`` are copied from it first; an artifact a control may delete needs an explicit
    kind, not ``AUTO``. ``argv`` runs in ``cwd`` as the verifier machine's user and must finish
    within ``timeout`` and the attempt's ``RolloutLimits.verifier_timeout``.
    """
    grader = ScriptGrader(
        argv=tuple(argv),
        cwd=cwd,
        env=dict(env or {}),
        environment=environment,
        collect=tuple(collect),
        artifacts=tuple(artifacts),
        answer_path=answer_path,
        reward=reward,
        timeout=timeout,
    )
    return GraderPackage(grader, tuple(files))


def python_grader(
    script: str,
    config: Mapping[str, JsonValue],
    *,
    environment: EnvironmentRequirements,
    answer_path: str | None,
    timeout: float,
    files: Sequence[TaskResource] = (),
) -> GraderPackage:
    """A Python program graded as ``python3 /tests/grade.py`` in a verifier machine.

    ``script`` is ``/tests/grade.py`` and ``config`` its private ``/tests/config.json``; ``files``
    are further private inputs, paths relative to ``/tests``. No private file may have the same
    bytes as a solver-visible file (``assemble`` refuses it); the verifier machine already holds the
    solver-visible files at their own paths, so the program reads them there. The program reads the extracted
    answer at ``answer_path`` (``ANSWER_PATH`` for a text answer, ``None`` for a file answer) and
    captured output files at their absolute paths, never stdin, and writes its verdict
    (``spec.verdict.verdict_json``: the reward and one result per check) to ``GRADER_VERDICT_PATH``.

    Raises:
        ValueError: a file is named ``grade.py`` or ``config.json``, or two files share a path.
    """
    paths = [resource.path for resource in files]
    if len(set(paths)) != len(paths) or PYTHON_GRADER_FILES & set(paths):
        raise ValueError(f"Python grader files repeat a path or shadow grade.py or config.json: {sorted(paths)}")
    resources = (
        inline_resource("grade.py", script.encode()),
        inline_resource("config.json", json.dumps(dict(config), sort_keys=True).encode()),
        *files,
    )
    return script_grader(
        PYTHON_GRADER_ARGV,
        VERDICT_REWARD,
        environment=environment,
        answer_path=answer_path,
        timeout=timeout,
        files=resources,
    )


def answer_grader(spec: Spec, *, environment: EnvironmentRequirements | None = None) -> GraderPackage:
    """A verifyit grader of the final answer.

    Without ``environment`` the mode must be one verifyit grades in process (exact, numeric, mcq,
    math, ifeval, json_schema, xml_elements, csv_columns, structured_exact, predicted_action). Any
    other mode except the LLM judge runs ``python -c 'import verifyit ...'`` in a verifier machine
    built from ``environment``, whose image must contain verifyit. ``GRADER_BASE_IMAGE`` does not.
    RolloutEngine grades an LLM judge (``JudgeSpec``) on the host through the judge service whatever
    ``environment`` says, and ``grade_candidate`` serves no judge, so a judge cannot be prototyped at
    build time: grade with deterministic checks in ``python_grader`` instead.

    Raises:
        ValueError: the mode does not grade in process and no environment was given, or the
            environment is ``GRADER_BASE_IMAGE``.
    """
    mode = mode_of(spec)
    if environment is None and mode not in IN_PROCESS_MODES:
        raise ValueError(
            f"verifyit mode {mode.value!r} does not grade in process; give it a grading environment with "
            "verifyit installed, or use script_grader"
        )
    if environment is not None and environment.docker_image == GRADER_BASE_IMAGE:
        raise ValueError(
            f"verifyit mode {mode.value!r} cannot run on GRADER_BASE_IMAGE, which has no verifyit (it fails with "
            "\"No module named 'verifyit'\"); grade_candidate calls no model either, so write deterministic checks "
            "with python_grader"
        )
    return verifyit_package(spec, environment=environment)


def judge_grader(rubric: Rubric, reference: str) -> GraderPackage:
    """An LLM-as-judge grader: a model judge applies ``rubric`` to the answer, reading ``reference`` beside it.

    For deliverables whose prose is the product. The grader is a verifyit ``JudgeSpec`` naming the judge of
    the default strength (``spec.judge``, a stub the author never sets), with the rubric spec
    (``JUDGE_SPEC_PATH``) as its context. RolloutEngine grades it on the host through the judge service of
    ``TaskRolloutSpec.judge_connection`` or ``VERIFYIT_JUDGE_BASE_URL``, with no verifier machine: the
    grader-base environment it declares is there because TaskCompendium's ``VerifyitGrader`` requires one
    for a mode that does not grade in process, and nothing starts it (``grading_environment``).

    Raises:
        ValueError: ``reference`` is empty.
    """
    if not reference.strip():
        raise ValueError("an LLM-as-judge grader needs a non-empty reference answer")
    strength = JudgeStrength.DEFAULT
    spec = JudgeRubricSpec(rubric=rubric, judge_strength=strength)
    rubric_text = json.dumps(rubric.model_dump(mode="json"), indent=1)
    judge = JudgeSpec(
        references=(reference,),
        question=f"Rubric:\n{rubric_text}\n\n{JUDGE_QUESTION}",
        context=JUDGE_SPEC_PATH,
        model=JUDGE_MODELS[strength],
    )
    files = (inline_resource(JUDGE_SPEC_PATH, judge_spec_json(spec)),)
    return verifyit_package(judge, files, environment=grader_environment(None))


def decision_grader(
    rubric: Rubric, reference: str, *, state: Mapping[str, StateSource] = SOLVER_TURN_STATE
) -> DecisionGrader:
    """A decision-model grader: a decision model compares the solver's answer with ``reference`` against ``rubric``.

    ``reference`` is the gold answer, which validation also replays as the reference control. ``state`` is
    the decision model's state template; the default passes the deliverable, the rubric, the reference and
    the visible output of the solver's final turn, and a program changes it only deliberately. The grader has
    no judge strength and names no model: the host's decision client answers. It is uncalibrated (the
    "partially matches" bar) until ``sandbox_effects.calibrate_decision_grader`` sets its bar.

    Raises:
        pydantic.ValidationError: ``reference`` is empty, or ``state`` leaves out the solver's output.
    """
    return DecisionGrader(rubric=rubric, reference=reference, state=dict(state), acceptance_level=None, calibration={})


def assemble(
    task_id: str,
    instruction: str,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
    grader: GraderPackage | DecisionGrader,
    source: Source,
    *,
    environment: EnvironmentRequirements | None,
    files: Sequence[TaskResource] = (),
    output_paths: Sequence[str] = (),
    system: str | None = None,
    final_tools: Sequence[FunctionDefinition] = (),
    tags: Sequence[str] = (),
) -> TaskSpec:
    """Build a TaskSpec and reject one that RolloutEngine could not grade as intended.

    ``answer_format`` says how the model is asked for its final answer and how the answer is read
    from its conversation; a file or workspace-state task carries one too, which grading never
    reads. ``grader`` is a package, or a ``DecisionGrader``, which the TaskSpec carries in the form
    ``decision.task_package`` gives it until TaskSpec has a decision grader type. ``environment`` is the
    task machine (``requirements(...)``), or ``NO_MACHINE`` or ``None`` for a task without one. ``files``
    are the agent-visible files installed relative to the machine root; the grader's files stay private,
    so none of them may have the same bytes as an agent-visible file.

    Raises:
        ValueError: TaskSpec validation failed, files or output paths were given without a task
            machine, the answer format cannot carry the answer for this grader, a file or
            workspace-state answer has no output path or artifact to reach the grader, or private
            grader content is agent-visible. TaskSpec validation covers output paths that are not
            absolute and normalized or that reach the grader's private directories.
    """
    if environment == NO_MACHINE:
        environment = None
    if isinstance(grader, DecisionGrader):
        grader = task_package(grader)
    if environment is None and (files or output_paths):
        raise ValueError("Files and output paths require a task machine")
    if environment is not None and SHELL_CAPABILITY not in environment.capabilities:
        raise ValueError(f"A task machine needs the {SHELL_CAPABILITY!r} capability")
    if answer_type in MACHINE_ANSWERS and not output_paths and not _reads_task_machine(grader):
        raise ValueError(f"A {answer_type.value!r} answer needs output paths or grader artifacts to reach the grader")
    presentation = _presentation(instruction, system, final_tools, answer_format)
    spec = TaskSpec(
        id=task_id,
        context=presentation.context,
        environment_requirements=environment or EnvironmentRequirements(),
        final_tools=presentation.final_tools,
        output_paths=tuple(output_paths),
        answer_type=answer_type,
        answer_format=presentation.answer_format,
        grader=grader.grader,
        source=source,
        resources=ResourceGroups(worker=tuple(files), verifier=grader.resources),
        tags=tuple(tags),
    )
    if spec.answer_type in CONVERSATION_ANSWERS:
        # As the shell harness does at start: only an answer read from the conversation has a format to check.
        require_submission_compatibility(spec)
    # A private file whose bytes the solver can read is no longer private: the verifier machine
    # already sees the agent-visible files at their own paths, so the grader reads them there.
    visible = {resource_bytes(item) for item in spec.resources.all + spec.resources.worker}
    leaked = sorted(item.path for item in spec.resources.verifier if resource_bytes(item) in visible)
    if leaked:
        raise ValueError(f"Private verifier file content is also agent-visible: {leaked}")
    restored = TaskSpec.model_validate_json(spec.model_dump_json())
    if restored != spec:
        raise ValueError(f"Task {task_id!r} does not survive a JSON round trip")
    return restored


def lower(
    task: TaskSpec,
    *,
    host: MachineHost,
    task_machine: MachineSettings | None,
    verifier_machine: MachineSettings | None,
    limits: RolloutLimits,
    factories: Mapping[str, MachineFactory],
) -> TaskRolloutSpec:
    """The ``TaskRolloutSpec`` that runs ``task`` on ``host``'s ``factories`` under ``SHELL_HARNESS``.

    Every Taskforge ``TaskRolloutSpec`` is built here. Each machine runs the requirements
    ``host_requirements`` gives for ``factories``, the same rule for the task machine and the verifier
    machine, on the backend ``machine_backend`` picks for them; the lowered task carries those
    requirements and each machine's ``MachineRuntimeSpec.backend`` names its backend. A grader with an
    environment (a script grader, or a verifyit grader that is neither in process nor a judge) takes a
    verifier machine; no other grader does. The spec keeps the engine's default exact-token trace; a
    validation trial replaces it (``stage50_validate.trials.EngineSettings.apply``).

    Raises:
        ValueError: A machine is given where the task has none or missing where it has one, or
            RolloutEngine rejects the lowered spec (``check``).
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    if (task_machine is None) != (SHELL_CAPABILITY not in task.environment_requirements.capabilities):
        raise ValueError("A task machine is given exactly when the task has the shell capability")
    if (verifier_machine is None) != (grading_environment(task) is None):
        raise ValueError("A verifier machine is given exactly when the task's grader has an environment")
    task = _hosted_task(task, factories)
    grading = grading_environment(task)
    spec = TaskRolloutSpec(
        task=task,
        harness=SHELL_HARNESS,
        task_machine=(
            None if task_machine is None else machine_runtime(task_machine, task.environment_requirements, host)
        ),
        verifier_machine=(
            None if verifier_machine is None or grading is None else machine_runtime(verifier_machine, grading, host)
        ),
        limits=limits,
    )
    check(spec, factories)
    return spec


def machine_runtime(
    settings: MachineSettings, requirements: EnvironmentRequirements, host: MachineHost
) -> MachineRuntimeSpec:
    """The ``MachineRuntimeSpec`` of a machine with ``requirements`` on ``host``.

    ``requirements`` are the machine's ``host_requirements``. ``lower`` builds every task and verifier
    machine with it; a builder prototyping a machine outside a task rollout spec uses it too, so the
    backend choice stays in one place.
    """
    return MachineRuntimeSpec(
        backend=machine_backend(requirements, host).value,
        network=settings.network,
        cpus=settings.resources.cpus,
        memory_mb=settings.resources.memory_mb,
        storage_mb=settings.resources.storage_mb,
        gpus=settings.resources.gpus,
        user=settings.user,
        startup_timeout=settings.startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def machine_backend(requirements: EnvironmentRequirements, host: MachineHost) -> Backend:
    """The host's container backend for an image, the local backend for Linux processes without one
    (a packages lock or a package spec), else ShellSim."""
    if requirements.docker_image is not None:
        return container_backend(host)
    if requirements.command_semantics == CommandSemantics.LINUX_PROCESS:
        return Backend.LOCAL
    return Backend.SHELLSIM


def host_requirements(
    requirements: EnvironmentRequirements, factories: Mapping[str, MachineFactory], writes: Sequence[str] = ()
) -> EnvironmentRequirements:
    """``requirements`` as a host with ``factories`` runs them.

    With a local factory, a machine on ``GRADER_BASE_IMAGE`` runs ``LOCAL_CPYTHON`` on the local
    backend, unless one of the absolute paths in ``writes`` (its files and its working directory)
    falls under ``LOCAL_READ_ONLY_ROOTS``, in which case it keeps the image on the host's container
    backend, as it does without a local factory. A package machine (``requirements(packages=...)``)
    runs on the local backend only, its packages installed into a uv venv at machine start; on a
    host without a local factory, ``RolloutEngine.check`` rejects it as an unknown factory. Every
    other requirement is unchanged.
    """
    if requirements.docker_image != GRADER_BASE_IMAGE or Backend.LOCAL.value not in factories:
        return requirements
    if any(_under(path, LOCAL_READ_ONLY_ROOTS) for path in writes):
        return requirements
    return requirements.model_copy(update={"docker_image": None, "package_spec": LOCAL_CPYTHON})


def _under(path: str, roots: Sequence[str]) -> bool:
    parts = PurePosixPath(path).parts
    return any(parts[: len(PurePosixPath(root).parts)] == PurePosixPath(root).parts for root in roots)


def _hosted_task(task: TaskSpec, factories: Mapping[str, MachineFactory]) -> TaskSpec:
    """``task`` with its task machine's and its grader's requirements as ``host_requirements`` gives them."""
    requirements = task.environment_requirements
    task_writes = [f"/{resource.path}" for resource in (*task.resources.all, *task.resources.worker)]
    if requirements.working_directory is not None:
        task_writes.append(requirements.working_directory)
    update: dict[str, object] = {"environment_requirements": host_requirements(requirements, factories, task_writes)}
    grading = grading_environment(task)
    if grading is not None:
        grader_writes = [f"/{resource.path}" for resource in task.resources.verifier]
        if isinstance(task.grader, ScriptGrader):
            grader_writes.append(task.grader.cwd)
        update["grader"] = task.grader.model_copy(
            update={"environment": host_requirements(grading, factories, grader_writes)}
        )
    return task.model_copy(update=update)


def _reads_task_machine(grader: GraderPackage) -> bool:
    return isinstance(grader.grader, ScriptGrader) and bool(grader.grader.artifacts or grader.grader.collect)


def _presentation(
    instruction: str, system: str | None, final_tools: Sequence[FunctionDefinition], answer_format: AnswerFormat
) -> _Presentation:
    return _Presentation(
        context=_conversation(instruction, system), final_tools=tuple(final_tools), answer_format=answer_format
    )


def _conversation(instruction: str, system: str | None = None) -> ConversationInput:
    system_events = () if system is None else (TextMessage(role="system", content=system),)
    return ConversationInput(events=(*system_events, TextMessage(role="user", content=instruction)))
