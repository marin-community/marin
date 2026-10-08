# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize converted shell, competitive-programming, and Python test tasks."""

import asyncio
import base64
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial

from shellbox.machine import Backend, MachineFactory, MachineSpec
from verifyit.spec import spec_from_table

from taskcompendium.datasets.raw_conversion import RawConverter, with_raw_converter
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionCall,
    GradingAttempt,
    OutputDirectory,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grader_workspace,
    require_compatible_backend,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.runtime.grading import GRADING_TIMEOUT
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import BASH, INTERFACE, ShellFactory, machine_spec_identity
from taskcompendium.runtime.task_grading import sandbox_grade

DEFAULT_OUTPUT_PATHS = ("/app/solution.py", "/app/solution.cpp")
# Controls submit files, not an answer; the grading attempt still needs a final assistant message.
CONTROL_RESPONSE = TextMessage(role="assistant", content="The submission is in the workspace.")


@dataclass(frozen=True)
class SubmissionControl:
    """A supplied submission and its source-declared reward expectation."""

    name: str
    files: dict[str, bytes]
    expected_reward: float


COMPLETION_CRITERIA = (
    "Assess the missing continuation using the public code prefix; identify missing cross-file context or ambiguity.",
    "The source grader awards full credit for an exact normalized match, a reference prefix with up to three extra "
    "lines, or only a matching first identifier. It awards 0.25 for any identifier-containing fragment of at most "
    "400 bytes. Assess these weak tiers explicitly; do not assume passing the grader establishes correctness.",
    "The public write_solution helper writes /app/solution.txt. Reference fragments are private controls, not "
    "executable solution scripts, and incomplete fragments need not compile as standalone files.",
)

CRITERIA = {
    "inferredbugs": (
        "The source supplies a buggy file in the public prompt and asks for its complete corrected replacement "
        "at an explicit path. No repository checkout, build context, or executable oracle is supplied.",
        "The source Python grader uses structural text and regex checks without compiling or executing the "
        "submitted program. Identify specific false positives, including matching tokens in comments, empty "
        "method bodies, or null guards with the wrong polarity; generic wrong-file controls may miss these.",
        "Preserve the original scoring when assessing quality. Do not infer behavioral correctness from a passing "
        "structural check or invent missing project context.",
    ),
    "crosscodeeval_csharp": COMPLETION_CRITERIA,
    "crosscodeeval_python": COMPLETION_CRITERIA,
    "crosscodeeval_typescript": (
        *COMPLETION_CRITERIA,
        "The source retains Python metadata and a Python fence around TypeScript code. Record concrete language "
        "or context defects without silently correcting the source or changing its comparison rules.",
    ),
    "crosscodeeval_java": (
        "Assess the missing Java continuation using only the supplied public code prefix. No repository or "
        "cross-file context is provided; identify a concrete missing dependency or ambiguity when it matters.",
        "The grader compares the exact reference fragment after stripping trailing whitespace per line and "
        "removing blank lines. Case, leading whitespace and internal whitespace remain significant.",
        "The fragment need not compile as a complete Java file. Do not invent additional context or require "
        "the legacy Python import tests, which are not used by the source grader.",
        "Give a concrete alternative continuation when exact reference matching rejects an equally valid answer; "
        "a passing golden establishes comparator agreement, not a uniquely determined continuation.",
    ),
    "nl2bash": (
        "The public seed/setup files must recreate the command's input files. Flag unavailable tools or inputs.",
        "The private comparator is a normalized multiset: ordering is ignored and non-error extra lines are allowed. "
        "Check whether the public request requires distinctions that this comparator cannot grade.",
        "The expected output is an oracle capture, not an instruction to print that output without doing the work.",
        "Every mandatory package or side-effect deliverable needs a public specification or provided helper. "
        "An undefined 'sandboxes task' package is a missing-context defect even if only stdout is graded. "
        "Test a literal minimal answer to the public request; unstated oracle output prefixes are a grading mismatch.",
    ),
    "taco": (
        "Check whether this is a complete stdin/stdout program problem or a function-call problem without a driver.",
        "Inspect public examples versus hidden case format and expected outputs; flag contradictions or leaked gold.",
        "Check numerical tolerances, multi-case parsing, and that a meaningful set of hidden cases exists.",
    ),
    "codeforces": (
        "Check the complete problem statement, constraints, input/output format, and examples against private cases.",
        "A special judge may allow many correct outputs; exact comparison must not replace that semantic contract.",
        "Flag unsupported language promises, absent input, inconsistent numerical tolerances, or sample-only tests.",
    ),
    "unitsyn": (
        "Check that the requested Python API, filenames, return values, and exceptions agree with the private tests.",
        "Look for tests that invent unstated behavior, incomplete definitions, missing fixtures, or dependencies.",
        "A passing oracle proves compatibility with the supplied tests, not that the tests cover the specification.",
        "A public example that contradicts the written rule is a defect even if the private tests follow the rule. "
        "Do not dismiss incorrect example comments, off-by-one boundaries or required unspecified behavior as minor.",
    ),
}


@dataclass(frozen=True)
class ExecutableConversion:
    """Converter and sandbox settings for an executable source."""

    image: str
    output_paths: tuple[str, ...]
    converter: RawConverter
    converter_revision: str
    timeout: float
    machine_factory: MachineFactory
    machine_spec: MachineSpec
    worker_image: str | None
    output_directories: tuple[OutputDirectory, ...] = ()
    executable_worker_paths: tuple[str, ...] = ()

    @property
    def verification_parameters(self) -> dict:
        machine = machine_spec_identity(self.machine_spec)
        return {
            "image": self.image,
            "output_paths": self.output_paths,
            "output_directories": [selection.model_dump(mode="json") for selection in self.output_directories],
            "executable_worker_paths": self.executable_worker_paths,
            "timeout": self.timeout,
            "machine": machine,
            "backend": self.machine_factory.backend.value,
            "worker_image": self.worker_image,
            "factory": f"{type(self.machine_factory).__module__}.{type(self.machine_factory).__qualname__}",
        }


def normalize(
    row: RawRow,
    image: str,
    *,
    output_paths: tuple[str, ...],
    output_directories: tuple[OutputDirectory, ...] = (),
    executable_worker_paths: tuple[str, ...] = (),
) -> TaskSpec | ImportRejection:
    """Import a converter result while keeping tests and oracle code private."""
    rejection = row.data.get("conversion_rejection")
    if isinstance(rejection, dict):
        return ImportRejection.model_validate(rejection)
    converted = row.data.get("converted")
    if not isinstance(converted, dict):
        return ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR,
            reason="missing_conversion",
            detail="Run the source converter binding first",
        )
    instruction = converted["instruction"]
    spec = converted["grader_spec"]
    worker = []
    oracle = []
    trusted = []
    for path, encoded in converted["data_files"].items():
        data = base64.b64decode(encoded, validate=True)
        destination = trusted if path.startswith("tests/") else worker
        if path.startswith("tests/setup_files/"):
            destination = oracle
        resource = inline_resource(path.removeprefix("tests/") if destination is trusted else path, data)
        if destination is worker and path in executable_worker_paths:
            resource = resource.model_copy(update={"mode": "0755"})
        destination.append(resource)
    oracle.extend(
        inline_resource(path, base64.b64decode(encoded, validate=True))
        for path, encoded in converted["control_files"].items()
    )
    package = verifyit_package(
        spec_from_table(spec),
        tuple(trusted),
        environment=EnvironmentRequirements(
            docker_image=image,
            compatible_backends=(Backend.DOCKER, Backend.GVISOR, Backend.QEMU),
        ),
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            compatible_backends=(Backend.DOCKER, Backend.GVISOR, Backend.QEMU),
            docker_image=image,
            capabilities=("shell", "filesystem", "python3") if output_directories else ("shell", "filesystem"),
            tool_providers={"shell": ProviderRequirement(action_interface=INTERFACE, initial_state={})},
        ),
        interaction_tools=(BASH,),
        resources=ResourceGroups(worker=tuple(worker), oracle=tuple(oracle), verifier=package.resources),
        output_paths=output_paths,
        output_directories=output_directories,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
    )


def executable_check_suite(conversion: ExecutableConversion, *, revision: str) -> CheckSuite:
    """Build executable controls for the configured Shellbox runtime."""
    return CheckSuite(
        id="isolated-executable-controls",
        revision=revision,
        parameters=conversion.verification_parameters,
        run=partial(
            verification_report,
            factory=conversion.machine_factory,
            timeout=conversion.timeout,
            machine_spec=conversion.machine_spec,
        ),
    )


def policy(name: str, conversion: ExecutableConversion) -> TaskPolicy:
    """Build executable normalization and controls with explicit sandbox limits."""
    return converted_policy(
        conversion,
        rubric=ReviewRubric(
            id=f"{name}-answerability",
            version="2" if name in {"nl2bash", "unitsyn"} else "1",
            criteria=(
                "Judge whether the underlying request is comprehensible and answerable using the public inputs.",
                "Private tests and oracle solutions are review evidence and must never be shown to the solving actor.",
                "Report a concrete defect rather than penalizing difficulty. Missing oracle controls imply verification "
                "uncertainty, not an automatically bad problem.",
                *CRITERIA[name],
            ),
        ),
        revision="2" if name == "nl2bash" else "3",
    )


def converted_policy(conversion: ExecutableConversion, *, rubric: ReviewRubric, revision: str) -> TaskPolicy:
    """Bind a source converter, review rubric and isolated executable controls."""
    base = TaskPolicy(
        normalize=partial(
            normalize,
            image=conversion.image,
            output_paths=conversion.output_paths,
            output_directories=conversion.output_directories,
            executable_worker_paths=conversion.executable_worker_paths,
        ),
        rubric=rubric,
        check_suite=executable_check_suite(conversion, revision=revision),
    )
    return with_raw_converter(base, conversion.converter, conversion.converter_revision)


def verification_report(
    task: TaskSpec,
    *,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float = GRADING_TIMEOUT,
    controls: tuple[SubmissionControl, ...] | None = None,
) -> VerificationReport:
    return asyncio.run(
        executable_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout, controls=controls)
    )


async def grade_files(
    task: TaskSpec,
    files: Mapping[str, bytes],
    factory: MachineFactory,
    *,
    machine_spec: MachineSpec,
    timeout: float,
) -> GradeResult:
    """Grade captured workspace files in a fresh machine."""
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, CONTROL_RESPONSE)), files)
    return await sandbox_grade(task, attempt, factory, machine_spec, timeout=timeout)


async def executable_checks(
    task: TaskSpec,
    *,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float = GRADING_TIMEOUT,
    controls: tuple[SubmissionControl, ...] | None = None,
) -> VerificationReport:
    """Check declared submission rewards and the oracle in fresh machines."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader | ScriptGrader) or grader.environment is None:
        raise ValueError("Executable controls require a grader with a grading environment")
    require_compatible_backend(grader.environment, factory.backend)
    checks = []
    if controls is None:
        path = task.output_paths[0]
        wrong = b"raise RuntimeError('__negative_control__')\n" if path.endswith(".py") else b"unexpected error\n"
        controls = (
            SubmissionControl("missing_submission", {}, 0.0),
            SubmissionControl("empty_submission", {path: b""}, 0.0),
            SubmissionControl("wrong_submission", {path: wrong}, 0.0),
        )
    for control in controls:
        result = await grade_files(task, control.files, factory, machine_spec=machine_spec, timeout=timeout)
        status = (
            CheckStatus.INFRA_ERROR
            if result.status == Outcome.INFRA_ERROR
            else (
                CheckStatus.PASS
                if result.status == Outcome.GRADED and result.reward == control.expected_reward
                else CheckStatus.FAIL
            )
        )
        checks.append(
            CheckResult(
                check=control.name,
                status=status,
                detail=(
                    f"{result.status}: reward={result.reward}; expected={control.expected_reward}; {result.error or ''}"
                ),
            )
        )
    oracle = next((resource for resource in task.resources.oracle if resource.path == "solution/solve.sh"), None)
    if oracle is None:
        checks.append(
            CheckResult(check="oracle", status=CheckStatus.SKIPPED, detail="Source ships no executable oracle solution")
        )
        return VerificationReport(checks)
    shell_factory = ShellFactory(
        machine_factory=factory,
        machine_spec=replace(machine_spec, workdir="/"),
        backend_identity={"image": grader.environment.docker_image},
        command_timeout=timeout,
        output_limit_bytes=1_048_576,
    )
    try:
        environment = await replace(shell_factory, mounted_roles=("worker", "oracle")).create(task)
    except (RuntimeError, OSError) as error:
        checks.append(CheckResult(check="oracle", status=CheckStatus.INFRA_ERROR, detail=str(error)))
        return VerificationReport(checks)
    try:
        workspace = shlex.quote(grader_workspace(grader))
        execution = json.loads(
            await environment.step(
                FunctionCall(
                    name="Bash",
                    arguments={"command": f"mkdir -p {workspace} && cd {workspace} && bash /solution/solve.sh"},
                )
            )
        )
        if execution["exit_code"] != 0:
            checks.append(
                CheckResult(
                    check="oracle",
                    status=CheckStatus.FAIL,
                    detail=f"Oracle command failed: {json.dumps(execution)[:2000]}",
                )
            )
            return VerificationReport(checks)
        evidence = await environment.evidence()
    finally:
        await environment.close()
    result = await grade_files(task, evidence.files, factory, machine_spec=machine_spec, timeout=timeout)
    status = (
        CheckStatus.INFRA_ERROR
        if result.status == Outcome.INFRA_ERROR
        else (CheckStatus.PASS if result.status == Outcome.GRADED and result.reward == 1.0 else CheckStatus.FAIL)
    )
    checks.append(
        CheckResult(
            check="oracle", status=status, detail=f"{result.status}: reward={result.reward}; {result.error or ''}"
        )
    )
    return VerificationReport(checks)
