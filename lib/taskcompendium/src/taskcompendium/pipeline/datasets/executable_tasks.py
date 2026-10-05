# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize converted shell, competitive-programming, and Python test tasks."""

import asyncio
import base64
import json
import shlex
from dataclasses import dataclass, replace
from functools import partial

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import DockerImage, MachineSpec, NetworkPolicy
from verifyit.spec import spec_from_table

from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, ProviderRequirement, RegistryImage
from taskcompendium.grader import grader_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionCall,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter, with_raw_converter
from taskcompendium.pipeline.datasets.shell_files import BASH
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.runtime.grading import GRADING_MEMORY_MB, GRADING_TIMEOUT, grade_submission
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import INTERFACE, ShellFactory

CRITERIA = {
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
    converter: RawConverter
    converter_revision: str
    timeout: float
    memory_mb: int


def normalize(row: RawRow, image: str) -> TaskSpec | ImportRejection:
    """Import a converter result while keeping tests and oracle code private."""
    rejection = row.data.get("conversion_rejection")
    if isinstance(rejection, dict):
        return ImportRejection.model_validate(rejection)
    converted = row.data.get("converted")
    if not isinstance(converted, dict):
        return ImportRejection(reason="missing_conversion", detail="Run the source converter binding first")
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
        destination.append(resource)
    oracle.extend(
        inline_resource(path, base64.b64decode(encoded, validate=True))
        for path, encoded in converted["control_files"].items()
    )
    paths = ("/output/command_capture.txt",) if spec["mode"] == "script" else ("/app/solution.py", "/app/solution.cpp")
    package = grader_package(spec_from_table(spec), tuple(trusted))
    verifier = package.model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.DOCKER, image=RegistryImage(reference=image), workdir=spec["workspace"]
            )
        }
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
        ),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.SHELLSIM,
            files=tuple(worker),
            tool_providers={"shell": ProviderRequirement(action_interface=INTERFACE, initial_state={})},
        ),
        interaction_tools=(BASH,),
        oracle_files=tuple(oracle),
        output_paths=paths,
        answer_type=AnswerType.FILE,
        verifier=verifier,
    )


def pipeline(name: str, conversion: ExecutableConversion) -> TaskPipeline:
    """Build executable normalization and controls with explicit sandbox limits."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, conversion.image)

    base = TaskPipeline(
        normalize=normalize_row,
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
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": conversion.image, "timeout": conversion.timeout, "memory_mb": conversion.memory_mb},
            run=partial(verification_report, timeout=conversion.timeout, memory_mb=conversion.memory_mb),
        ),
    )
    return with_raw_converter(base, conversion.converter, conversion.converter_revision)


def verification_report(
    task: TaskSpec, *, timeout: float = GRADING_TIMEOUT, memory_mb: int = GRADING_MEMORY_MB
) -> VerificationReport:
    return asyncio.run(executable_checks(task, timeout=timeout, memory_mb=memory_mb))


async def executable_checks(
    task: TaskSpec, *, timeout: float = GRADING_TIMEOUT, memory_mb: int = GRADING_MEMORY_MB
) -> VerificationReport:
    """Check missing, empty, wrong, and oracle submissions in fresh machines."""
    environment = task.verifier.environment
    if environment is None or not isinstance(environment.image, RegistryImage):
        raise ValueError("Executable controls require a pinned image")
    image = environment.image.reference
    checks = []
    path = task.output_paths[0]
    wrong = b"raise RuntimeError('__negative_control__')\n" if path.endswith(".py") else b"unexpected error\n"
    for name, files in (
        ("missing_submission", {}),
        ("empty_submission", {path: b""}),
        ("wrong_submission", {path: wrong}),
    ):
        result = await grade_submission(task, files, DockerMachineFactory(), timeout=timeout, memory_mb=memory_mb)
        status = (
            CheckStatus.INFRA_ERROR
            if result.status == Outcome.INFRA_ERROR
            else (CheckStatus.PASS if result.reward == 0.0 else CheckStatus.FAIL)
        )
        checks.append(
            CheckResult(
                check=name, status=status, detail=f"{result.status}: reward={result.reward}; {result.error or ''}"
            )
        )
    oracle = next((resource for resource in task.oracle_files if resource.path == "/solution/solve.sh"), None)
    if oracle is None:
        checks.append(
            CheckResult(
                check="oracle", status=CheckStatus.UNSUPPORTED, detail="Source ships no executable oracle solution"
            )
        )
        return VerificationReport(checks)
    spec = json.loads(task.verifier.parameters_json)
    factory = ShellFactory(
        machine_factory=DockerMachineFactory(),
        machine_spec=MachineSpec(
            DockerImage(image),
            workdir="/",
            network=NetworkPolicy.DENY,
            memory_mb=memory_mb,
        ),
        backend_identity={"backend": "docker", "image": image},
        command_timeout=timeout,
        output_limit_bytes=1_048_576,
    )
    try:
        environment = await replace(factory, mounted_roles=("worker", "oracle")).create(task)
    except (RuntimeError, OSError) as error:
        checks.append(CheckResult(check="oracle", status=CheckStatus.INFRA_ERROR, detail=str(error)))
        return VerificationReport(checks)
    try:
        workspace = shlex.quote(spec["workspace"])
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
    result = await grade_submission(task, evidence.files, DockerMachineFactory(), timeout=timeout, memory_mb=memory_mb)
    status = (
        CheckStatus.INFRA_ERROR
        if result.status == Outcome.INFRA_ERROR
        else (CheckStatus.PASS if result.reward == 1.0 else CheckStatus.FAIL)
    )
    checks.append(
        CheckResult(
            check="oracle", status=status, detail=f"{result.status}: reward={result.reward}; {result.error or ''}"
        )
    )
    return VerificationReport(checks)
