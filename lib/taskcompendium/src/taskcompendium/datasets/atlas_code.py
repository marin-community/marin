# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests and CodeNet policies using the shared executable task boundary."""

import asyncio

from shellbox.machine import MachineFactory, MachineSpec
from verifyit.spec import StdioSpec

from taskcompendium.datasets.executable_tasks import ExecutableConversion, grade_files, normalize
from taskcompendium.datasets.executable_tasks import verification_report as executable_verification_report
from taskcompendium.datasets.raw_conversion import with_raw_converter
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ResourceGroups, TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.runtime.grading import GRADING_TIMEOUT
from taskcompendium.runtime.resources import inline_resource


async def exit_status_result(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> CheckResult:
    """Probe the installed stdio runner without changing the source's private cases."""
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StdioSpec)
    resources = tuple(
        inline_resource(f"{spec.cases}/{kind}_{index}.txt", b"exit-status-control\n")
        for index in range(max(2, spec.min_cases))
        for kind in ("input", "output")
    )
    probe = task.model_copy(update={"resources": ResourceGroups(verifier=resources)})
    result = await grade_files(
        probe,
        {task.output_paths[0]: b"print('exit-status-control', flush=True)\nraise SystemExit(7)\n"},
        factory,
        machine_spec=machine_spec,
        timeout=timeout,
    )
    if result.status == Outcome.INFRA_ERROR:
        status = CheckStatus.INFRA_ERROR
    elif (
        result.status == Outcome.GRADED
        and result.reward == 0.0
        and result.detail is not None
        and result.detail.get("reason") == "runtime_error"
        and result.detail.get("returncode") == 7
    ):
        status = CheckStatus.PASS
    else:
        status = CheckStatus.UNSUPPORTED
    return CheckResult(
        check="source_exit_status_parity",
        status=status,
        detail=f"Matching stdout followed by exit 7: {result.status}; reward={result.reward}; "
        f"detail={result.detail}; error={result.error}",
    )


def verification_report(
    task: TaskSpec,
    name: str,
    *,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float = GRADING_TIMEOUT,
) -> VerificationReport:
    report = executable_verification_report(task, factory=factory, machine_spec=machine_spec, timeout=timeout)
    if name == "codenet":
        parity = asyncio.run(exit_status_result(task, factory=factory, machine_spec=machine_spec, timeout=timeout))
        return VerificationReport(checks=[*report.checks, parity], rollouts=report.rollouts)
    return report


def policy(name: str, conversion: ExecutableConversion) -> TaskPolicy:
    """Build CodeNet or contest conversion and sandbox controls."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(
            row, conversion.image, output_paths=conversion.output_paths, output_directories=conversion.output_directories
        )

    def checks(task: TaskSpec) -> VerificationReport:
        return verification_report(
            task,
            name,
            factory=conversion.machine_factory,
            timeout=conversion.timeout,
            machine_spec=conversion.machine_spec,
        )

    base = TaskPolicy(
        normalize=normalize_row,
        rubric=RUBRICS[name],
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="4" if name == "codenet" else "3",
            parameters=conversion.verification_parameters,
            run=checks,
        ),
    )
    return with_raw_converter(base, conversion.converter, conversion.converter_revision)


RUBRICS: dict[str, ReviewRubric] = {
    "code_contests": ReviewRubric(
        id="code_contests-answerability",
        version="1",
        criteria=(
            "Check the full stdin/stdout problem, constraints, examples, and private cases for "
            "agreement. Absent diagrams, interactive protocols without an interactor, and contradictory"
            " outputs are defects.",
            "Inspect numerical error clauses and special-output semantics. Exact line comparison cannot"
            " grade an arbitrary valid construction unless the task specifies a canonical output.",
            "The source supplies no oracle solution. Assess static coherence from the problem and cases"
            " anyway; missing executable positive controls and inability to solve quickly are not "
            "quality defects.",
        ),
    ),
    "codenet": ReviewRubric(
        id="codenet-answerability",
        version="2",
        criteria=(
            "An example or private input contradicting explicit public bounds is a task defect even if the "
            "main algorithm is clear and the oracle passes. Do not treat that contradiction as a minor issue.",
            "Check that the Python stdin/stdout instruction agrees with private inputs, outputs, and "
            "oracle. The grader compares whitespace-separated tokens and requires at least two cases.",
            "Check every visible private input against the public domain: extra values, too few values,"
            " and violated size bounds can penalize a correct program even when the supplied oracle "
            "passes.",
            "Check whether rewritten statements preserve the original algorithmic problem. Flag "
            "invented behavior, incorrect examples, missing definitions, and inconsistent reference "
            "outputs.",
        ),
    ),
}
