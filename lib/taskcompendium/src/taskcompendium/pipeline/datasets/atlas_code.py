# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests and CodeNet policies using the shared executable task boundary."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.executable_tasks import normalize
from taskcompendium.pipeline.datasets.executable_tasks import verification_report as executable_verification_report
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter, with_raw_converter
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
from taskcompendium.runtime.grading import GRADING_MEMORY_MB, GRADING_TIMEOUT

CODENET_EXIT_PARITY = CheckResult(
    check="source_exit_status_parity",
    status=CheckStatus.UNSUPPORTED,
    detail="CodeNet source rejects nonzero solution exit codes; shared stdio scorer currently compares stdout "
    "without checking exit status. Passing controls do not establish source-grader parity.",
)


def verification_report(
    task: TaskSpec, name: str, timeout: float = GRADING_TIMEOUT, memory_mb: int = GRADING_MEMORY_MB
) -> VerificationReport:
    report = executable_verification_report(task, timeout=timeout, memory_mb=memory_mb)
    if name == "codenet":
        return VerificationReport(checks=[*report.checks, CODENET_EXIT_PARITY], rollouts=report.rollouts)
    return report


def pipeline(
    name: str,
    image: str,
    *,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> TaskPipeline:
    """Build CodeNet or contest conversion and sandbox controls."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, image)

    def checks(task: TaskSpec) -> VerificationReport:
        return verification_report(task, name, timeout, memory_mb)

    base = TaskPipeline(
        normalize=normalize_row,
        rubric=RUBRICS[name],
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": image, "timeout": timeout, "memory_mb": memory_mb},
            run=checks,
        ),
    )
    return with_raw_converter(base, converter, converter_revision)


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
