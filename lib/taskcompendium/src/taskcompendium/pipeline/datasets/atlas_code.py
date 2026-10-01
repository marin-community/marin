# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests and CodeNet snapshots using the shared executable task boundary."""

from pathlib import Path

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.executable_tasks import REVISION, normalize
from taskcompendium.pipeline.datasets.executable_tasks import verification_report as executable_verification_report
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
    VerificationReport,
)

CONFIGS = {
    "code_contests": "DCAgent__code-contests-noblock",
    "codenet": "laion__exp_rpt_codenet-python-v4",
}
CODENET_EXIT_PARITY = CheckResult(
    check="source_exit_status_parity",
    status=CheckStatus.UNSUPPORTED,
    detail="CodeNet source rejects nonzero solution exit codes; shared stdio scorer currently compares stdout "
    "without checking exit status. Passing controls do not establish source-grader parity.",
)


def verification_report(task: TaskSpec, name: str) -> VerificationReport:
    report = executable_verification_report(task)
    if name == "codenet":
        return VerificationReport(checks=[*report.checks, CODENET_EXIT_PARITY], rollouts=report.rollouts)
    return report


def recipe(
    name: str, snapshot: Path, image: str, *, rubric: ReviewRubric, timeout: float, memory_mb: int
) -> DatasetRecipe:
    """Bind a converted snapshot to static quality review and sandbox diagnostics."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, image, timeout, memory_mb)

    def checks(task: TaskSpec) -> VerificationReport:
        return verification_report(task, name)

    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", REVISION, CONFIGS[name], "train", str(snapshot)),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": image, "timeout": timeout, "memory_mb": memory_mb},
            run=checks,
        ),
    )
