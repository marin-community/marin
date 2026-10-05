# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests and CodeNet sources using the shared executable task boundary."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.executable_tasks import REVISION, normalize
from taskcompendium.pipeline.datasets.executable_tasks import verification_report as executable_verification_report
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter, with_raw_converter
from taskcompendium.pipeline.datasets.source_definitions import TASKTROVE_DATASET, tasktrove_inputs, tasktrove_source
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.runtime.grading import GRADING_MEMORY_MB, GRADING_TIMEOUT

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


def verification_report(
    task: TaskSpec, name: str, timeout: float = GRADING_TIMEOUT, memory_mb: int = GRADING_MEMORY_MB
) -> VerificationReport:
    report = executable_verification_report(task, timeout=timeout, memory_mb=memory_mb)
    if name == "codenet":
        return VerificationReport(checks=[*report.checks, CODENET_EXIT_PARITY], rollouts=report.rollouts)
    return report


def recipe(
    name: str,
    image: str,
    *,
    rubric: ReviewRubric,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    """Bind a converted source to static quality review and sandbox diagnostics."""

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, image)

    def checks(task: TaskSpec) -> VerificationReport:
        return verification_report(task, name, timeout, memory_mb)

    source_recipe = DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=HFSource(TASKTROVE_DATASET, REVISION, CONFIGS[name], "train"),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        inputs=tasktrove_inputs(CONFIGS[name], REVISION),
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": image, "timeout": timeout, "memory_mb": memory_mb},
            run=checks,
        ),
    )
    return with_raw_converter(source_recipe, converter, converter_revision)


SOURCES = {
    "code_contests": tasktrove_source(
        config=CONFIGS["code_contests"],
        revision=REVISION,
        rubric=ReviewRubric(
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
    ),
    "codenet": tasktrove_source(
        config=CONFIGS["codenet"],
        revision=REVISION,
        rubric=ReviewRubric(
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
    ),
}


def recipe_for_source(
    name: str,
    image: str,
    *,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    source = SOURCES[name]
    return recipe(
        name,
        image,
        rubric=source.rubric,
        converter=converter,
        converter_revision=converter_revision,
        timeout=timeout,
        memory_mb=memory_mb,
    )
