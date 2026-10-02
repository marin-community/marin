# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron stdin/stdout coding source and case-alignment rubric."""

from dataclasses import replace

from verifyit.modes.extract import collapse_whitespace

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets import python_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import CheckResult, CheckStatus, DatasetRecipe, ReviewRubric, VerificationReport

CONFIG = "laion__nemotron-gym-competitive-coding-v2"
RUBRIC = ReviewRubric(
    id="competitive-coding-answerability",
    version="1",
    criteria=(
        "Check the complete problem statement, constraints, stdin format, stdout format, and public examples.",
        "Compare private case inputs and outputs with the public rules; identify wrong keys and missing drivers.",
        "The source grades exact line output with trailing whitespace normalized and requires all cases to pass.",
        "Flag tasks needing a special judge, numerical tolerance, or multiple valid outputs that exact grading rejects.",
        "Public examples are legitimate cases. A sample-only set limits coverage; it does not by itself show leaked "
        "gold or a defective task. Report coverage separately from content quality.",
        "Missing oracle controls imply verification uncertainty, not an automatically bad programming problem.",
    ),
)


def recipe(image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    source_recipe = python_tasks.recipe(
        "competitive_coding",
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )

    suite = source_recipe.check_suite
    assert suite is not None

    def checks(task: TaskSpec) -> VerificationReport:
        report = suite.run(task)
        inputs = [
            resource.data().decode() for resource in task.resources if resource.path.startswith("/tests/cases/input_")
        ]
        instruction = task.context.events[0]
        assert isinstance(instruction, TextMessage)
        prompt = collapse_whitespace(instruction.content)
        sample_only = bool(inputs) and all(collapse_whitespace(stdin) in prompt for stdin in inputs)
        coverage = CheckResult(
            check="source_case_coverage",
            status=CheckStatus.UNSUPPORTED if sample_only else CheckStatus.PASS,
            detail=f"{len(inputs)} source cases; "
            + (
                "all inputs occur in public examples, so held-out coverage is not established"
                if sample_only
                else "at least one input is absent from the public examples"
            ),
        )
        return VerificationReport(checks=[*report.checks, coverage])

    return replace(source_recipe, check_suite=replace(suite, run=checks))
