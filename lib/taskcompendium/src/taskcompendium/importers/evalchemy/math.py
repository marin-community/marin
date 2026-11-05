# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy math prompts and source conversion."""

import json

from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.judge_prompts import EQUIVALENCE_JUDGE_TEMPLATE
from taskcompendium.importers.evalchemy.scorers.olympiad import reference_answers
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, RawRow

MATH_PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"


def boxed_math_task(
    row: RawRow,
    context: ConversionContext,
    *,
    answer_field: str,
    benchmark: str,
) -> TaskSpec | ImportRejection:
    """Render an exact source prompt and retain the boxed-math grading protocol."""
    answer = row.data.get(answer_field)
    if answer is None or not str(answer).strip():
        return source_defect("missing_reference", f"The source requires {answer_field}")
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=MATH_PROMPT.format_map(row.data)),),
        package=source_package(
            "boxed_math", {"expected": str(answer)}, environment=required_grader_environment(context)
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{benchmark}")})


def convert_amc(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=MATH_PROMPT.format(problem=row.data["question"])),),
        package=source_package(
            "amc", {"expected": str(row.data["answer"])}, environment=required_grader_environment(context)
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:AMC23")})


def convert_olympiad(row: RawRow, context: ConversionContext, *, judged: bool) -> TaskSpec:
    answers = reference_answers(row.data["answer"])
    judge = (
        JudgeSpec(
            references=(json.dumps(list(answers), ensure_ascii=False),),
            question=row.data["problem"],
            rubric="labels",
            exact_gate=False,
            prompt_template=EQUIVALENCE_JUDGE_TEMPLATE.replace("{reference_answers}", "{reference}").replace(
                "{candidate_answer}", "{candidate}"
            ),
            label_scores={"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0},
            label_metrics={
                label: {
                    "accuracy": float(label == "correct"),
                    "incorrect": float(label == "incorrect"),
                    "not_attempted": float(label == "not_attempted"),
                    "judge_failed": 0.0,
                }
                for label in ("correct", "incorrect", "not_attempted")
            },
            label_scan="whole",
            label_case="upper",
            max_completion_tokens=2048,
        )
        if judged
        else None
    )
    environment = required_grader_environment(context)
    if judged:
        environment = environment.model_copy(update={"capabilities": (*environment.capabilities, "network")})
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=MATH_PROMPT.format(problem=row.data["problem"])),),
        package=source_package(
            "olympiad",
            {"expected": answers, "judged": judged},
            environment=environment,
            judge=judge,
            timeout=60 + judge.transport_budget if judge is not None else 60,
        ),
        evidence=dict(row.data),
    )
    benchmark = "OlympiadBench" if judged else "OlympiadBenchDeterministic"
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{benchmark}")})
