# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve JEEBench's source type-specific prompts and private scoring references."""

from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation import ChatGeneration, EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.scorers.jee import JeeQuestionType
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import source_order
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

PROMPTS = {
    JeeQuestionType.SINGLE_CHOICE: (
        "In this problem, only one option will be correct. Give a detailed solution and end the solution "
        "with the final answer. Mark your solution, which should be exactly one multiple-choice letter, "
        "with \\boxed\nAnswer:"
    ),
    JeeQuestionType.MULTIPLE_CHOICE: (
        "In this problem, multiple options can be correct. Give a detailed solution and end the solution "
        "with the final answer. Mark your solution, which should be one or more multiple-choice letter(s), "
        "with \\boxed\nAnswer:"
    ),
    JeeQuestionType.INTEGER: (
        "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end "
        "the solution with the final answer. Mark your solution with \\boxed\nAnswer:"
    ),
    JeeQuestionType.NUMERIC: (
        "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd "
        "decimal digit. Give a detailed solution and end the solution with the final answer. "
        "Mark your solution with \\boxed\nAnswer:"
    ),
}


def convert_jee(row: RawRow, context: ConversionContext) -> TaskSpec:
    question_type = JeeQuestionType(row.data["type"])
    question = row.data["question"].replace("\n\n", "\n").strip()
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=PROMPTS[question_type] + "\n\nProblem: " + question),),
        package=source_package(
            "jee",
            {"expected": str(row.data["gold"]), "question_type": question_type},
            environment=required_grader_environment(context),
        ),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:JEEBench")})


def jee_item(task: TaskSpec) -> EvaluationItem:
    source = source_evidence(task)
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
        operation=ChatGeneration(),
        groups={
            "source_id": f"{source['description']}/{source['subject']}/{source['index']}",
            "subject": source["subject"],
            "type": source["type"],
        },
    )
