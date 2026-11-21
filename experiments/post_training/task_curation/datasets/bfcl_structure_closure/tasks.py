# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert generated closure cases to ordinary text-prefill TaskSpecs."""

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.modes.grade_structure_closure import validate_structure_closure
from verifyit.spec import StructureClosureSpec

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    ClosureCase,
    closure_format,
)


def case_task(case: ClosureCase, source: Source, task_id: str, *, tools: tuple[FunctionDefinition, ...]) -> TaskSpec:
    """Keep task content in messages and the completion contract in the private grader."""
    spec = StructureClosureSpec(
        prefix=case.prefix,
        expected=case.expected,
        format=closure_format(case.family),
    )
    validate_structure_closure(spec)
    return TaskSpec(
        id=task_id,
        source=source,
        context=ConversationInput(
            events=(
                TextMessage(role="user", content=case.instruction),
                TextMessage(role="assistant", content=case.prefix),
            )
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        final_tools=tools,
        grader=verifyit_package(spec).grader,
        tags=(f"family:{case.family}", f"control:{case.control}", f"split:{case.split}"),
    )
