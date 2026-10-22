# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert generated closure cases to ordinary text-prefill TaskSpecs."""

from dataclasses import replace

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
from transformers import PreTrainedTokenizerFast
from verifyit.modes.grade_structure_closure import validate_structure_closure
from verifyit.spec import StructureClosureSpec

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CLOSE_REQUIRED,
    TASK_ID_PREFIX,
    ClosureCase,
    Family,
    closure_format,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tool_serialization import (
    paired_tool_variants,
    write_result_tool,
)


def case_task(case: ClosureCase, source: Source, *, tools: tuple[FunctionDefinition, ...] = ()) -> TaskSpec:
    """Keep task content in messages and the completion contract in the private grader."""
    spec = StructureClosureSpec(
        prefix=case.prefix,
        expected=case.expected,
        format=closure_format(case.family),
    )
    validate_structure_closure(spec)
    return TaskSpec(
        id=TASK_ID_PREFIX + case.case_id,
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


def paired_task_probes(case: ClosureCase, source: Source, tokenizer: PreTrainedTokenizerFast) -> list[TaskSpec]:
    """Build the four declaration/serialization arms without requiring a live tool."""
    if case.family != Family.TOOL or case.control != CLOSE_REQUIRED:
        raise ValueError("Paired probes require a close-required tool case")
    definition = FunctionDefinition.model_validate(write_result_tool()["function"])
    tasks = []
    for variant in paired_tool_variants(case.case_id, case.prefix, case.instruction, tokenizer):
        tools = (definition,) if variant.declared else ()
        updated = replace(case, case_id=variant.case_id, prefix=variant.prefix, expected=variant.expected)
        task = case_task(updated, source, tools=tools)
        tasks.append(task.model_copy(update={"tags": (*task.tags, f"probe:{variant.name}")}))
    return tasks
