# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL instruction-following sources graded with verifyit IFEval checks."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finestore.rl.status_taxonomy import DataRestriction
from pydantic import ValidationError
from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import ConversationInput, TaskResource, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow
from verifyit.adapters.open_instruct_ifeval import constraints_from_descriptors, normalize_ground_truth
from verifyit.grade import InvalidTask
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.spec import DEFAULT_OUTPUT, Constraint

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
IFEVAL_GRADE = grade_script(HERE / "nemotron_if_grade.py")
NEMOTRON_GRADE = IFEVAL_GRADE
GRADER_TIMEOUT = 40.0

RUBRIC_REQUESTS = """
Identify every public content request and requirement across the complete conversation; do not discard earlier
requests.

Check the hidden constraint configuration against the public wording and identify missing, additional, or
contradictory constraints.

Formal constraint rewards do not establish factual correctness or useful content; judge the underlying request
separately.
"""
RUBRIC_DEFECTS = """
Missing requested documents or inputs are defects.
"""
RUBRIC = (
    RUBRIC_REQUESTS
    + """
The scorer rewards the fraction of constraints satisfied with verifyit's IFEval checks. Some scores differ from
the source checker, especially for language, punctuation, and Markdown formatting.
"""
    + RUBRIC_DEFECTS
)
NEMOTRON_RUBRIC = (
    RUBRIC_REQUESTS
    + """
The scorer rewards the fraction of constraints satisfied, with verifyit's IFEval checks.
"""
    + RUBRIC_DEFECTS
)

# The IFEval arguments each supported Nemotron instruction needs; a row without one cannot be satisfied.
# language:response_language has no entry: verifyit's language check is an approximation that one hint word passes.
NEMOTRON_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "change_case:capital_word_frequency": ("capital_frequency", "capital_relation"),
    "change_case:english_capital": (),
    "change_case:english_lowercase": (),
    "combination:two_responses": (),
    "detectable_content:number_placeholders": ("num_placeholders",),
    "detectable_content:postscript": ("postscript_marker",),
    "detectable_format:constrained_response": (),
    "detectable_format:json_format": (),
    "detectable_format:multiple_sections": ("num_sections", "section_spliter"),
    "detectable_format:number_bullet_lists": ("num_bullets",),
    "detectable_format:number_highlighted_sections": ("num_highlights",),
    "detectable_format:title": (),
    "keywords:existence": ("keywords",),
    "keywords:forbidden_words": ("forbidden_words",),
    "keywords:frequency": ("keyword", "frequency", "relation"),
    "keywords:letter_frequency": ("letter", "let_frequency", "let_relation"),
    "length_constraints:nth_paragraph_first_word": ("num_paragraphs", "nth_paragraph", "first_word"),
    "length_constraints:number_paragraphs": ("num_paragraphs",),
    "length_constraints:number_sentences": ("num_sentences", "relation"),
    "length_constraints:number_words": ("num_words", "relation"),
    "punctuation:no_comma": (),
    "startend:end_checker": ("end_phrase",),
    "startend:quotation": (),
}
MISSING_ARGUMENT_VALUES = (None, "", [])
"""An empty keyword list would make the check pass on any reply."""


def nemotron_constraint(instruction_id: str, arguments: Mapping[str, Any]) -> Constraint:
    """The verifyit constraint for one Nemotron instruction, without the arguments the row leaves null."""
    try:
        required = NEMOTRON_ARGUMENTS[instruction_id]
    except KeyError as error:
        raise ValueError(f"Unsupported Nemotron instruction id: {instruction_id!r}") from error
    params = {name: value for name, value in arguments.items() if value is not None}
    missing = [name for name in required if params.get(name) in MISSING_ARGUMENT_VALUES]
    if missing:
        raise ValueError(f"Nemotron instruction {instruction_id!r} requires {missing}")
    return Constraint(instruction_id, params)


def _messages(value: Any) -> tuple[TextMessage, ...]:
    messages = tuple(TextMessage.model_validate(message) for message in value)
    ConversationInput(events=messages)
    return messages


def scripted_task(
    row: RawRow,
    context: ConversionContext,
    events: Sequence[TextMessage],
    grade: tuple[TaskResource, ...],
    constraints: list[Any],
    evidence: Mapping[str, Any],
) -> TaskSpec:
    package = script_package(
        grade,
        {"constraints": constraints},
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
    )
    return conversation_task(row, events=events, package=package, evidence=evidence)


def ifeval_task(
    row: RawRow,
    context: ConversionContext,
    events: Sequence[TextMessage],
    constraints: list[Any],
    evidence: Mapping[str, Any],
) -> TaskSpec | ImportRejection:
    try:
        normalized = json.loads(normalize_ground_truth(constraints))
        converted = constraints_from_descriptors(normalized)
    except (InvalidTask, ValueError, TypeError) as error:
        return unsupported("unsupported_ifeval_constraint", str(error))
    if any(constraint.name == "language:response_language" for constraint in converted):
        return unsupported("unsupported_ifeval_constraint", "Response language requires a detector")
    return scripted_task(row, context, events, IFEVAL_GRADE, [asdict(constraint) for constraint in converted], evidence)


def convert_nemotron_if(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        events = _messages(row.data["input"])
        names, arguments = row.data["args"]["instruction_id_list"], row.data["args"]["instruction_kwargs"]
        if not names or len(names) != len(arguments) or not all(isinstance(argument, dict) for argument in arguments):
            raise ValueError("Instruction identifiers and argument objects must be nonempty and aligned")
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_instruction_contract", str(error))
    try:
        constraints = tuple(nemotron_constraint(name, argument) for name, argument in zip(names, arguments, strict=True))
        resolve_checks(constraints)
    except (ValueError, InvalidTask) as error:
        return unsupported("unsupported_ifeval_constraint", str(error))
    evidence = {key: value for key, value in row.data.items() if key not in {"input", "args", "path"}}
    config = [asdict(constraint) for constraint in constraints]
    return scripted_task(row, context, events, NEMOTRON_GRADE, config, evidence)


def convert_rlvr_ifeval(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        events = _messages(row.data["messages"])
        constraint = json.loads(row.data["ground_truth"])
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_instruction_contract", str(error))
    evidence = {key: value for key, value in row.data.items() if key not in {"messages", "ground_truth", "path"}}
    return ifeval_task(row, context, events, [constraint], evidence)


CONTROLS = Controls()


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="Nemotron:nemotron_if",
                task_format=TaskFormat.TASKSPEC,
                upstream_dataset="nemotron_if",
                upstream_name="nemotron_if",
                subset=None,
                build_path="MarinSkyRL",
                origin=SourceOrigin.NEMOTRON,
                family="instruction-following",
                tags=("rlvr", "single-turn", "license:cc-by-4.0", "gym/ifeval"),
            ),
            config=CurationRecipe(
                name="nemotron_if",
                source=HfSource(
                    "nvidia/Llama-Nemotron-Post-Training-Dataset",
                    "ab2a40d258a6a4d9d4c277d702aeea445081766c",
                    ("RL/instruction_following/instruction_following.jsonl",),
                    SourceFormat.JSONL,
                ),
                convert=convert_nemotron_if,
                # L1 keeps the sealed version string so the receipt bindings hold;
                # the admission landing bumps it to "2" with the re-cut receipts.
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=NEMOTRON_RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="AllenAI:rlvr_ifeval",
                task_format=TaskFormat.TASKSPEC,
                upstream_dataset="rlvr_ifeval",
                upstream_name="rlvr_ifeval",
                subset=None,
                build_path="MarinSkyRL",
                origin=SourceOrigin.ALLENAI,
                family="instruction-following",
                tags=("rlvr", "single-turn", "license:odc-by", "gym/ifeval"),
            ),
            config=CurationRecipe(
                name="rlvr_ifeval",
                source=HfSource(
                    "allenai/RLVR-IFeval",
                    "47c03c73621c4aab2b824b7818681117d662770e",
                    ("data/train-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                convert=convert_rlvr_ifeval,
                # L1 keeps the sealed version string so the receipt bindings hold;
                # the admission landing bumps it to "3" with the re-cut receipts.
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
    ]
