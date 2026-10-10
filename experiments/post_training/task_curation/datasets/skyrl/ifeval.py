# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL instruction-following sources, graded by the vendored SkyRL IFEval scorer.

The scorer rewards the fraction of constraints a reply satisfies. RLVR-IFeval rows already carry
SkyRL constraint descriptors; Nemotron instruction IDs are mapped to the same descriptors.
Conversion normalizes them with the scorer's own preparation function, which rejects unknown
constraints and missing arguments. The sources publish no passing replies, so the controls check
only that an empty reply scores 0.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.skyrl.scorers import ifeval_utils
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import ANSWER_PATH
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference

IFEVAL_VERIFIER = SourceReference(
    "ifeval",
    "c7600581c6ff27b8ebdc5a02954952c77e15b520f8d6da7edb69839dc9948158",
    (
        "https://github.com/marin-community/MarinSkyRL/tree/"
        "e44c4bfcb62c489286a1264094e6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/ifeval"
    ),
)

HERE = Path(__file__).parent
SCORERS = HERE / "scorers"
IFEVAL_GRADE = grade_script(HERE / "ifeval_grade.py", *shipped_files(SCORERS, "ifeval_utils.py"))
GRADER_TIMEOUT = 40.0

RUBRIC = """
Identify every public content request and requirement across the complete conversation; do not discard earlier
requests.

Check the hidden constraint configuration against the public wording and identify missing, additional, or
contradictory constraints.

Formal constraint rewards do not establish factual correctness or useful content; judge the underlying request
separately.

The source scorer rewards the fraction of constraints satisfied; TaskTrove similarly named checks can differ in
counting and punctuation semantics.

Missing requested documents or inputs are defects.
"""

# Nemotron instruction ID -> (SkyRL constraint function, {Nemotron argument: SkyRL argument}).
# keywords:letter_frequency has no entry: Nemotron asks for a relation such as "at least N times",
# and the scorer's letter check counts exactly N.
NEMOTRON_CONSTRAINTS: dict[str, tuple[str, dict[str, str]]] = {
    "keywords:existence": ("verify_keywords", {"keywords": "keyword_list"}),
    "keywords:frequency": (
        "verify_keyword_frequency_relation",
        {"keywords": "keyword_list", "frequency": "N", "relation": "quantifier"},
    ),
    "keywords:forbidden_words": ("validate_forbidden_words", {"forbidden_words": "forbidden_words"}),
    "language:response_language": ("validate_response_language", {"language": "language"}),
    "length_constraints:number_paragraphs": ("verify_paragraph_count", {"num_paragraphs": "N"}),
    "length_constraints:number_words": ("validate_word_constraint", {"num_words": "N", "relation": "quantifier"}),
    "length_constraints:number_sentences": (
        "verify_sentence_constraint",
        {"num_sentences": "N", "relation": "quantifier"},
    ),
    "length_constraints:nth_paragraph_first_word": (
        "validate_paragraphs",
        {"num_paragraphs": "N", "first_word": "first_word", "nth_paragraph": "i"},
    ),
    "detectable_content:postscript": ("verify_postscript", {"postscript_marker": "postscript_marker"}),
    "detectable_content:number_placeholders": ("validate_placeholders", {"num_placeholders": "N"}),
    "detectable_format:number_bullet_lists": ("verify_bullet_points", {"num_bullets": "N"}),
    "detectable_format:title": ("validate_title", {}),
    "detectable_format:constrained_response": ("validate_choice", {"options": "options"}),
    "detectable_format:number_highlighted_sections": ("validate_highlighted_sections", {"num_highlights": "N"}),
    "detectable_format:multiple_sections": (
        "validate_sections",
        {"num_sections": "N", "section_spliter": "section_splitter"},
    ),
    "detectable_format:json_format": ("validate_json_format", {}),
    "combination:repeat_prompt": ("validate_repeat_prompt", {"original_prompt": "original_prompt"}),
    "combination:two_responses": ("validate_two_responses", {}),
    "change_case:capital_word_frequency": (
        "validate_frequency_capital_words",
        {"frequency": "N", "relation": "quantifier"},
    ),
    "change_case:english_capital": ("validate_uppercase", {}),
    "change_case:english_lowercase": ("validate_lowercase", {}),
    "startend:end_checker": ("validate_end", {"end_phrase": "end_phrase"}),
    "punctuation:no_comma": ("validate_no_commas", {}),
    "startend:quotation": ("validate_quotation", {}),
}


def nemotron_constraint(instruction_id: str, arguments: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    """The SkyRL constraint descriptor for one Nemotron instruction; ``prompt`` fills repeat-prompt checks."""
    try:
        function_name, argument_names = NEMOTRON_CONSTRAINTS[instruction_id]
    except KeyError as error:
        raise ValueError(f"Unsupported Nemotron instruction id: {instruction_id!r}") from error
    constraint: dict[str, Any] = {"func_name": function_name}
    for source_name, target_name in argument_names.items():
        value = prompt if source_name == "original_prompt" else arguments.get(source_name)
        if instruction_id == "keywords:frequency" and source_name == "keywords" and value is None:
            # Some rows name one keyword, or only quote it in the instruction's first paragraph.
            keyword = arguments.get("keyword")
            value = [keyword] if isinstance(keyword, str) else re.findall(r'"([^"\n]+)"', prompt.split("\n\n", 1)[0])
        # An empty keyword list makes the check pass on any reply.
        if value is None or value == []:
            raise ValueError(f"Nemotron instruction {instruction_id!r} requires {source_name!r}")
        constraint[target_name] = value
    return constraint


def _messages(value: Any) -> tuple[TextMessage, ...]:
    messages = tuple(TextMessage.model_validate(message) for message in value)
    ConversationInput(events=messages)
    return messages


def ifeval_task(
    row: RawRow,
    context: ConversionContext,
    events: Sequence[TextMessage],
    constraints: list[Any],
    evidence: Mapping[str, Any],
) -> TaskSpec | ImportRejection:
    try:
        normalized = json.loads(ifeval_utils.normalize_ground_truth(constraints))
    except (ValueError, TypeError) as error:
        return unsupported("unsupported_ifeval_constraint", str(error))
    package = script_package(
        IFEVAL_GRADE,
        {"constraints": normalized},
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=ANSWER_PATH,
    )
    return conversation_task(row, events=events, package=package, evidence=evidence)


def convert_nemotron_if(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        events = _messages(row.data["input"])
        names, arguments = row.data["args"]["instruction_id_list"], row.data["args"]["instruction_kwargs"]
        if not names or len(names) != len(arguments):
            raise ValueError("Instruction identifiers and arguments must be nonempty and aligned")
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_instruction_contract", str(error))
    prompt = next(event.content for event in reversed(events) if event.role == "user")
    try:
        constraints = [
            nemotron_constraint(name, argument, prompt) for name, argument in zip(names, arguments, strict=True)
        ]
    except ValueError as error:
        return unsupported("unsupported_ifeval_constraint", str(error))
    evidence = {key: value for key, value in row.data.items() if key not in {"input", "args", "path"}}
    return ifeval_task(row, context, events, constraints, evidence)


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
                id="MarinSkyRL:nemotron_if",
                title="nvidia/Llama-Nemotron-Post-Training-Dataset · RL/instruction_following",
                origin="MarinSkyRL",
                family="instruction-following",
                tags=("rlvr", "single-turn", "license:cc-by-4.0", "gym/ifeval"),
                verifier=IFEVAL_VERIFIER,
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
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
                ships=(SCORERS,),
            ),
        ),
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="MarinSkyRL:rlvr_ifeval",
                title="allenai/RLVR-IFeval",
                origin="MarinSkyRL",
                family="instruction-following",
                tags=("rlvr", "single-turn", "license:odc-by", "gym/ifeval"),
                verifier=IFEVAL_VERIFIER,
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
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=CONTROLS,
                grader=GRADER_PACKAGES,
                ships=(SCORERS,),
            ),
        ),
    ]
