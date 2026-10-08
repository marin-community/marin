# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL instruction-following sources, graded by the pinned SkyRL IFEval scorer in the IFEval image.

The scorer rewards the fraction of constraints a reply satisfies. RLVR-IFeval rows already carry
SkyRL constraint descriptors; Nemotron instruction IDs are mapped to the same descriptors. The
sources publish no passing replies, so no control can check these graders offline.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError
from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.source_scorer import source_scorer_package
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.images import IFEVAL_IMAGE
from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, ShellSim

IFEVAL_SCORER_PATH = "/opt/skyrl_gym/skyrl_gym/envs/ifeval/utils.py"
IFEVAL_INVOCATION = {
    "function": "pinned_skyrl_ifeval:compute_score",
    "source_path": IFEVAL_SCORER_PATH,
    # SkyRL revision 544d5d6f14116a06bde0209352585903133bd618.
    "source_sha256": "3194b7a44ada0a4cd185ab3c85d883f85da8641970fef3dfc4b66d13780079fc",
    "args": ["answer", "contract.constraints"],
    "reward_key": "score",
}
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
NEMOTRON_CONSTRAINTS: dict[str, tuple[str, dict[str, str]]] = {
    "keywords:existence": ("verify_keywords", {"keywords": "keyword_list"}),
    "keywords:frequency": (
        "verify_keyword_frequency_relation",
        {"keywords": "keyword_list", "frequency": "N", "relation": "quantifier"},
    ),
    "keywords:forbidden_words": ("validate_forbidden_words", {"forbidden_words": "forbidden_words"}),
    "keywords:letter_frequency": ("verify_letter_frequency", {"letter": "letter", "let_frequency": "N"}),
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
    "detectable_format:quotation": ("validate_quotation", {}),
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
        if value is None:
            raise ValueError(f"Nemotron instruction {instruction_id!r} requires {source_name!r}")
        constraint[target_name] = value
    return constraint


def _messages(value: Any) -> tuple[TextMessage, ...]:
    messages = tuple(TextMessage.model_validate(message) for message in value)
    ConversationInput(events=messages)
    return messages


def ifeval_scorer_task(
    row: RawRow, events: Sequence[TextMessage], constraints: Any, evidence: Mapping[str, Any]
) -> TaskSpec:
    package = source_scorer_package(
        invocation=IFEVAL_INVOCATION,
        config={"contract": {"constraints": constraints}},
        environment=IFEVAL_IMAGE.requirements(),
        timeout=GRADER_TIMEOUT,
    )
    return conversation_task(row, events=events, package=package, evidence=evidence)


def convert_nemotron_if(row: RawRow) -> TaskSpec | ImportRejection:
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
    return ifeval_scorer_task(row, events, constraints, evidence)


def convert_rlvr_ifeval(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        events = _messages(row.data["messages"])
        constraints = json.loads(row.data["ground_truth"])
        if not isinstance(constraints, dict) or not isinstance(constraints.get("func_name"), str):
            raise ValueError("The source ground truth must name its constraint function")
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_instruction_contract", str(error))
    evidence = {key: value for key, value in row.data.items() if key not in {"messages", "ground_truth", "path"}}
    return ifeval_scorer_task(row, events, constraints, evidence)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="nemotron_if",
            source=HfSource(
                "nvidia/Llama-Nemotron-Post-Training-Dataset",
                "ab2a40d258a6a4d9d4c277d702aeea445081766c",
                ("RL/instruction_following/instruction_following.jsonl",),
                SourceFormat.JSONL,
            ),
            convert=convert_nemotron_if,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=RUBRIC,
            atlas_id="MarinSkyRL:nemotron_if",
        ),
        RlDataPipeline(
            name="rlvr_ifeval",
            source=HfSource(
                "allenai/RLVR-IFeval",
                "47c03c73621c4aab2b824b7818681117d662770e",
                ("data/train-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            convert=convert_rlvr_ifeval,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=RUBRIC,
            atlas_id="MarinSkyRL:rlvr_ifeval",
        ),
    ]
