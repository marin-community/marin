# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL instruction-following sources, graded by the vendored SkyRL IFEval scorer.

The scorer rewards the fraction of constraints a reply satisfies. RLVR-IFeval rows already carry
SkyRL constraint descriptors; Nemotron instruction IDs are mapped to the same descriptors.
Conversion normalizes them with the scorer's own preparation function, which rejects unknown
constraints and missing arguments. The sources publish no passing replies, so the controls check
only that a reply violating every constraint scores 0.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from taskcompendium.convert.answers import unsupported
from taskcompendium.grader import grader_config
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, Reply

from experiments.post_training.task_curation.datasets.skyrl.scorer_tasks import SCORERS, scorer_task
from experiments.post_training.task_curation.datasets.skyrl.scorers import ifeval_utils
from experiments.post_training.task_curation.images.recipes import GRADER
from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, ShellSim

IFEVAL_GRADE = Path(__file__).with_name("ifeval_grade.py")
GRADER_TIMEOUT = 40.0
FILLER = "vx"
"""A word no constraint names; the violating reply pads counts with it."""

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
    return scorer_task(
        row,
        events=events,
        script=IFEVAL_GRADE,
        scorer="ifeval_utils.py",
        config={"constraints": normalized},
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        env={},
        evidence=evidence,
    )


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


def _overshoot(quantifier: str, bound: int, tolerance: int) -> int:
    """How many items take a count past an upper ``bound``; zero for a lower bound, which the short reply misses."""
    return {"less than": bound, "at most": bound + 1, "around": bound + tolerance + 1}.get(quantifier, 0)


def violating_text(constraints: Sequence[Mapping[str, Any]]) -> str:
    """A reply that fails every constraint.

    The reply starts as one short mixed-case word without punctuation, markup or keywords, which
    fails the case and format checks and any lower bound above one. Each other constraint adds what
    it forbids, or more words, sentences or capital words than its upper bound allows.
    """
    head, words = "Qz", []
    for constraint in constraints:
        name, bound, quantifier = constraint["func_name"], constraint.get("N"), constraint.get("quantifier")
        match name:
            case "validate_no_commas":
                head += ","
            case "validate_forbidden_words":
                words.append(constraint["forbidden_words"][0])
            case "verify_keyword_frequency" if bound == 0:
                words.append(constraint["word"])
            case "verify_keyword_frequency_relation":
                words += [constraint["keyword_list"][0]] * _overshoot(quantifier, bound, 0)
            case "validate_word_constraint":
                words += [FILLER] * _overshoot(quantifier, bound, max(round(bound * 0.1), 1))
            case "verify_sentence_constraint":
                words += [f"{FILLER}."] * _overshoot(quantifier, bound, 1)
            case "validate_frequency_capital_words":
                words += [FILLER.upper()] * _overshoot(quantifier, bound, max(round(bound * 0.1), 1))
            case "verify_bullet_points" if bound == 0:
                words.append(f"\n- {FILLER}")
            case "verify_paragraph_count" if bound == 1:
                words.append(f"\n* * *\n{FILLER}")
            case "validate_paragraphs" if bound == 1:
                words.append(f"\n\n{FILLER}")
            case "validate_sections" if bound == 1:
                words += [constraint["section_splitter"], FILLER] * 2
    text = " ".join((head, *words))
    for constraint in constraints:
        if constraint["func_name"] == "verify_letter_frequency" and text.count(constraint["letter"]) == constraint["N"]:
            text = constraint["letter"] + text
    return text


def violating_reply(task: TaskSpec) -> Reply:
    return answer_reply(task, violating_text(grader_config(task)["constraints"]))


CONTROLS = Controls(negative=violating_reply)


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
            controls=CONTROLS,
            grader_image=GRADER,
            ships=(SCORERS,),
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
            controls=CONTROLS,
            grader_image=GRADER,
            ships=(SCORERS,),
            atlas_id="MarinSkyRL:rlvr_ifeval",
        ),
    ]
