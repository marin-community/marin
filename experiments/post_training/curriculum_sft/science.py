# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate GPQA-style four-option science questions with GLM and keep those GLM answers blind.

GLM writes a question, its correct option, three distractors that each follow from a common
reasoning error, and a worked rationale. The step shuffles the options with a generator seeded by
the step seed and request index, so the keyed letter is uniform over A-D instead of following GLM's
letter preferences. It rejects questions with duplicate options or whose text contains a long
correct option. GLM then answers each remaining question ``verify_samples`` times without the key;
a question is kept when a strict majority of samples box the keyed letter. Agreement filters
ambiguous and mis-keyed questions, but a mistake the author and solver share still passes.

The problems Parquet matches ``generation.PROBLEMS_FILENAME``: ``problem`` asks for the final letter
in ``\\boxed{}`` and ``answer`` is that letter, so ``self_distill`` grades it with ``AnswerCheck.CHOICE``.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT
from marin.inference.structured_output import StructuredTool
from pydantic import Field, ValidationError
from rigging.filesystem.storage_path import StoragePath
from tasktrove_verify.modes.extract import extract_boxed

from experiments.post_training.curriculum_sft.generation import (
    MANIFEST_FILENAME,
    PROBLEMS_FILENAME,
    RAW_RESPONSES_FILENAME,
    _failure_reason,
    _glm_client,
    _responses_by_id,
    _run_batch,
    capability_packet,
    problem_targets,
    write_table,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact
from experiments.post_training.task_curriculum.models import StrictModel

logger = logging.getLogger(__name__)

# Probe failures on GPQA Diamond cluster in NMR assignment, polar organic mechanisms, and special relativity.
NMR_CAPABILITY = "d04.measurement.spectroscopy.nmr"
ORGANIC_MECHANISMS_CAPABILITY = "d04.structure.organic_mechanisms.polar"
SPECIAL_RELATIVITY_CAPABILITY = "d03.relativity_gravity.special_relativity"

LETTERS = "ABCD"
VERIFY_RESPONSES_FILENAME = "verify-responses.jsonl"
# Short options such as "0.6c" or "2" can appear in a question as given data; only longer ones count as a leak.
LEAK_MIN_CHARACTERS = 12

# Catalog capabilities have one or two generation targets, so varying the question type spreads problems further.
QUESTION_TYPES = (
    "Ask for a computed quantity; the options are values with units.",
    "Ask which structure, product, or configuration is consistent with the given data or conditions.",
    "Ask which of four statements about the scenario is correct.",
)

SOLVER_SYSTEM_PROMPT = (
    "Answer the multiple-choice science question. After reasoning, write a concise solution that states "
    "the key steps and ends with the letter of the correct option in \\boxed{}."
)


class GeneratedQuestion(StrictModel):
    question: str = Field(min_length=1)
    correct_option: str = Field(min_length=1)
    distractors: list[str] = Field(min_length=3, max_length=3)
    rationale: str = Field(min_length=1)


QUESTION_TOOL = StructuredTool(
    name="submit_question",
    description="Submit one multiple-choice science question, its correct option, three distractors, and a rationale.",
    output_type=GeneratedQuestion,
)

SCIENCE_PROBLEM_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("capability_id", pa.string(), nullable=False),
        pa.field("facet_id", pa.string(), nullable=False),
        pa.field("question_type", pa.string(), nullable=False),
        pa.field("problem", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("rationale", pa.string()),
        pa.field("agreeing_samples", pa.int32()),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)


@dataclass(frozen=True)
class GenerateScienceConfig:
    catalog_path: str
    output_path: str
    capability_id: str
    requested: int
    verify_samples: int
    seed: int
    max_completion_tokens: int
    relay_job: str


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def shuffle_options(correct: str, distractors: list[str], seed: int, index: int) -> tuple[list[str], str]:
    """Shuffle the correct option among the distractors and return the options and the correct letter.

    The order depends only on ``seed`` and ``index``, so reruns reproduce the same key.
    """
    options = [correct, *distractors]
    random.Random(f"{seed}:{index}").shuffle(options)
    return options, LETTERS[options.index(correct)]


def mcq_problem(question: str, options: list[str]) -> str:
    choices = "\n".join(f"{letter}) {option.strip()}" for letter, option in zip(LETTERS, options, strict=True))
    return (
        f"{question.strip()}\n\n{choices}\n\n"
        "Choose the correct option and give its letter in \\boxed{}, for example \\boxed{A}."
    )


def question_prompt(packet: dict[str, Any], target: dict[str, str], question_type: str) -> str:
    capability = {key: packet[key] for key in ("subject", "capability_id", "name", "outcome", "includes", "excludes")}
    return (
        "Write one original graduate-level multiple-choice science question at the difficulty of GPQA Diamond.\n"
        "- `question`: a self-contained problem that needs several steps of expert reasoning; a PhD student in "
        "the field should answer it, a skilled non-expert with web search should not. Give every quantity and "
        "observation needed. Do not list options or letters, and do not name the method or the answer.\n"
        "- `correct_option`: the single correct answer, stated briefly.\n"
        "- `distractors`: exactly three wrong options. Build each from a specific common reasoning error, such "
        "as a sign or frame error, a missed factor, or a misassigned signal, and match the correct option's "
        "form and length. Each must be unambiguously wrong.\n"
        "- `rationale`: a worked solution that derives the correct option and names the error behind each "
        "distractor.\n"
        "- Do not copy published exam or benchmark questions. Solve the question yourself before submitting.\n"
        f"Capability: {json.dumps(capability, ensure_ascii=False, sort_keys=True)}\n"
        f"Focus: {target['description']}\n"
        f"Question type: {question_type}"
    )


def _assignment(packet: dict[str, Any], index: int) -> tuple[dict[str, str], str]:
    """Cycle requests through every target, then every question type."""
    targets = problem_targets(packet)
    return targets[index % len(targets)], QUESTION_TYPES[(index // len(targets)) % len(QUESTION_TYPES)]


def question_request(config: GenerateScienceConfig, packet: dict[str, Any], index: int) -> dict[str, Any]:
    target, question_type = _assignment(packet, index)
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Write one original science question and check its key before submitting."},
            {"role": "user", "content": question_prompt(packet, target, question_type)},
        ],
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "temperature": 1.0,
        "seed": config.seed + index,
        "max_tokens": config.max_completion_tokens,
    }
    body.update(QUESTION_TOOL.request_fields())
    return {"custom_id": f"question-{index:05d}", "method": "POST", "url": CHAT_COMPLETIONS_ENDPOINT, "body": body}


def _question_fields(question: GeneratedQuestion, seed: int, index: int, seen: set[str]) -> tuple[dict, str | None]:
    options, letter = shuffle_options(question.correct_option, question.distractors, seed, index)
    fields = {"problem": mcq_problem(question.question, options), "answer": letter, "rationale": question.rationale}
    if len({_normalized(option) for option in options}) < len(options):
        return fields, "duplicate_options"
    correct = _normalized(question.correct_option)
    if len(correct) >= LEAK_MIN_CHARACTERS and correct in _normalized(question.question):
        return fields, "answer_in_question"
    if _normalized(question.question) in seen:
        return fields, "duplicate_question"
    seen.add(_normalized(question.question))
    return fields, None


def parse_question_batch(raw_output: str, config: GenerateScienceConfig, packet: dict[str, Any]) -> list[dict]:
    """Shuffle each well-formed question into a boxed-letter problem; keep rejection accounting for the rest."""
    responses = _responses_by_id(raw_output, {f"question-{index:05d}" for index in range(config.requested)})
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(config.requested):
        request_id = f"question-{index:05d}"
        target, question_type = _assignment(packet, index)
        fields: dict[str, str] = {}
        reason = _failure_reason(responses[request_id])
        if reason is None:
            try:
                question = QUESTION_TOOL.parse(responses[request_id]["response"]["body"])
            except (UnicodeError, ValidationError, ValueError):
                reason = "invalid_question"
            else:
                fields, reason = _question_fields(question, config.seed, index, seen)
        records.append(
            {
                "request_id": request_id,
                "capability_id": config.capability_id,
                "facet_id": target["id"],
                "question_type": question_type,
                "problem": fields.get("problem"),
                "answer": fields.get("answer"),
                "rationale": fields.get("rationale"),
                "agreeing_samples": None,
                "accepted": reason is None,
                "rejection_reason": reason,
            }
        )
    return records


def verify_requests(config: GenerateScienceConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blind GLM answer requests, ``verify_samples`` per well-formed question."""
    requests = []
    for record in records:
        if not record["accepted"]:
            continue
        for sample in range(config.verify_samples):
            requests.append(
                {
                    "custom_id": f"{record['request_id']}-v{sample:02d}",
                    "method": "POST",
                    "url": CHAT_COMPLETIONS_ENDPOINT,
                    "body": {
                        "model": GLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                            {"role": "user", "content": record["problem"]},
                        ],
                        "chat_template_kwargs": {"reasoning_effort": "high"},
                        "temperature": 1.0,
                        "seed": config.seed + len(requests),
                        "max_tokens": config.max_completion_tokens,
                    },
                }
            )
    return requests


def _boxed_letter(response: dict[str, Any]) -> str | None:
    if _failure_reason(response) is not None:
        return None
    boxed = extract_boxed(response["response"]["body"]["choices"][0]["message"].get("content") or "")
    return boxed.strip().strip("()").upper() if boxed is not None else None


def apply_verification(raw_output: str, config: GenerateScienceConfig, records: list[dict]) -> list[dict]:
    """Keep well-formed questions whose keyed letter a strict majority of blind samples box."""
    responses = _responses_by_id(raw_output, {request["custom_id"] for request in verify_requests(config, records)})
    verified = []
    for record in records:
        if not record["accepted"]:
            verified.append(record)
            continue
        letters = [
            _boxed_letter(responses[f"{record['request_id']}-v{sample:02d}"]) for sample in range(config.verify_samples)
        ]
        agreeing = sum(letter == record["answer"] for letter in letters)
        majority = 2 * agreeing > config.verify_samples
        verified.append(
            {
                **record,
                "agreeing_samples": agreeing,
                "accepted": majority,
                "rejection_reason": None if majority else "solver_disagrees",
            }
        )
    return verified


def generate_questions(config: GenerateScienceConfig) -> Artifact:
    """Write verified problem Parquet, exact GLM question and verification responses, and a manifest."""
    catalog = TaskCurriculumCatalogArtifact(path=config.catalog_path).read_catalog()
    packet = capability_packet(catalog, config.capability_id)
    client = _glm_client(config.relay_job)
    requests = [question_request(config, packet, index) for index in range(config.requested)]
    raw_output = _run_batch(client, requests, f"curriculum-science-{config.capability_id}.jsonl")
    records = parse_question_batch(raw_output, config, packet)
    if not any(record["accepted"] for record in records):
        raise ValueError(f"GLM wrote no well-formed questions for {config.capability_id}")
    verify_output = _run_batch(
        client, verify_requests(config, records), f"curriculum-science-verify-{config.capability_id}.jsonl"
    )
    records = apply_verification(verify_output, config, records)

    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, PROBLEMS_FILENAME, records, SCIENCE_PROBLEM_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(raw_output)
    (output / VERIFY_RESPONSES_FILENAME).write_text(verify_output)
    reasons: dict[str, int] = {}
    for record in records:
        key = record["rejection_reason"] or "accepted"
        reasons[key] = reasons.get(key, 0) + 1
    letters = {letter: sum(r["accepted"] and r["answer"] == letter for r in records) for letter in LETTERS}
    manifest = {
        "catalog_version": catalog.catalog_version,
        "capability_id": config.capability_id,
        "generator": GLM_MODEL,
        "requested": len(records),
        "outcomes": reasons,
        "accepted_answer_letters": letters,
        "verified": f"a strict majority of {config.verify_samples} blind GLM answers box the keyed letter",
        "problems": PROBLEMS_FILENAME,
        "raw_responses": RAW_RESPONSES_FILENAME,
        "verify_responses": VERIFY_RESPONSES_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("science questions for %s: %s", config.capability_id, reasons)
    return Artifact(path=config.output_path)


def generate_science_questions(
    capability_id: str,
    *,
    version: str,
    requested: int,
    verify_samples: int,
    seed: int,
    max_completion_tokens: int,
    catalog: ArtifactStep[TaskCurriculumCatalogArtifact] = TASK_CURRICULUM,
) -> ArtifactStep[Artifact]:
    """Build one GLM question generation and verification step; run it on `cw-us-east-08a`, near the relay."""

    def build_config(ctx: StepContext) -> GenerateScienceConfig:
        return GenerateScienceConfig(
            catalog_path=ctx.artifact_path(catalog),
            output_path=ctx.output_path,
            capability_id=capability_id,
            requested=requested,
            verify_samples=verify_samples,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/science-questions"),
        version=version,
        artifact_type=Artifact,
        run=generate_questions,
        build_config=build_config,
        deps=(catalog,),
    )
