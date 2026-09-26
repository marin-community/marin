# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate curriculum math problems with GLM, then keep blind GLM solutions that reach the reference answer.

The problem step asks GLM for one problem and its short final answer per request, with an explicit
content target and difficulty target. The solve step samples several independent GLM solutions per
problem without showing the reference answer and keeps solutions whose boxed answer is
math-verify-equivalent to it. Kept rows carry GLM's reasoning as ``reasoning_content`` and set
``enable_thinking``, so the Marin chat template renders the ``Reasoning: /think`` header and think
block that Snowball sees at inference.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from math_verify import parse, verify
from pydantic import Field, ValidationError
from rigging.filesystem.storage_path import StoragePath
from tasktrove_verify.modes.extract import extract_boxed, strip_math_delimiters
from zephyr.writers import write_parquet_file

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact
from experiments.post_training.task_curriculum.models import CapabilitySection, CurriculumCatalog, StrictModel

logger = logging.getLogger(__name__)

PROBLEMS_FILENAME = "problems/part-00000-of-00001.parquet"
SOLUTIONS_FILENAME = "solutions/part-00000-of-00001.parquet"
CHAT_FILENAME = "chat/part-00000-of-00001.parquet"
RAW_RESPONSES_FILENAME = "raw-responses.jsonl"
MANIFEST_FILENAME = "manifest.json"
POLL_SECONDS = 5.0
MATH_VERIFY_TIMEOUT = 5

DIFFICULTY_TARGETS = (
    "MATH level 3: a multi-step competition exercise that needs a correct plan, not one formula.",
    "MATH level 4: combines two ideas or needs a non-obvious substitution or case split.",
    "MATH level 5: a hard competition problem with several stages or a tempting wrong path.",
    "AMC 12 or early AIME: a concise statement whose solution needs insight and careful casework.",
)

SOLVER_SYSTEM_PROMPT = (
    "Solve the math problem. After reasoning, write a concise solution that states the key steps "
    "and ends with the final answer in \\boxed{}."
)

PROBLEM_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("capability_id", pa.string(), nullable=False),
        pa.field("facet_id", pa.string(), nullable=False),
        pa.field("difficulty", pa.string(), nullable=False),
        pa.field("problem", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)

SOLUTION_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("problem_request_id", pa.string(), nullable=False),
        pa.field("sample", pa.int32(), nullable=False),
        pa.field("extracted", pa.string()),
        pa.field("correct", pa.bool_(), nullable=False),
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)

REASONING_CHAT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field(
            "messages",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                        pa.field("reasoning_content", pa.string()),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("chat_template_kwargs", pa.struct([pa.field("enable_thinking", pa.bool_())]), nullable=False),
    ]
)


class GeneratedProblem(StrictModel):
    problem: str = Field(min_length=1)
    answer: str = Field(min_length=1)


PROBLEM_TOOL = StructuredTool(
    name="submit_problem",
    description="Submit one self-contained math problem and its short final answer.",
    output_type=GeneratedProblem,
)


@dataclass(frozen=True)
class ProblemAssignment:
    facet_id: str
    facet_description: str
    difficulty: str


@dataclass(frozen=True)
class GenerateProblemsConfig:
    catalog_path: str
    output_path: str
    capability_id: str
    requested_problems: int
    seed: int
    max_completion_tokens: int
    task_specification: str
    relay_job: str


@dataclass(frozen=True)
class SolveProblemsConfig:
    problems_path: str
    output_path: str
    capability_id: str
    samples_per_problem: int
    solutions_per_problem: int
    max_solution_chars: int
    seed: int
    max_completion_tokens: int
    relay_job: str


def capability_packet(catalog: CurriculumCatalog, capability_id: str) -> dict[str, Any]:
    """Select one trainable capability and its subject context from the pinned catalog."""
    for entry in catalog.curricula:
        for section in entry.curriculum.sections:
            if section.id == capability_id and isinstance(section, CapabilitySection):
                return {
                    "catalog_version": catalog.catalog_version,
                    "subject": entry.curriculum.subject_name,
                    "capability_id": section.id,
                    "name": section.name,
                    "outcome": section.outcome,
                    "includes": section.includes,
                    "excludes": section.excludes,
                    "sampling_facets": [facet.model_dump(mode="json") for facet in section.sampling_facets],
                }
    raise ValueError(f"unknown curriculum capability: {capability_id}")


def problem_targets(packet: dict[str, Any]) -> list[dict[str, str]]:
    """The capability's sampling facets followed by each of its ``includes`` entries.

    Many catalog capabilities declare few or no sampling facets; their ``includes`` entries name the
    content the capability covers, so they also serve as generation targets.
    """
    includes = [{"id": f"includes-{index}", "description": text} for index, text in enumerate(packet["includes"])]
    return [*packet["sampling_facets"], *includes]


def problem_assignment(packet: dict[str, Any], index: int) -> ProblemAssignment:
    """Cycle requests through every target, then every difficulty, so coverage does not depend on sampling luck."""
    targets = problem_targets(packet)
    target = targets[index % len(targets)]
    difficulty = DIFFICULTY_TARGETS[(index // len(targets)) % len(DIFFICULTY_TARGETS)]
    return ProblemAssignment(facet_id=target["id"], facet_description=target["description"], difficulty=difficulty)


def problem_prompt(packet: dict[str, Any], assignment: ProblemAssignment, task_specification: str) -> str:
    capability = {key: packet[key] for key in ("subject", "capability_id", "name", "outcome", "includes", "excludes")}
    return (
        "Write one original, self-contained competition-style math problem for the capability below, "
        "and give its final answer.\n"
        "- The answer must be unique and short: a number, exact expression, or small set or tuple, written "
        "in LaTeX without \\boxed and without explanation.\n"
        "- The problem must need several reasoning steps. Avoid textbook drills that only apply one "
        "procedure, and avoid telling the solver which method, checks, or restrictions to use.\n"
        "- Do not copy published contest or benchmark problems, rely on external facts, or ask for a proof.\n"
        "- Solve the problem yourself before submitting, and change it if the answer is not unique.\n"
        f"Capability: {json.dumps(capability, ensure_ascii=False, sort_keys=True)}\n"
        f"Focus for this problem: {assignment.facet_description}\n"
        f"Difficulty target: {assignment.difficulty}\n"
        f"Task specification: {task_specification}"
    )


def problem_request(config: GenerateProblemsConfig, packet: dict[str, Any], index: int) -> dict[str, Any]:
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Write one original math problem with a verified short answer."},
            {
                "role": "user",
                "content": problem_prompt(packet, problem_assignment(packet, index), config.task_specification),
            },
        ],
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "temperature": 1.0,
        "seed": config.seed + index,
        "max_tokens": config.max_completion_tokens,
    }
    body.update(PROBLEM_TOOL.request_fields())
    return {"custom_id": f"problem-{index:05d}", "method": "POST", "url": CHAT_COMPLETIONS_ENDPOINT, "body": body}


def _parse_timeout() -> int | None:
    # math-verify enforces its timeout with signal.alarm, which only the main thread may arm.
    return MATH_VERIFY_TIMEOUT if threading.current_thread() is threading.main_thread() else None


def _parse_answer(text: str) -> list:
    timeout = _parse_timeout()
    return parse(f"${strip_math_delimiters(text)}$", parsing_timeout=timeout) or parse(text, parsing_timeout=timeout)


def is_parsable_answer(text: str) -> bool:
    return any(not isinstance(item, str) for item in _parse_answer(text))


def answers_match(reference: str, candidate: str) -> bool:
    """Whether a candidate answer is math-verify-equivalent to the reference answer."""
    parsed = _parse_answer(candidate)
    return bool(parsed) and verify(_parse_answer(reference), parsed, timeout_seconds=_parse_timeout())


def _responses_by_id(raw_output: str, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        request_id = response["custom_id"]
        if request_id in responses:
            raise ValueError(f"duplicate GLM request ID: {request_id}")
        responses[request_id] = response
    if set(responses) != expected_ids:
        raise ValueError(
            f"GLM batch request IDs differ: missing={sorted(expected_ids - set(responses))}, "
            f"unexpected={sorted(set(responses) - expected_ids)}"
        )
    return responses


def _failure_reason(response: dict[str, Any]) -> str | None:
    result = response.get("response") or {}
    if response.get("error") or result.get("status_code") != 200:
        return "request_failed"
    if result["body"]["choices"][0].get("finish_reason") == "length":
        return "truncated"
    return None


def parse_problem_batch(raw_output: str, config: GenerateProblemsConfig, packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept distinct problems whose reference answer math-verify can parse; keep rejection accounting."""
    responses = _responses_by_id(raw_output, {f"problem-{index:05d}" for index in range(config.requested_problems)})
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(config.requested_problems):
        request_id = f"problem-{index:05d}"
        assignment = problem_assignment(packet, index)
        problem: GeneratedProblem | None = None
        reason = _failure_reason(responses[request_id])
        if reason is None:
            try:
                problem = PROBLEM_TOOL.parse(responses[request_id]["response"]["body"])
            except (UnicodeError, ValidationError, ValueError):
                reason = "invalid_problem"
        if problem is not None and reason is None:
            normalized = " ".join(problem.problem.lower().split())
            if not is_parsable_answer(problem.answer):
                reason = "unparsable_answer"
            elif normalized in seen:
                reason = "duplicate_problem"
            else:
                seen.add(normalized)
        records.append(
            {
                "request_id": request_id,
                "capability_id": config.capability_id,
                "facet_id": assignment.facet_id,
                "difficulty": assignment.difficulty,
                "problem": problem.problem.strip() if problem is not None else None,
                "answer": problem.answer.strip() if problem is not None else None,
                "accepted": reason is None,
                "rejection_reason": reason,
            }
        )
    return records


def solve_request(config: SolveProblemsConfig, problem: dict[str, Any], sample: int, index: int) -> dict[str, Any]:
    return {
        "custom_id": f"{problem['request_id']}-s{sample:02d}",
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
        "body": {
            "model": GLM_MODEL,
            "messages": [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": problem["problem"]},
            ],
            "chat_template_kwargs": {"reasoning_effort": "high"},
            "temperature": 1.0,
            "seed": config.seed + index,
            "max_tokens": config.max_completion_tokens,
        },
    }


def solve_requests(config: SolveProblemsConfig, problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        solve_request(config, problem, sample, problem_index * config.samples_per_problem + sample)
        for problem_index, problem in enumerate(problems)
        for sample in range(config.samples_per_problem)
    ]


def parse_solution_batch(
    raw_output: str, config: SolveProblemsConfig, problems: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grade every blind solution and keep the first correct ones per problem as thinking-mode chat rows."""
    expected_ids = {request["custom_id"] for request in solve_requests(config, problems)}
    responses = _responses_by_id(raw_output, expected_ids)
    solution_records: list[dict[str, Any]] = []
    chat_rows: list[dict[str, Any]] = []
    selected_per_problem: dict[str, int] = defaultdict(int)
    for problem in problems:
        for sample in range(config.samples_per_problem):
            request_id = f"{problem['request_id']}-s{sample:02d}"
            response = responses[request_id]
            reason = _failure_reason(response)
            extracted: str | None = None
            correct = False
            message: dict[str, Any] = {}
            if reason is None:
                message = response["response"]["body"]["choices"][0]["message"]
                content = message.get("content") or ""
                extracted = extract_boxed(content)
                if not (message.get("reasoning") or "").strip():
                    reason = "no_reasoning"
                elif extracted is None:
                    reason = "no_boxed_answer"
                else:
                    correct = answers_match(problem["answer"], extracted)
                    if not correct:
                        reason = "wrong_answer"
                    elif len(message["reasoning"]) + len(content) > config.max_solution_chars:
                        reason = "too_long"
            selected = reason is None and selected_per_problem[problem["request_id"]] < config.solutions_per_problem
            if selected:
                selected_per_problem[problem["request_id"]] += 1
                chat_rows.append(
                    {
                        "id": request_id,
                        "messages": [
                            {"role": "user", "content": problem["problem"], "reasoning_content": None},
                            {
                                "role": "assistant",
                                "content": message["content"].strip(),
                                "reasoning_content": message["reasoning"].strip(),
                            },
                        ],
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                )
            solution_records.append(
                {
                    "request_id": request_id,
                    "problem_request_id": problem["request_id"],
                    "sample": sample,
                    "extracted": extracted,
                    "correct": correct,
                    "selected": selected,
                    "rejection_reason": reason,
                }
            )
    return solution_records, chat_rows


def _glm_client(relay_job: str) -> OpenAIBatchClient:
    return OpenAIBatchClient(resolve_glm_base_url(relay_job), os.environ[GLM_BULK_TOKEN_ENV])


def _run_batch(client: OpenAIBatchClient, requests: list[dict[str, Any]], filename: str) -> str:
    submission = client.submit(requests, filename)
    batch_output = client.output(client.wait(submission.batch_id, POLL_SECONDS))
    if batch_output.errors:
        raise RuntimeError(f"GLM batch {submission.batch_id} returned errors")
    return batch_output.output


def _write_table(output: StoragePath, filename: str, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path = output / filename
    path.parent.mkdirs()
    write_parquet_file(rows, str(path), schema=schema)


def generate_problems(config: GenerateProblemsConfig) -> Artifact:
    """Write problem audit Parquet, exact GLM responses, and a manifest."""
    catalog = TaskCurriculumCatalogArtifact(path=config.catalog_path).read_catalog()
    packet = capability_packet(catalog, config.capability_id)
    requests = [problem_request(config, packet, index) for index in range(config.requested_problems)]
    raw_output = _run_batch(_glm_client(config.relay_job), requests, f"curriculum-problems-{config.capability_id}.jsonl")
    records = parse_problem_batch(raw_output, config, packet)

    output = StoragePath(config.output_path)
    output.mkdirs()
    _write_table(output, PROBLEMS_FILENAME, records, PROBLEM_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(raw_output)
    accepted = sum(record["accepted"] for record in records)
    manifest = {
        "catalog_version": catalog.catalog_version,
        "capability_id": config.capability_id,
        "generator": GLM_MODEL,
        "requested": len(records),
        "accepted": accepted,
        "problems": PROBLEMS_FILENAME,
        "raw_responses": RAW_RESPONSES_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("accepted %s of %s problems for %s", accepted, len(records), config.capability_id)
    return Artifact(path=config.output_path)


def solve_problems(config: SolveProblemsConfig) -> Artifact:
    """Write graded solution audit Parquet, thinking-mode chat Parquet, exact GLM responses, and a manifest."""
    problems_table = pq.read_table(str(StoragePath(config.problems_path) / PROBLEMS_FILENAME))
    problems = [row for row in problems_table.to_pylist() if row["accepted"]]
    requests = solve_requests(config, problems)
    raw_output = _run_batch(
        _glm_client(config.relay_job), requests, f"curriculum-solutions-{config.capability_id}.jsonl"
    )
    solution_records, chat_rows = parse_solution_batch(raw_output, config, problems)
    if not chat_rows:
        raise ValueError(f"no GLM solution matched a reference answer for {config.capability_id}")

    output = StoragePath(config.output_path)
    output.mkdirs()
    _write_table(output, SOLUTIONS_FILENAME, solution_records, SOLUTION_SCHEMA)
    _write_table(output, CHAT_FILENAME, chat_rows, REASONING_CHAT_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(raw_output)
    solved = len({record["problem_request_id"] for record in solution_records if record["correct"]})
    manifest = {
        "capability_id": config.capability_id,
        "generator": GLM_MODEL,
        "problems": len(problems),
        "samples_per_problem": config.samples_per_problem,
        "correct_samples": sum(record["correct"] for record in solution_records),
        "problems_with_correct_sample": solved,
        "chat_rows": len(chat_rows),
        "verified": "answer matches the problem author's reference under math-verify",
        "solutions": SOLUTIONS_FILENAME,
        "chat_data": CHAT_FILENAME,
        "raw_responses": RAW_RESPONSES_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "kept %s solutions over %s of %s problems for %s", len(chat_rows), solved, len(problems), config.capability_id
    )
    return Artifact(path=config.output_path)


def generate_curriculum_problems(
    capability_id: str,
    *,
    catalog: ArtifactStep[TaskCurriculumCatalogArtifact] = TASK_CURRICULUM,
    version: str,
    requested_problems: int,
    seed: int,
    max_completion_tokens: int,
    task_specification: str,
) -> ArtifactStep[Artifact]:
    """Build one GLM problem-generation step for a pinned curriculum capability."""

    def build_config(ctx: StepContext) -> GenerateProblemsConfig:
        return GenerateProblemsConfig(
            catalog_path=ctx.artifact_path(catalog),
            output_path=ctx.output_path,
            capability_id=capability_id,
            requested_problems=requested_problems,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            task_specification=task_specification,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/problems"),
        version=version,
        artifact_type=Artifact,
        run=generate_problems,
        build_config=build_config,
        deps=(catalog,),
    )


def solve_curriculum_problems(
    problems: ArtifactStep[Artifact],
    *,
    capability_id: str,
    version: str,
    samples_per_problem: int,
    solutions_per_problem: int,
    max_solution_chars: int,
    seed: int,
    max_completion_tokens: int,
) -> ArtifactStep[Artifact]:
    """Build one blind GLM solve-and-verify step over a problem artifact.

    ``max_solution_chars`` drops correct solutions whose reasoning and answer would not fit the SFT
    sequence length, so training never sees a think block cut off before its answer.
    """

    def build_config(ctx: StepContext) -> SolveProblemsConfig:
        return SolveProblemsConfig(
            problems_path=ctx.artifact_path(problems),
            output_path=ctx.output_path,
            capability_id=capability_id,
            samples_per_problem=samples_per_problem,
            solutions_per_problem=solutions_per_problem,
            max_solution_chars=max_solution_chars,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/solved-chat"),
        version=version,
        artifact_type=Artifact,
        run=solve_problems,
        build_config=build_config,
        deps=(problems,),
    )
