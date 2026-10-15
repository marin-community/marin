# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL candidates as single-turn tasks: every expert turn of the prepared pivot trajectories.

The rows are the long-format candidate sets PivotRL measures pass rates on: NVIDIA's SWE and
Terminal pivot releases laid out one candidate per expert turn by ``experiments/datasets/nemotron_pivot.py``,
and the OpenHands candidates of ``experiments/post_training/pivotrl/runs.py``. They are not the
pre-filtered pivot components of the Nemotron Ultra blend. Each source reads its candidate artifact
directly, so the curated tasks and the pass rates describe the same turns.

Every task is graded in process by a deterministic verifyit mode that applies PivotRL's grader:
``pivot-tool-call`` (NeMo Gym's SWE argument comparison), ``terminus`` (Terminus-2 keystroke
similarity) or ``next-action`` (OpenHands functional next action, rule tier only). Preparation grades
each task's expert turn, which must score 1, and an empty reply, which must score 0; a SWE pivot whose
expert action is a text message accepts any reply, even an empty one, so curation rejects it. A task carries
its candidate's ``source_id`` as a :data:`CANDIDATE_TAG_PREFIX` tag, which joins it to pass rates.

The prompts differ from the candidate rows in two ways. TaskSpec conversations carry no reasoning,
so the teacher's earlier reasoning, which candidate prompts keep as ``reasoning_content``, is dropped.
Terminal tasks answer in plain text, whose answer format adds its instruction as a final user message.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.rl.nemotron_pivot import CANDIDATES_FILENAME, VALIDATION_FILENAME
from pydantic import JsonValue, ValidationError
from taskcompendium.chat import chat_input
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.nemotron_ultra import functions
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    PlainText,
    TaskSpec,
    VerifyitGrader,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import Mode

from experiments.datasets.nemotron_pivot import nemotron_pivot_datasets
from experiments.post_training.pivotrl.compendium import CANDIDATE_TAG_PREFIX
from experiments.post_training.pivotrl.pipeline import full_candidate_step
from experiments.post_training.pivotrl.runs import OPENHANDS_CANDIDATES
from experiments.post_training.task_curation.pipeline import ArtifactSource, CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

RECIPE_VERSION = "1"
PUBLIC_MESSAGE_FIELDS = ("role", "content", "tool_calls", "tool_call_id")


@dataclass(frozen=True)
class PivotGrade:
    """A candidate's verifyit mode and parameters, and the tools it may answer with."""

    mode: Mode
    parameters: dict[str, JsonValue]
    tools: tuple[FunctionDefinition, ...]


def _ultra_record(data: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(data["extra_info"]["nemotron_ultra"]["record_json"])


def swe_grade(data: Mapping[str, Any]) -> PivotGrade:
    request = json.loads(data["extra_info"]["nemotron_ultra"]["request_json"])
    expected_action = json.dumps(_ultra_record(data)["expected_action"], sort_keys=True)
    return PivotGrade(Mode.PIVOT_TOOL_CALL, {"expected_action": expected_action}, functions(request.get("tools", [])))


def terminal_grade(data: Mapping[str, Any]) -> PivotGrade:
    record = _ultra_record(data)
    parameters: dict[str, JsonValue] = {"expected_answer": record["expected_answer"]}
    if record.get("threshold") is not None:
        parameters["threshold"] = record["threshold"]
    return PivotGrade(Mode.TERMINUS, parameters, ())


def openhands_grade(data: Mapping[str, Any]) -> PivotGrade:
    turn = data["extra_info"]["openhands"]
    parameters: dict[str, JsonValue] = {
        "expected_call": turn["expected_call_json"],
        "repo_root": turn["repo_root"],
        "cwd": turn["cwd"],
    }
    return PivotGrade(Mode.NEXT_ACTION, parameters, functions(json.loads(turn["tools_json"])))


def public_messages(prompt: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], tuple[NormalizationChange, ...]]:
    """The chat fields a TaskSpec conversation keeps, dropping Parquet's nulls and earlier reasoning."""
    messages = [
        {key: message[key] for key in PUBLIC_MESSAGE_FIELDS if message.get(key) is not None} for message in prompt
    ]
    reasoned = sum(bool(message.get("reasoning_content")) for message in prompt)
    if not reasoned:
        return messages, ()
    change = NormalizationChange(
        field="prompt[*].reasoning_content",
        reason="TaskSpec conversations carry no reasoning; the teacher's earlier reasoning is dropped",
        original=f"reasoning on {reasoned} assistant messages",
        replacement="Removed",
    )
    return messages, (change,)


def convert_candidate(
    grade: Callable[[Mapping[str, Any]], PivotGrade], row: RawRow, _context: ConversionContext
) -> NormalizedTask | ImportRejection:
    """Convert a candidate row to a single-turn task that ``grade`` describes."""
    pivot = grade(row.data)
    if pivot.mode != Mode.TERMINUS and not pivot.tools:
        return source_defect("missing_tools", f"A {pivot.mode} candidate advertises no tools to answer with")
    messages, changes = public_messages(row.data["prompt"])
    try:
        context = chat_input(messages)
    except (ValueError, ValidationError) as error:
        return source_defect("malformed_prompt", str(error))
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(),
        final_tools=pivot.tools,
        answer_type=AnswerType.NATIVE_ACTION if pivot.tools else AnswerType.TEXT,
        answer_format=FinalAction() if pivot.tools else PlainText(),
        grader=VerifyitGrader(mode=pivot.mode, parameters=pivot.parameters),
        tags=("pivotrl", f"{CANDIDATE_TAG_PREFIX}{row.data['extra_info']['source_id']}"),
    )
    return NormalizedTask(task, changes)


@dataclass(frozen=True)
class CandidateSet:
    key: str
    title: str
    step: ArtifactStep[Artifact]
    grade: Callable[[Mapping[str, Any]], PivotGrade]


def candidate_sets() -> tuple[CandidateSet, ...]:
    releases = nemotron_pivot_datasets()
    return (
        CandidateSet("swe", "Nemotron SWE pivot turns", releases["swe"], swe_grade),
        CandidateSet("terminal", "Nemotron Terminal pivot turns", releases["terminal"], terminal_grade),
        CandidateSet(
            "openhands", "SWE-rebench OpenHands turns", full_candidate_step(OPENHANDS_CANDIDATES), openhands_grade
        ),
    )


def pipeline_name(key: str, split: IntendedUse) -> str:
    return f"pivotrl-{key}" if split == IntendedUse.TRAIN else f"pivotrl-{key}-validation"


def sources() -> list[RlDataSource]:
    """One training and one validation source for each candidate set."""
    splits = ((IntendedUse.TRAIN, CANDIDATES_FILENAME), (IntendedUse.EVAL, VALIDATION_FILENAME))
    return [
        RlDataSource(
            info=SourceInfo(
                id=f"PivotRL:{pipeline_name(candidates.key, split)}",
                title=f"{candidates.title} · {'train' if split == IntendedUse.TRAIN else 'validation'}",
                origin="PivotRL",
                family="pivotrl",
                tags=("agentic", "tool-use", "single-turn", "pivot"),
            ),
            config=CurationRecipe(
                name=pipeline_name(candidates.key, split),
                source=ArtifactSource(candidates.step, (filename,), SourceFormat.PARQUET),
                convert=partial(convert_candidate, candidates.grade),
                version=RECIPE_VERSION,
                intended_use=split,
            ),
            pipeline=process_rows,
        )
        for candidates in candidate_sets()
        for split, filename in splits
    ]
