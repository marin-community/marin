# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Edge's SFT preparation plans with a separately declared planning judge."""

import json
from dataclasses import replace

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import RUBRIC_CHECKLIST

from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPO = "yatin-superintelligence/Edge-Agent-Reasoning-WebSearch-260K"
REVISION = "7e8e455fff52e6d21dce4ce4a5a1bddd13031e1a"
SOURCE_ID = "Yatin:Edge-Agent-Reasoning-WebSearch-260K"
VERSION = "edge-websearch-preparation-judge-v1"
NATIVE_FIELDS = frozenset({"batch_index_id", "role", "industry", "os", "user_prompt", "agent_reasoning"})
CARD_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"
PREPARATION_CONTRACT = (
    "You are a preparatory router for a downstream executor. Produce a preparation plan for the simulated "
    "user request below, rather than carrying out its requested actions. Do not claim to have searched, "
    "executed commands or verified facts. Cover the objective and constraints; known facts versus assumptions; "
    "material ambiguities and useful clarification questions; a concrete verification checklist; and proposed "
    "targeted web-search queries. State when an aspect has no material relevance instead of inventing a gap. "
    "Use the supplied source context. No particular section heading, word count or number of queries is required."
)
CRITERIA = (
    "The preparation plan correctly identifies the simulated request's objective and explicit constraints, "
    "respecting the supplied role, industry and operating-system context without pretending to execute the task.",
    "The plan distinguishes grounded facts from assumptions and identifies material knowledge gaps without "
    "unsupported certainty or fabricated results.",
    "The plan identifies material ambiguities and useful clarification questions where they exist; it does not "
    "invent requirements or penalize an already clear request merely for lacking ambiguities.",
    "The verification checklist names concrete facts, dependencies or authoritative sources needed to prepare "
    "execution, tailored to the actual request rather than generic boilerplate.",
    "The proposed search queries target the request's actual verification needs and are useful to downstream "
    "retrieval. They are plans, not claims that a live search or tool call occurred.",
)


def convert_plan(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    if (
        set(native) != NATIVE_FIELDS
        or type(native["batch_index_id"]) is not int
        or any(
            not isinstance(native[field], str) or not native[field].strip()
            for field in NATIVE_FIELDS - {"batch_index_id"}
        )
    ):
        return source_defect("incomplete_edge_native_contract", "Expected the six original fields with nonempty text")
    metadata = {field: native[field] for field in ("role", "industry", "os")}
    instruction = PREPARATION_CONTRACT + "\n\nSource context:\n" + json.dumps(metadata, ensure_ascii=False)
    events = (
        TextMessage(role="system", content=instruction),
        TextMessage(role="user", content=native["user_prompt"]),
    )
    spec = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        rubric=RUBRIC_CHECKLIST,
        criteria=CRITERIA,
        context="conversation.json",
        question="Evaluate the candidate preparation plan against the source-derived planning contract and request.",
        references=(),
        system_prompt="",
        prompt_template="",
        label_scores={},
        exact_gate=False,
    )
    package = verifyit_package(
        spec,
        (
            inline_resource(
                "conversation.json", json.dumps([event.model_dump(mode="json") for event in events]).encode()
            ),
            inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
        ),
    )
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "preparation-planning", "prompt-only-judge"),
            "labels": {"source_id": SOURCE_ID, "native_batch_index": str(native["batch_index_id"])},
        }
    )
    change = NormalizationChange(
        field="context.events[0]",
        reason=f"Declare the preparatory-router task described in the pinned source card: {CARD_URL}",
        original="No system instruction is stored in the native six-field row; the task scope is described by its card.",
        replacement=instruction,
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.YATIN,
            upstream_dataset="Edge-Agent-Reasoning-WebSearch-260K",
            upstream_name="Edge-Agent-Reasoning-WebSearch-260K",
            build_path=VERSION,
            family="preparation-planning",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=260293,
            notes=(
                "New MEDIUM checklist judge for the preparation-planning task described by the pinned source card. "
                "An explicit source-derived system contract is recorded as normalization; raw user_prompt and "
                "role/industry/OS strings are unchanged. Generated agent_reasoning stays private provenance, "
                "never judge gold or native oracle. There is no native executable grader or live search tool. "
                "All 267 data shards contain 260,293 rows; four non-data files are separately accounted for. "
                "The card declares 263,098 rows, a 2,805-row metadata discrepancy. Generator checkpoint identity "
                "is undeclared. No conversion or grading quality claim follows from those source descriptions."
            ),
        ),
        config=CurationRecipe(
            name="edge-websearch-preparation",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            convert=convert_plan,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
