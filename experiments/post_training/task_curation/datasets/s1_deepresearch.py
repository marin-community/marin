# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert S1-DeepResearch questions with two declared judge families.

The source ships deep-research trajectories with a dataset-authored reference answer for
its verifiable (closed-ended multi-hop) tasks and no answer for most generation
(open-ended exploration) tasks. Closed-ended rows are graded by a reference judge against
``meta.answer``; every open-ended row - answered or not - is graded by the same prompt-only
checklist rubric, so each family has exactly one contract. The generated assistant and tool
trajectory turns stay private provenance: the task context carries only the source's own
system instruction and first user message, both verbatim.
"""

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

REPO = "ScienceOne-AI/S1-DeepResearch-15k"
REVISION = "1e9685c7c7277514a7eac1ba286a5b941cfca312"
SOURCE_ID = "ScienceOne:S1-DeepResearch-15k"
VERSION = "s1-deepresearch-research-judge-v1"
COUNT = 15000
CLOSED_ENDED = "Closed-ended Multi-hop Resolution"
OPEN_ENDED = "Open-ended Exploration"
LANGUAGES = frozenset({"en", "zh"})
ROLES = frozenset({"system", "user", "assistant", "tool"})
CARD_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"
ANSWER_PROVENANCE_RECEIPT = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/s1-deepresearch/2026.11.26.1/"
    "answer-provenance-v1/receipt.json"
)
REFERENCE_JUDGE_QUESTION = (
    "Determine whether the candidate's final answer is correct for the posed research question, "
    "judged against the reference answer. A response that states the same answer in different "
    "wording is correct; an answer that disagrees with, contradicts or fails to address the "
    "reference is incorrect."
)
CHECKLIST_CRITERIA = (
    "The response directly addresses the posed research request and answers what was actually asked, "
    "including any explicit output-structure or constraint requirements stated in the request.",
    "Claims are grounded: the response distinguishes what the request's materials and stated facts "
    "support from inference, and does not fabricate sources, data or results.",
    "The reasoning is coherent: intermediate steps or sub-questions are resolved consistently and the "
    "conclusion follows from the response's own content.",
    "The response is complete for the request's scope: material sub-parts of a multi-part request are "
    "each addressed rather than dropped.",
    "The response is usable as delivered: it is self-contained, organized so the answer is identifiable, "
    "and does not refer to unavailable tools, files or follow-up turns.",
)


def _reference_judge(answer: str):
    """The closed-ended contract: the dataset-authored reference answer is the judge's gold."""
    return replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        question=REFERENCE_JUDGE_QUESTION,
        references=(answer,),
        temperature=0.0,
        max_completion_tokens=64,
    )


def _checklist_judge():
    """The one open-ended contract: prompt-only research-quality checklist, no references."""
    return replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        rubric=RUBRIC_CHECKLIST,
        criteria=CHECKLIST_CRITERIA,
        context="conversation.json",
        question="Evaluate the candidate's research response against the posed open-ended research request.",
        references=(),
        system_prompt="",
        prompt_template="",
        label_scores={},
        exact_gate=False,
    )


def convert_research(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    messages, meta = native.get("messages"), native.get("meta")
    if (
        set(native) != {"messages", "meta"}
        or not isinstance(messages, list)
        or len(messages) < 2
        or not isinstance(meta, dict)
        or set(meta) != {"id", "question", "answer", "language", "type"}
    ):
        return source_defect(
            "incomplete_s1_native_contract", "Expected the messages/meta pair with the five meta fields"
        )
    if any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or message["role"] not in ROLES
        or not isinstance(message["content"], str)
        for message in messages
    ):
        return source_defect(
            "invalid_s1_native_messages", "Every native message must carry a known role and text content"
        )
    system_message, user_message = messages[0], messages[1]
    if system_message["role"] != "system" or user_message["role"] != "user":
        return source_defect(
            "invalid_s1_native_opening", "The trajectory must open with a system message then a user message"
        )
    task_type, answer = meta["type"], meta["answer"]
    if task_type not in {CLOSED_ENDED, OPEN_ENDED}:
        return source_defect("invalid_s1_task_type", "The type label must be one of the two declared families")
    if meta["language"] not in LANGUAGES or not isinstance(meta["id"], str) or not meta["id"].strip():
        return source_defect("invalid_s1_task_metadata", "The id must be a nonempty string and the language en or zh")
    if not isinstance(meta["question"], str) or not meta["question"].strip():
        return source_defect("missing_s1_question", "The question field is required")
    if task_type == CLOSED_ENDED and (not isinstance(answer, str) or not answer.strip()):
        return source_defect(
            "missing_closed_ended_reference",
            "The card requires a reference answer for every verifiable closed-ended task",
        )
    family = "closed-ended-multi-hop" if task_type == CLOSED_ENDED else "open-ended-exploration"
    judge = _reference_judge(answer.strip()) if task_type == CLOSED_ENDED else _checklist_judge()
    if family == "closed-ended-multi-hop":
        package = verifyit_package(judge)
    else:
        events_for_grader = [
            {"role": system_message["role"], "content": system_message["content"]},
            {"role": user_message["role"], "content": user_message["content"]},
        ]
        package = verifyit_package(
            judge,
            (
                inline_resource("conversation.json", json.dumps(events_for_grader, ensure_ascii=False).encode()),
                inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
            ),
        )
    events = (
        TextMessage(role="system", content=system_message["content"]),
        TextMessage(role="user", content=user_message["content"]),
    )
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": (
                "train",
                "deep-research",
                "judge",
                "reference" if family == "closed-ended-multi-hop" else "checklist",
                "s1",
            ),
            "labels": {
                "source_id": SOURCE_ID,
                "native_id": meta["id"],
                "family": family,
                "language": meta["language"],
            },
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason=(
            f"Carry only the source's own task definition into the context; generated turns stay private "
            f"provenance. Answer provenance receipt: {ANSWER_PROVENANCE_RECEIPT}. Source card: {CARD_URL}"
        ),
        original=(
            "The native trajectory interleaves generated assistant and tool turns after the opening "
            "system and user messages; the source card describes meta.answer as a reference answer."
        ),
        replacement=(
            "The task context carries the verbatim system instruction and first user message; the "
            "generated assistant and tool turns stay private provenance, excluded from the context "
            "and never serve as judge gold."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.SCIENCE_ONE,
            upstream_dataset="S1-DeepResearch-15k",
            upstream_name="S1-DeepResearch-15k",
            build_path=VERSION,
            family="deep-research",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "Two declared judge families under one 15,000-task population: closed-ended multi-hop rows are "
                "graded by a MEDIUM reference judge against the dataset-authored meta.answer, and every "
                "open-ended exploration row - answered or not - by the same MEDIUM prompt-only research-quality "
                "checklist, so each family has exactly one contract. The card describes meta.answer as a "
                "reference answer required for verifiable and optional for generation tasks; the sealed "
                "provenance receipt records that the answer string appears only inside the final assistant "
                "turn for 14/1,192 answered rows, nowhere in the trajectory for 586, and disagrees with the "
                "final turn's stated answer in 234 - the label is dataset-authored, not trajectory-derived. "
                "The task context carries the source's own system instruction and first user message verbatim; "
                "generated assistant and tool turns stay private provenance and never serve as judge gold. "
                "Bilingual en/zh is kept as shipped; the card declares approximately 15k samples and the "
                "sealed inventory measures exactly 15,000 native rows."
            ),
        ),
        config=CurationRecipe(
            name="s1-deepresearch-research-judge",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.JSONL),
            convert=convert_research,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
