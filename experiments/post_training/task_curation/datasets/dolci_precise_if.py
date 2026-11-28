# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare a prompt-only instruction judge for Dolci's Precise IF SFT subset."""

import json
from dataclasses import replace

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import RUBRIC_CHECKLIST

from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPO = "allenai/Dolci-Think-SFT-Clean-v2"
REVISION = "b83abf4073b66569ba788ba25f4a092f9ad2a116"
SOURCE_ID = "AllenAI:Dolci-Think-SFT-Clean-v2/precise-if"
VERSION = "dolci-precise-if-prompt-judge-v1"
NATIVE_SUBSET = "Dolci Think Precise IF"


def selected(row: dict, context: ConversionContext) -> bool:
    return row.get("source") == NATIVE_SUBSET


def convert_precise_if(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    native = dict(row.data)
    messages = native.get("messages")
    if (
        native.get("source") != NATIVE_SUBSET
        or not isinstance(native.get("id"), str)
        or not isinstance(messages, list)
        or len(messages) != 2
        or any(not isinstance(message, dict) for message in messages)
        or [message.get("role") for message in messages] != ["user", "assistant"]
        or any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages)
    ):
        return source_defect(
            "invalid_precise_if_conversation", "Expected a Precise IF user/assistant pair with native ID"
        )
    prompt = messages[0]["content"]
    events = (TextMessage(role="user", content=prompt),)
    # The SFT export has no native rubric or executable verifier. Its generated
    # assistant answer is provenance, not a trusted gold answer or judge reference.
    spec = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        rubric=RUBRIC_CHECKLIST,
        criteria=(
            "The candidate answers the user's complete request substantively and accurately and satisfies every "
            "explicit content, language, formatting and counting constraint in the original user message. "
            "Do not infer missing requirements or relax conflicting instructions. If any required condition "
            "is unmet, score zero. Evaluate only the candidate's final answer.",
        ),
        context="conversation.json",
        question="Evaluate the candidate as the next assistant response to the original user message.",
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
    task = conversation_task(row, events=events, package=package)
    return task.model_copy(
        update={
            "tags": ("train", "instruction-following", "prompt-only-judge"),
            "labels": {"source_id": SOURCE_ID, "native_id": native["id"]},
        }
    )


def source(files: tuple[str, ...]) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.ALLENAI,
            upstream_dataset="Dolci-Think-SFT-Clean-v2",
            upstream_name="Dolci-Think-SFT-Clean-v2",
            subset="precise-if",
            build_path=VERSION,
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=132159,
            notes=(
                "New prompt-only MEDIUM judge; SFT references remain private provenance, never gold. "
                "No native scorer or executable constraint metadata is claimed."
            ),
        ),
        config=CurationRecipe(
            name="dolci-precise-if",
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET, select=selected),
            convert=convert_precise_if,
            version=VERSION,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
