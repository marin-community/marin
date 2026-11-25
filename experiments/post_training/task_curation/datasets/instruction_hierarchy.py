# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the source's system/user messages and private instruction rubrics."""

import json
from dataclasses import replace

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
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

REPO = "nvidia/Nemotron-System-User-Instruction-Hierarchy-v1"
REVISION = "7f4939cdafbcc7d67aa643157a106f5f96866237"
SOURCE_ID = "Nvidia:Nemotron-System-User-Instruction-Hierarchy-v1"
VERSION = "instruction-hierarchy-checklist-v1"
SUBSET_COUNTS = {"conflicting": 14607, "supplementary": 13782, "user_only": 15414}
CONTEXT_FILE = "conversation.json"


def convert_hierarchy(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    subset = next((name for name in SUBSET_COUNTS if row.source.row.startswith(f"data/{name}/")), None)
    if subset is None:
        return unsupported("unknown_hierarchy_subset", row.source.row)
    native = dict(row.data)
    if not isinstance(native.get("user_query"), str) or not native["user_query"].strip():
        return source_defect("missing_user_query", "The source row has no user instruction")
    system = native.get("system_prompt")
    if not isinstance(system, str) or not system.strip() or (subset == "user_only") != (system == "N/A"):
        return source_defect("invalid_system_instruction", "Only user_only declares the documented N/A sentinel")
    entries = native.get("evaluation_criteria")
    if not isinstance(entries, list) or not entries:
        return source_defect("missing_evaluation_criteria", "No checklist criteria are declared")
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("criterion"), str)
        or not entry["criterion"].strip()
        or entry.get("raw_value_json") != ""
        for entry in entries
    ):
        return unsupported("unhandled_evaluation_criterion", "Preserve non-object or empty criteria for repair")
    events = (
        *((TextMessage(role="system", content=system),) if subset != "user_only" else ()),
        TextMessage(role="user", content=native["user_query"]),
    )
    # The source supplies criteria, not an executable scorer or a named judge.
    # Bind the maintained checklist fraction harness with the MEDIUM deployment settings.
    spec = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        rubric=RUBRIC_CHECKLIST,
        criteria=tuple(entry["criterion"] for entry in entries),
        context=CONTEXT_FILE,
        question="Evaluate the candidate as the next assistant response to this system/user conversation.",
        system_prompt="",
        prompt_template="",
        label_scores={},
    )
    package = verifyit_package(
        spec,
        (
            inline_resource(
                CONTEXT_FILE, json.dumps([{"role": event.role, "content": event.content} for event in events]).encode()
            ),
            inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
        ),
    )
    task = conversation_task(row, events=events, package=package)
    return task.model_copy(
        update={
            "tags": ("train", "instruction-following", "instruction-hierarchy", "judge", "checklist"),
            "labels": {"source_id": SOURCE_ID, "source_subset": subset, "source_rubric_id": str(native.get("id", ""))},
        }
    )


def source() -> RlDataSource[CurationRecipe]:
    """Declare the three source TRAIN configurations as one pinned population."""
    return RlDataSource(
        pipeline=process_rows,
        info=SourceInfo(
            origin=SourceOrigin.NVIDIA,
            upstream_dataset="Nemotron-System-User-Instruction-Hierarchy-v1",
            upstream_name="Nemotron-System-User-Instruction-Hierarchy-v1",
            build_path="Source hierarchy rubrics with the maintained checklist judge",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=sum(SUBSET_COUNTS.values()),
            family="instruction-following",
            notes="Source criteria preserved; no reference responses. The MEDIUM judge awaits qualification.",
        ),
        config=CurationRecipe(
            name="nemotron-instruction-hierarchy",
            source=HfSource(
                repo=REPO,
                revision=REVISION,
                files=tuple(f"data/{subset}/rubrics.jsonl" for subset in SUBSET_COUNTS),
                format=SourceFormat.JSONL,
            ),
            convert=convert_hierarchy,
            version=VERSION,
            intended_use=IntendedUse.TRAIN,
        ),
    )
