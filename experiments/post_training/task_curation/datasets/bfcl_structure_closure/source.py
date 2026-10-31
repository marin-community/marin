# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate BFCL structural completion tasks within the normal curation pipeline."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import FunctionDefinition, TaskSpec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, RawRow
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    ClosureCase,
    Family,
    Split,
    closure_format,
    oracle_cases,
    split_membership,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.prefixes import near_boundary_case
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tasks import case_task
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tool_serialization import (
    ToolPrompt,
    native_tool_case,
    write_result_tool,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


class TrainingPrefix(StrEnum):
    ORIGINAL = "original"
    NEAR_BOUNDARY = "near-boundary"


@dataclass(frozen=True)
class CaseReader:
    """Expand the staged manifest and oracle files into one case per source row."""

    split: Split
    split_filename: str
    training_prefix: TrainingPrefix

    def __call__(self, manifest_file: StoragePath, context: ConversionContext) -> Iterator[dict[str, Any]]:
        partition = json.loads(manifest_file.read_text())
        split = json.loads((context.inputs["split"] / self.split_filename).read_text())
        train, validation = split_membership(partition, split)
        members = train if self.split == Split.TRAIN else validation
        for parent in partition["complement"]:
            if parent["source_id"] not in members:
                continue
            oracle = context.inputs["oracles"] / "bfcl_complement" / parent["name"] / "solution" / "solve.sh"
            for case in oracle_cases(parent["source_id"], oracle.read_bytes(), self.split):
                if self.training_prefix == TrainingPrefix.NEAR_BOUNDARY:
                    case = near_boundary_case(case)
                yield asdict(case)


@lru_cache(maxsize=2)
def _tokenizer(path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)


@dataclass(frozen=True)
class CaseConverter:
    """Convert generated cases to assistant-prefill TaskSpecs."""

    tool_prompt: ToolPrompt

    def __call__(self, row: RawRow, context: ConversionContext) -> TaskSpec:
        case = ClosureCase(**(dict(row.data) | {"family": Family(row.data["family"])}))
        tools = ()
        if self.tool_prompt == ToolPrompt.NATIVE_DECLARED and closure_format(case.family) == ClosureFormat.TOOL:
            case = native_tool_case(case, _tokenizer(str(context.inputs["tokenizer"])))
            tools = (FunctionDefinition.model_validate(write_result_tool()["function"]),)
        task = case_task(case, row.source, tools=tools)
        return task.model_copy(update={"id": row.id})


def sources(
    manifest: HfSource,
    oracles: HfSource,
    split_manifest: HfSource,
    *,
    split_filename: str,
    training_prefix: TrainingPrefix,
    tool_prompt: ToolPrompt,
    tokenizer: HfSource | None = None,
) -> list[RlDataSource[CurationRecipe]]:
    """Declare both cohorts after their source and split revisions are pinned."""
    if tool_prompt == ToolPrompt.NATIVE_DECLARED and tokenizer is None:
        raise ValueError("Native tool serialization requires a pinned tokenizer")
    inputs = {"oracles": oracles, "split": split_manifest}
    if tokenizer is not None:
        inputs["tokenizer"] = tokenizer
    return [
        RlDataSource(
            info=SourceInfo(
                id=f"BFCLStructureClosure:{cohort}",
                title=f"BFCL structural closure · {cohort}",
                origin="BFCL complement",
                family="structural",
                tags=("rlvr", "single-turn", "structural-closure", "assistant-prefill", "generator"),
                notes="Oracle-derived completion cases; validation parents are disjoint from training.",
            ),
            config=CurationRecipe(
                name=f"bfcl-structure-closure-{cohort}",
                source=HfSource(
                    manifest.repo,
                    manifest.revision,
                    manifest.files,
                    SourceFormat.GENERATED,
                    read=CaseReader(cohort, split_filename, training_prefix),
                ),
                convert=CaseConverter(tool_prompt),
                version="1",
                intended_use=IntendedUse.TRAIN if cohort == Split.TRAIN else IntendedUse.EVAL,
                controls=Controls(golden=reference_reply),
                inputs=inputs,
            ),
            pipeline=process_rows,
        )
        for cohort in Split
    ]
