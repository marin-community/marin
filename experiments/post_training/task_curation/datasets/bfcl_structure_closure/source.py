# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate BFCL structural completion tasks within task curation."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, RawRow
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    Split,
    closure_format,
    oracle_cases,
    parse_case,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.inputs import (
    INVENTORY_FILE,
    ORACLE_ROOT,
    REPOSITORY,
    REVISION,
    SPLIT_FILE,
    TOKENIZER_FILES,
    OracleDownload,
    cohort_parents,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.prefixes import near_boundary_case
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tasks import case_task
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tool_serialization import result_tool
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def _load_tokenizer(root: StoragePath) -> PreTrainedTokenizerFast:
    with TemporaryDirectory(prefix="bfcl-tokenizer-") as directory:
        for filename in TOKENIZER_FILES:
            file = root / filename
            file.download_to(str(Path(directory) / file.name))
        return PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)


@dataclass(frozen=True)
class CaseReader:
    """Expand each selected oracle into structural completion rows."""

    split: Split

    def __call__(self, inventory_file: StoragePath, context: ConversionContext) -> Iterator[dict[str, Any]]:
        parents = cohort_parents(inventory_file, context.inputs["split"] / SPLIT_FILE, self.split)
        tokenizer = _load_tokenizer(context.inputs["tokenizer"])
        for parent in parents:
            oracle = parent.verify(parent.staged_path(inventory_file).read_bytes())
            for case in oracle_cases(parent.task, oracle, self.split, tokenizer):
                yield asdict(near_boundary_case(case))


def _convert_case(row: RawRow, _context: ConversionContext) -> TaskSpec:
    case = parse_case(row.data)
    tools = (result_tool(),) if closure_format(case.family) == ClosureFormat.TOOL else ()
    return case_task(case, row.source, row.id, tools=tools)


def sources() -> list[RlDataSource[CurationRecipe]]:
    """Declare both audited cohorts with immutable input revisions."""
    split = HfSource(REPOSITORY, REVISION, (SPLIT_FILE,), SourceFormat.JSON)
    tokenizer = HfSource(REPOSITORY, REVISION, TOKENIZER_FILES, SourceFormat.GENERATED)
    return [
        RlDataSource(
            info=SourceInfo(
                id=f"BFCLStructureClosure:{cohort}",
                title=f"BFCL structural closure · {cohort}",
                origin="BFCL complement",
                family="structural",
                tags=("rlvr", "single-turn", "structural-closure", "assistant-prefill", "generator"),
                notes="Oracle-derived completion cases with disjoint training and validation parents.",
            ),
            config=CurationRecipe(
                name=f"bfcl-structure-closure-{cohort}",
                source=HfSource(
                    REPOSITORY,
                    REVISION,
                    (INVENTORY_FILE,),
                    SourceFormat.GENERATED,
                    read=CaseReader(cohort),
                    prepare=OracleDownload(ORACLE_ROOT),
                ),
                convert=_convert_case,
                version="1",
                intended_use=IntendedUse.TRAIN if cohort == Split.TRAIN else IntendedUse.EVAL,
                controls=Controls(golden=reference_reply),
                inputs={"split": split, "tokenizer": tokenizer},
            ),
            pipeline=process_rows,
        )
        for cohort in Split
    ]
