# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-only NeMo next actions graded by exact function names and JSON arguments."""

from taskcompendium.convert.answers import unsupported
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.nemo_predicted_action import DATASET, REVISION, import_row
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        task = import_row(row.data, canonical_sha256(row.data))
    except ValueError as error:
        return unsupported("predicted_action_contract", str(error))
    return task.model_copy(
        update={"id": row.id, "source": row.source, "tags": ("eval", "agentic", "single-turn", "tool-call-proxy")}
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="NeMo:predicted-action",
                title="NeMo conversational tool-use next actions",
                origin="NeMo",
                family="function-calling",
                tags=("eval", "agentic", "single-turn", "license:cc-by-4.0"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                notes="Exact function and JSON argument matching; alternative valid calls receive zero.",
            ),
            config=CurationRecipe(
                name="nemo-predicted-action",
                version="1",
                source=HfSource(DATASET, REVISION, ("train.jsonl",), SourceFormat.JSONL),
                convert=convert,
                intended_use=IntendedUse.EVAL,
            ),
            pipeline=process_rows,
        )
    ]
