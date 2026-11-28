# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Requalify the pinned CodeContests TaskSpecs without inventing grader changes.

The exclusion claim calls for QA of the existing converter-v4 output. The repair
retains every task contract and records the ingestion identity change. Actual
backend/oracle qualification must still verify the environment's dependency pins.
"""

from finestore.rl.curation import TaskFormat
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, NormalizationChange, NormalizedTask, OracleCommand, RawRow

from experiments.post_training.task_curation.ingest_stages.converter_binding import RowConverterBinding


def convert(row: RawRow, _context: ConversionContext) -> NormalizedTask:
    original = TaskSpec.model_validate_json(row.data["task_spec"])
    repaired = original.model_copy(update={"id": row.id, "source": row.source})
    changes = tuple(
        NormalizationChange(
            field=name,
            reason="Requalification at the pinned Atlas input through the maintained repair workflow",
            original=before,
            replacement=after,
        )
        for name, before, after in (
            ("id", original.id, repaired.id),
            ("source", original.source.model_dump_json(), repaired.source.model_dump_json()),
        )
        if before != after
    )
    return NormalizedTask(repaired, changes)


def golden(task: TaskSpec) -> OracleCommand | None:
    if any(resource.path == "solution/solve.sh" for resource in task.resources.oracle):
        return OracleCommand("bash /solution/solve.sh")
    return None


INGEST_CONVERTER = RowConverterBinding(
    convert=convert,
    source_format=SourceFormat.PARQUET,
    task_format=TaskFormat.TASKSPEC,
    grader=None,
    controls=Controls(golden=golden),
)
