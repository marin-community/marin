# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local ingestion fixture shared by callable, CLI and catalog scenarios."""

import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.invocation import CurationSource, PipelineResult, PipelineRun
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference


@dataclass(frozen=True)
class LocalNumbers:
    path: str

    def __call__(self, source: CurationSource, run: PipelineRun) -> PipelineResult:
        path = Path(run.source_input or self.path)
        values = [int(line) for line in path.read_text().splitlines()]
        if run.mode == SourceProcessingMode.SAMPLE:
            values = values[:2]
        factor = int(Path(run.inputs["factor"]).read_text()) if "factor" in run.inputs else 2
        numbers = [value * factor for value in values]
        output = StoragePath(run.output_path) / "numbers.json"
        output.write_text(json.dumps(numbers))
        evidence = StoragePath(run.output_path) / "ingestion.json"
        evidence.write_text(json.dumps({"source": source.name, "mode": run.mode, "input": str(path)}))
        return PipelineResult(
            SourceStatus.SAMPLED if run.mode == SourceProcessingMode.SAMPLE else SourceStatus.COMPLETED,
            {"numbers": str(output)},
            {"ingestion": str(evidence)},
            ("ingest", "multiply"),
        )


def number_source(path: Path, name: str = "numbers") -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            id=f"fixture:{name}",
            title=name,
            origin="fixture",
            dataset=SourceReference("local-numbers", "pinned", "https://example.org/numbers"),
        ),
        pipeline=LocalNumbers(str(path)),
        version="1",
    )
