# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local ingestion with a real upstream artifact, independent of recipe services."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from finestore.rl.curation import OversightEvidence, OversightLevel
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, SourceReference
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence

NUMBERS_DATASET = SourceReference(
    "open-athena/numbers-fixture",
    "a" * 40,
    "https://huggingface.co/datasets/open-athena/numbers-fixture/tree/" + "a" * 40,
)


@dataclass(frozen=True)
class NumbersConfig:
    name: str
    input_path: Path
    version: str = "1"
    dataset: SourceReference = NUMBERS_DATASET
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestionRun:
    input_path: str
    output_path: str


def ingest_numbers(run: IngestionRun) -> None:
    (StoragePath(run.output_path) / "numbers.txt").write_text(Path(run.input_path).read_text())


@dataclass(frozen=True)
class NumbersRun:
    input_path: str
    output_path: str
    factor_path: str | None
    limit: int | None


def numbers_pipeline(source: RlDataSource[NumbersConfig], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    settings = source.config
    assert settings is not None
    path = settings.input_path
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    upstream = ArtifactStep(
        name=f"task-curation/numbers-input/{digest}",
        version="2026.10.07",
        artifact_type=Artifact,
        run=ingest_numbers,
        build_config=lambda ctx: IngestionRun(str(path), ctx.output_path),
    )
    overridden = options.inputs.root
    dependencies = (upstream,) if overridden is None else ()

    def config(ctx: StepContext) -> NumbersRun:
        input_path = overridden or str(StoragePath(ctx.artifact_path(upstream)) / "numbers.txt")
        return NumbersRun(
            input_path,
            ctx.output_path,
            options.inputs.auxiliary.get("factor"),
            2 if options.mode == SourceProcessingMode.SAMPLE else None,
        )

    def execute(run: NumbersRun) -> CampaignArtifact:
        values = [int(line) for line in Path(run.input_path).read_text().splitlines()][: run.limit]
        factor = int(Path(run.factor_path).read_text()) if run.factor_path is not None else 2
        output = StoragePath(run.output_path) / "numbers.json"
        output.write_text(json.dumps([value * factor for value in values]))
        evidence = StoragePath(run.output_path) / "ingestion.json"
        evidence.write_text(
            json.dumps(
                {"source": source.name, "catalog_id": source.info.id, "mode": options.mode, "input": run.input_path}
            )
        )
        result = PipelineResult(
            SourceStatus.SAMPLED if options.mode == SourceProcessingMode.SAMPLE else SourceStatus.COMPLETED,
            {"numbers": str(output)},
            {"ingestion": str(evidence)},
            ("ingest", "multiply"),
        )
        return CampaignArtifact(path=run.output_path, status=result.status, result=result)

    return ArtifactStep(
        name=f"data/rl/{source.name}",
        version="2026.10.07",
        artifact_type=CampaignArtifact,
        run=execute,
        build_config=config,
        deps=dependencies,
    )


def number_source(path: Path, name: str = "numbers") -> RlDataSource[NumbersConfig]:
    return RlDataSource(
        info=SourceInfo(
            id=f"Marin:{name}",
            upstream_dataset=name,
            upstream_name=name,
            build_path="test",
            origin=SourceOrigin.MARIN,
            oversight=OversightEvidence(
                level=OversightLevel.FULL, verification="Fixture arithmetic ingestion is executed and checked locally."
            ),
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture ingestion performs arithmetic locally and creates no actor.",
            lexical_contamination=decontamination_evidence(NUMBERS_DATASET.name, NUMBERS_DATASET.revision, 1),
        ),
        config=NumbersConfig(name, path),
        pipeline=numbers_pipeline,
    )
