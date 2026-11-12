# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize exact published TaskSpec artifacts without repeating their converters."""

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.source_identity import canonical_source_id, original_source_id
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.local import artifact_prefix


@dataclass(frozen=True)
class PublishedPopulation:
    source_id: str
    dataset_id: str
    dataset_revision: str
    converter_version: str
    converter_revision: str
    tasks_parquet_uri: str
    tasks_parquet_sha256: str
    task_count: int
    converter_rejections: dict[str, int]
    admission_rejections: dict[str, int]


@dataclass(frozen=True)
class PublishedTextConfig:
    population: PublishedPopulation
    tasks_path: str
    output_path: str


def published_documents(config: PublishedTextConfig) -> Iterable[tuple[dict, dict]]:
    source = StoragePath(config.tasks_path)
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != config.population.tasks_parquet_sha256:
        raise ValueError(f"Published TaskSpec bytes changed: {config.population.source_id}")
    with source.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec"]):
            for encoded in batch.column(0).to_pylist():
                yield json.loads(encoded), {}


def normalize_published(config: PublishedTextConfig) -> SourceTextPopulation:
    files = write_text_population(published_documents(config), config.output_path)
    metadata = config.population
    if files.task_count != metadata.task_count or files.task_count == 0:
        raise ValueError(f"{metadata.source_id}: emitted {files.task_count}, expected {metadata.task_count} TaskSpecs")
    return SourceTextPopulation(
        path=config.output_path,
        source_id=metadata.source_id,
        dataset_id=metadata.dataset_id,
        dataset_revision=metadata.dataset_revision,
        source_version=metadata.converter_version,
        converter_revision=metadata.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections=metadata.converter_rejections,
        admission_rejections=metadata.admission_rejections,
    )


def published_text_config(
    population: PublishedPopulation, source: ArtifactStep, ctx: StepContext
) -> PublishedTextConfig:
    return PublishedTextConfig(
        replace(population, source_id=original_source_id(population.source_id)),
        ctx.artifact_path(source),
        ctx.output_path,
    )


def published_text_steps(populations: list[PublishedPopulation], version: str) -> list[ArtifactStep]:
    steps = []
    for population in populations:
        key = canonical_sha256(original_source_id(population.source_id))[:20]
        source = ArtifactStep.adopt(f"rl-data/atlas/lexical/published/{key}", version, population.tasks_parquet_uri)
        steps.append(
            ArtifactStep(
                name=f"rl-data/atlas/lexical/text/{key}",
                version=version,
                artifact_type=SourceTextPopulation,
                deps=[source],
                run=normalize_published,
                build_config=partial(published_text_config, population, source),
            )
        )
    return steps


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    populations = [
        PublishedPopulation(**{**row, "source_id": canonical_source_id(row["source_id"])})
        for row in json.loads(args.manifest.read_text())
    ]
    if len({p.source_id for p in populations}) != len(populations):
        raise ValueError("Each published source must occur once")
    with artifact_prefix(args.artifact_root):
        steps = published_text_steps(populations, args.artifact_version)
        if not args.run:
            args.output.write_text(
                json.dumps(
                    [
                        {"source_id": p.source_id, "path": step.path()}
                        for p, step in zip(populations, steps, strict=True)
                    ],
                    indent=2,
                )
                + "\n"
            )
            return
        with ExitStack() as cleanup:
            client = LocalClient()
            cleanup.callback(client.shutdown)
            cleanup.enter_context(set_current_client(client))
            results = run(*steps, max_concurrent=2)
        args.output.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
