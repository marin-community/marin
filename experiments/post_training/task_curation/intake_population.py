# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind completed, licensed QUICK conversions to the standard lexical population writer."""

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.license_binding import licensed_source
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.sources import pending_measurement_sources


@dataclass(frozen=True)
class QuickPopulationConfig:
    source_id: str
    dataset_id: str
    revision: str
    version: str
    converter_revision: str
    manifest_path: str
    manifest_sha256: str
    inputs: tuple[tuple[str, str], ...]
    converted_rows: int
    rejections: dict[str, int]
    output_path: str


def quick_documents(config: QuickPopulationConfig) -> Iterable[tuple[dict, dict]]:
    for filename, expected in config.inputs:
        path = StoragePath(filename)
        if file_sha256(path) != expected:
            raise ValueError(f"QUICK normalization bytes changed: {path}")
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=16, columns=["task_json"]):
                for encoded in batch.column(0).to_pylist():
                    if encoded is not None:
                        yield json.loads(encoded), {}
        if file_sha256(path) != expected:
            raise ValueError(f"QUICK normalization changed during measurement: {path}")


def normalize_quick(config: QuickPopulationConfig) -> SourceTextPopulation:
    if file_sha256(StoragePath(config.manifest_path)) != config.manifest_sha256:
        raise ValueError("QUICK manifest changed before population preparation")
    files = write_text_population(quick_documents(config), config.output_path)
    if files.raw_task_count != config.converted_rows or files.task_count == 0:
        raise ValueError("QUICK normalization population differs from its completed manifest")
    if file_sha256(StoragePath(config.manifest_path)) != config.manifest_sha256:
        raise ValueError("QUICK manifest changed during population preparation")
    return SourceTextPopulation(
        path=config.output_path,
        source_id=config.source_id,
        dataset_id=config.dataset_id,
        dataset_revision=config.revision,
        source_version=config.version,
        converter_revision=config.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections=config.rejections,
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )


def bind_output(config: QuickPopulationConfig, ctx: StepContext) -> QuickPopulationConfig:
    return replace(config, output_path=ctx.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Explicit pending pipeline name")
    parser.add_argument("--quick", required=True, help="Completed source QUICK output directory")
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    source = licensed_source(next(source for source in pending_measurement_sources() if source.name == args.source))
    dataset = source.dataset
    assert dataset is not None and dataset.revision is not None
    root = StoragePath(args.quick)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in (
        ("source", source.name),
        ("source_dataset", dataset.name),
        ("source_revision", dataset.revision),
        ("recipe_revision", source.version),
        ("mode", "quick"),
    ):
        if manifest[name] != expected:
            raise ValueError(f"QUICK {name} differs from the pending declaration")
    files = tuple((str(path), file_sha256(path)) for path in sorted((root / "normalize" / "*.parquet").glob(), key=str))
    config = QuickPopulationConfig(
        source_id=source.info.id,
        dataset_id=dataset.name,
        revision=dataset.revision,
        version=source.version,
        converter_revision=args.converter_revision,
        manifest_path=str(manifest_path),
        manifest_sha256=file_sha256(manifest_path),
        inputs=files,
        converted_rows=manifest["converted_rows"],
        rejections=manifest["rejections"],
        output_path="",
    )
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name=f"rl-data/atlas/intake-population/{source.name}",
            version=args.artifact_version,
            artifact_type=SourceTextPopulation,
            run=normalize_quick,
            build_config=partial(bind_output, config),
        )
        if not args.run:
            args.output.write_text(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}, indent=2) + "\n")
            return
        with local_client():
            result = run(step)[0]
        args.output.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")


if __name__ == "__main__":
    main()
