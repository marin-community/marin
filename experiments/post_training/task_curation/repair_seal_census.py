# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render, measure and normalize a sealed repair cohort's published population.

One job per cohort turns the private oracle-filtered release into the retained-chain
evidence the ``OracleAdmittedReleaseReceipt`` binds: the rendered ``SourceTextPopulation``
(public text plus preserved task and runtime bytes), the fresh actor census over that
population, the normalized QUICK view whose shards carry the grading assessments, and the
within-source output-deduplication evidence. The published task bytes are never altered.
"""

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import DatasetRelease, WithinSourceDedupEvidence
from huggingface_hub import hf_hub_download
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.census_binding import (
    integrity_summary_for_actor,
    receipt_from_census,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population

LOGGER = logging.getLogger(__name__)
SEAL_SCHEMA = NORMALIZED_SCHEMA.append(pa.field("runtime", pa.large_string()))


@dataclass(frozen=True)
class SealCensusConfig:
    source_id: str
    converter_revision: str
    recipe_version: str
    intended_use: IntendedUse
    original_release: DatasetRelease
    filtered_release: DatasetRelease
    tasks_sha256: str
    expected_rows: int
    filtration_uri: str
    output_root: str


def published_rows(tasks_path: str) -> list[tuple[dict, dict, str]]:
    """Every published row as (task document, runtime allocation, stored task bytes)."""
    rows: list[tuple[dict, dict, str]] = []
    with StoragePath(tasks_path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec", "runtime"]):
            for row in batch.to_pylist():
                encoded = row["task_spec"]
                task = TaskSpec.model_validate_json(encoded)
                if task.model_dump_json() != encoded:
                    raise ValueError(f"Published task bytes are not their canonical serialization: {task.id}")
                rows.append((json.loads(encoded), json.loads(row["runtime"]), encoded))
    return rows


def render_population(config: SealCensusConfig, rows: list[tuple[dict, dict, str]], output: str) -> SourceTextPopulation:
    """Render the published cohort's public text without changing task or runtime bytes."""
    files = write_text_population(iter((document, runtime) for document, runtime, _ in rows), output)
    if files.raw_task_count != config.expected_rows or files.task_count != files.raw_task_count:
        raise ValueError(
            f"Sealed cohort must render exactly its admitted population with no exact duplicates: "
            f"{config.source_id}: raw {files.raw_task_count}, unique {files.task_count}, "
            f"expected {config.expected_rows}"
        )
    population = SourceTextPopulation(
        path=output,
        source_id=config.source_id,
        dataset_id=config.filtered_release.repo,
        dataset_revision=config.filtered_release.revision,
        source_version=config.recipe_version,
        converter_revision=config.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    write_record(
        ArtifactRecord(
            output_path=output, result_type=result_type_name(SourceTextPopulation), result=population.result_payload()
        )
    )
    return population


def measure_census(population: SourceTextPopulation, output: str) -> dict[str, Any]:
    """Measure the population's actor and integrity census and bind its receipt."""
    census = measure_population(CensusConfig(population.path, output))
    write_record(
        ArtifactRecord(
            output_path=census.path, result_type=result_type_name(type(census)), result=census.result_payload()
        )
    )
    record = census.model_dump(mode="json")
    integrity = integrity_summary_for_actor(record, {**record, "summary": census.integrity_contracts})
    receipt = receipt_from_census(record, integrity)
    (StoragePath(output) / "actor-census-receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    LOGGER.info(
        "Actor census %s: %d slots, %d failures",
        census.source_id,
        census.daytona_machine_slots,
        census.daytona_execution_failures,
    )
    return receipt.model_dump(mode="json")


def normalized_view(config: SealCensusConfig, rows: list[tuple[dict, dict, str]], output: str) -> str:
    """Write the QUICK normalized shards whose grading assessments feed the grading receipt."""
    normalized = StoragePath(output) / "normalize" / "part-000000.parquet"
    seen: set[str] = set()

    def records() -> Iterator[list[dict]]:
        batch = []
        for _document, runtime, encoded in rows:
            task = TaskSpec.model_validate_json(encoded)
            if task.id in seen:
                raise ValueError(f"Repeated sealed task: {task.id}")
            seen.add(task.id)
            batch.append(
                {
                    **normalization_columns(task.id, task.source, task, None, (), task.source.row),
                    "task_json": encoded,
                    "runtime": json.dumps(runtime),
                }
            )
            if len(batch) >= 32:
                yield batch
                batch = []
        if batch:
            yield batch

    with normalized.open("wb", auto_mkdir=True) as stream, pq.ParquetWriter(stream, SEAL_SCHEMA) as writer:
        for batch in records():
            writer.write_table(pa.Table.from_pylist(batch, schema=SEAL_SCHEMA))
    if len(seen) != config.expected_rows:
        raise ValueError("Normalized view count differs from the admitted population")
    manifest = {
        "mode": SourceProcessingMode.QUICK.value,
        "source_id": config.source_id,
        "repo": config.filtered_release.repo,
        "revision": config.filtered_release.revision,
        "tasks_sha256": config.tasks_sha256,
        "input_rows": len(seen),
        "converted_rows": len(seen),
        "original_dataset": {
            "repo": config.original_release.repo,
            "revision": config.original_release.revision,
            "url": config.original_release.url,
        },
        "retention_evidence": config.filtration_uri,
    }
    (StoragePath(output) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def dedup_evidence(
    population: SourceTextPopulation, config: SealCensusConfig, summary_uri: str
) -> WithinSourceDedupEvidence:
    """The output population is exactly its admitted tasks with zero within-source duplicates."""
    drops = population.within_source_dedup or {}
    if drops.get("within_source_duplicates") != 0 or drops.get("unique_count") != population.task_count:
        raise ValueError(f"Sealed population must be exactly unique: {config.source_id}")
    return WithinSourceDedupEvidence(
        raw_count=population.task_count,
        unique_count=population.task_count,
        within_source_duplicates=0,
        source_dataset_id=config.filtered_release.repo,
        source_revision=config.filtered_release.revision,
        converter_revision=config.converter_revision,
        converter_version=config.recipe_version,
        drop_manifest_uri=str(drops["drop_manifest_uri"]),
        evidence_url=summary_uri,
    )


def census_release(config: SealCensusConfig) -> dict[str, Any]:
    """Run the complete per-cohort seal census and return the qualification inputs."""
    root = StoragePath(config.output_root)
    if root.exists():
        raise FileExistsError(f"Seal census requires a new output root: {config.output_root}")
    tasks_path = hf_hub_download(
        config.filtered_release.repo, "tasks.parquet", repo_type="dataset", revision=config.filtered_release.revision
    )
    if file_sha256(StoragePath(tasks_path)) != config.tasks_sha256:
        raise ValueError("Published cohort differs from its filtration readback")
    rows = published_rows(tasks_path)
    if len(rows) != config.expected_rows:
        raise ValueError(f"Published cohort carries {len(rows)} rows, expected {config.expected_rows}")
    population = render_population(config, rows, f"{config.output_root}/population")
    actor = measure_census(population, f"{config.output_root}/actor")
    normalized = normalized_view(config, rows, f"{config.output_root}/normalized")
    summary_uri = f"{config.output_root}/dedup/summary.json"
    dedup = dedup_evidence(population, config, summary_uri)
    summary = {
        "source_id": config.source_id,
        "population": population.model_dump(mode="json"),
        "dedup": dedup.model_dump(mode="json"),
    }
    StoragePath(summary_uri).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    result = {
        "source_id": config.source_id,
        "publication_evidence": config.filtration_uri,
        "population": population.model_dump(mode="json"),
        "normalized_path": normalized,
        "actor": actor,
        "dedup": dedup.model_dump(mode="json"),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "censuses.json").write_text(json.dumps([result], indent=2) + "\n")
    LOGGER.info("Seal census complete: %s, %d tasks", config.source_id, population.task_count)
    return result


def seal_config_from_filtration(
    filtration_uri: str,
    source_id: str,
    recipe_version: str,
    intended_use: str,
    converter_revision: str,
    output_root: str,
) -> SealCensusConfig:
    filtration = json.loads(StoragePath(filtration_uri).read_text())
    if filtration["original_count"] != filtration["admitted_count"]:
        raise ValueError(f"Sealed cohort must admit its complete population: {source_id}")
    return SealCensusConfig(
        source_id=source_id,
        converter_revision=converter_revision,
        recipe_version=recipe_version,
        intended_use=IntendedUse(intended_use),
        original_release=DatasetRelease(**filtration["original_release"]),
        filtered_release=DatasetRelease(**filtration["filtered_release"]),
        tasks_sha256=filtration["tasks_parquet"]["sha256"],
        expected_rows=filtration["admitted_count"],
        filtration_uri=filtration_uri,
        output_root=output_root,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtration", required=True, help="Filtration evidence JSON on s3")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--recipe-version", required=True)
    parser.add_argument("--intended-use", required=True, choices=["train", "eval"])
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    config = seal_config_from_filtration(
        args.filtration,
        args.source_id,
        args.recipe_version,
        args.intended_use,
        args.converter_revision,
        args.output_root,
    )
    census_release(config)


if __name__ == "__main__":
    main()
