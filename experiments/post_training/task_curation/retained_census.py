# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure retained private releases with the existing complete actor census."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import WithinSourceDedupEvidence
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec, format_conversation
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.census_binding import integrity_summary_for_actor, receipt_from_census
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import PARTITION_ROWS, TEXT_SCHEMA
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population
from experiments.post_training.task_curation.remediation_binding import PublicationReceipt
from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    RetainedImport,
    run_retained_import,
)
from experiments.post_training.task_curation.within_source_census import DROP_SCHEMA

LOGGER = logging.getLogger(__name__)


def retained_population(publication: PublicationReceipt, output: str, converter_revision: str) -> SourceTextPopulation:
    """Render retained public text from stored specs while keeping task and runtime bytes unchanged."""
    root = StoragePath(output)
    text_output = root / "outputs/main"
    spec_digest, text_digest = hashlib.sha256(), hashlib.sha256()
    count, partition = 0, 0
    rows = []

    def flush() -> None:
        nonlocal partition
        with (text_output / f"part-{partition:06d}.parquet").open("wb", auto_mkdir=True) as stream:
            pq.write_table(pa.Table.from_pylist(rows, schema=TEXT_SCHEMA), stream, compression="zstd")
        rows.clear()
        partition += 1

    for path in sorted(StoragePath(publication.specs_output_dir + "/*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=16, columns=["task_spec", "task_sha256"]):
                for row in batch.to_pylist():
                    encoded = row["task_spec"]
                    if hashlib.sha256(encoded.encode()).hexdigest() != row["task_sha256"]:
                        raise ValueError("Retained regional copy differs from its content hash")
                    task = TaskSpec.model_validate_json(encoded)
                    text = format_conversation(task.context.events)
                    spec_digest.update(encoded.encode())
                    text_digest.update(json.dumps([task.id, text], ensure_ascii=False).encode())
                    rows.append({"id": task.id, "text": text, "partition_id": partition})
                    count += 1
                    if len(rows) >= PARTITION_ROWS:
                        flush()
    if rows:
        flush()
    if count != publication.verified_retained_tasks:
        raise ValueError("Retained census input count differs from publication readback")
    population = SourceTextPopulation(
        path=output,
        source_id=publication.source_id,
        dataset_id=publication.repo,
        dataset_revision=publication.revision,
        source_version=RETAINED_IMPORT_VERSION,
        converter_revision=converter_revision,
        text_renderer_version=publication.lexical.text_renderer_version,
        task_count=count,
        text_sha256=text_digest.hexdigest(),
        task_specs_sha256=spec_digest.hexdigest(),
        main_output_dir=str(text_output),
        specs_output_dir=publication.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=output, result_type=result_type_name(SourceTextPopulation), result=population.result_payload()
        )
    )
    return population


def census_release(publication: PublicationReceipt, root: str, converter_revision: str) -> dict:
    output = StoragePath(root) / hashlib.sha256(publication.source_id.encode()).hexdigest()[:20]
    if output.exists():
        raise FileExistsError(f"Retained census requires a new output version: {output}")
    population = retained_population(publication, str(output / "population"), converter_revision)
    census = measure_population(CensusConfig(population.path, str(output / "actor")))
    write_record(
        ArtifactRecord(
            output_path=census.path, result_type=result_type_name(type(census)), result=census.result_payload()
        )
    )
    record = census.model_dump(mode="json")
    integrity = integrity_summary_for_actor(record, {**record, "summary": census.integrity_contracts})
    actor = receipt_from_census(record, integrity)
    cohort = RetainedCohort(
        name=publication.source_id,
        version=RETAINED_IMPORT_VERSION,
        release=publication.release,
        expected_rows=publication.verified_retained_tasks,
        tasks_parquet_sha256=publication.tasks_parquet_sha256,
        intended_use=IntendedUse.TRAIN if publication.plan.lexical_use.value == "train" else IntendedUse.EVAL,
        evidence_url=publication.evidence_url,
    )
    # The corpus adapter uses task_json; its importer preserves the published JSON bytes.
    run_retained_import(RetainedImport(cohort, str(output / "normalized")))
    manifest = output / "dedup/drops.parquet"
    with manifest.open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist([], schema=DROP_SCHEMA), stream)
    dedup = WithinSourceDedupEvidence(
        raw_count=population.task_count,
        unique_count=population.task_count,
        within_source_duplicates=0,
        source_dataset_id=publication.repo,
        source_revision=publication.revision,
        converter_revision=converter_revision,
        converter_version=RETAINED_IMPORT_VERSION,
        drop_manifest_uri=str(manifest),
        evidence_url=publication.evidence_url,
    )
    result = {
        "source_id": publication.source_id,
        "publication_evidence": publication.evidence_url,
        "population": population.model_dump(mode="json"),
        "normalized_path": str(output / "normalized"),
        "actor": actor.model_dump(mode="json"),
        "dedup": dedup.model_dump(mode="json"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Measured retained population: %s, %d tasks", publication.source_id, population.task_count)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    publications = [
        PublicationReceipt.model_validate(row)
        for path in args.releases
        for row in json.loads(StoragePath(path).read_text())
    ]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_coreweave_s3) as pool:
        futures = [
            pool.submit(census_release, publication, args.output_root, args.converter_revision)
            for publication in publications
        ]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: row["source_id"])
    (StoragePath(args.output_root) / "censuses.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
