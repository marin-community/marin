# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive the repaired identity-following variant's readmission gates at the variant identity.

The repair re-entered ingestion at stage b (a converter/prompt change), so the variant
re-runs every later gate on its own pin: the normalized text population with its
within-source deduplication, the actor census, the lexical measurement, and the source
fingerprint for the cross-source gate. Each subcommand writes its durable receipt beside
the population; the registry appends and the declaration wiring are separate ruled steps.
"""

import argparse
import hashlib
import json
import logging
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import write_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.source_fingerprints import fingerprint_source

logger = logging.getLogger(__name__)

VARIANT_SOURCE_ID = "TaskTrove:laion__nemotron-gym-identity-following-v4@repaired-20261123"
RELEASE_REPO = "open-athena/tasktrove-identity-following-repaired-20261123"
RELEASE_REVISION = "94967dd29123f9d1274a660491b6612fbd6ebd40"
UPSTREAM_REPO = "open-thoughts/TaskTrove"
UPSTREAM_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
UPSTREAM_URL = f"https://huggingface.co/datasets/{UPSTREAM_REPO}/tree/{UPSTREAM_REVISION}"


def population_documents(parquet_uri: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every published TaskSpec as (task document, runtime document), in stable shard order."""
    documents = []
    with StoragePath(parquet_uri).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                document = json.loads(row["task_spec"])
                runtime = json.loads(row["runtime"]) if row.get("runtime") else {}
                documents.append((document, runtime))
    return documents


def write_population_record(parquet_uri: str, output_path: str, converter_revision: str) -> SourceTextPopulation:
    """Deduplicate the published population into the lexical input population and record it."""
    files = write_text_population(population_documents(parquet_uri), output_path)
    logger.info(
        "Population written: %d unique of %d raw (%d duplicates), task_specs_sha256 %s",
        files.task_count,
        files.raw_task_count,
        files.within_source_duplicates,
        files.task_specs_sha256,
    )
    return SourceTextPopulation(
        path=output_path,
        source_id=VARIANT_SOURCE_ID,
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        source_version="identity-repair-20261123-v2",
        converter_revision=converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=f"{output_path}/outputs/main",
        specs_output_dir=f"{output_path}/specs",
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )


def fingerprint_population(population: SourceTextPopulation) -> dict[str, Any]:
    """The variant's SourceFingerprint over its own spec shards and input digests."""
    inputs: dict[str, str] = {}
    for shard in sorted(str(path) for path in StoragePath(f"{population.specs_output_dir}/*.parquet").glob()):
        data = StoragePath(shard).read_bytes()
        inputs[shard] = hashlib.sha256(data).hexdigest()
    documents = []
    for shard in sorted(inputs):
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["task_spec"])
        documents.extend(json.loads(encoded) for encoded in table.column("task_spec").to_pylist())
    fingerprint = fingerprint_source(documents, inputs)
    return fingerprint.model_dump(mode="json")


def main() -> None:
    """Run the gates CLI for the repaired variant's population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("population", "fingerprints"))
    parser.add_argument("--parquet", required=True, help="The published population's tasks.parquet URI")
    parser.add_argument("--output-root", required=True, help="Versioned artifact prefix for the gate receipts")
    parser.add_argument("--converter-revision", required=True, help="The landing commit the producer ran at")
    parser.add_argument("--populations-out", required=True, help="Write the SourceTextPopulation record JSON here")
    parser.add_argument("--output", required=True, help="Write the subcommand's result JSON here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()

    if args.command == "population":
        population = write_population_record(args.parquet, args.output_root, args.converter_revision)
        # The artifact-store record excludes ``path``: raw_load passes the location itself as the
        # handle, so a payload that carried it would collide (the backfill lane's lesson).
        write_artifact(population.model_dump(mode="json", exclude={"path"}), args.output_root)
        encoded = population.model_dump_json(indent=2) + "\n"
        (StoragePath(args.output_root) / "population-record.json").write_text(encoded)
        StoragePath(args.populations_out).write_text(json.dumps([population.model_dump(mode="json")], indent=2) + "\n")
        logger.info("Population record written to %s", args.populations_out)
        return
    record = json.loads(StoragePath(args.populations_out).read_text())
    population = SourceTextPopulation.model_validate(record[0] if isinstance(record, list) else record)
    fingerprints = {population.source_id: fingerprint_population(population)}
    payload = json.dumps(fingerprints, indent=2, sort_keys=True) + "\n"
    (StoragePath(args.output_root) / "fingerprints.json").write_text(payload)
    StoragePath(args.output).write_text(payload)
    logger.info("Fingerprints written to %s", args.output_root)


if __name__ == "__main__":
    main()
