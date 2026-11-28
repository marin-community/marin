# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the re-pinned native conversions' source-identity survey in-region.

Re-pinned conversion populations need measured fingerprints and prepared caches
at their current inputs. Run in the populations' region, this stage:

1. copies each staged population's spec shards to the survey's durable
   source-identity prefix and verifies SHA-256 digests after upload;
2. fingerprints each complete population and creates its prepared content cache,
   refusing unless both task counts match the declared census;
3. replaces the target fingerprints and prepared-population locators together,
   discarding reviews and folds that touch replaced fingerprints while
   preserving other sources' measurements and locations;
4. runs the deterministic content-NN neighbor survey focused on the re-pinned
   sources (no LLM), using the fresh locations during input preflight.

Prints one JSON summary with the artifact URIs and SHA-256 digests the
FingerprintReference registry append consumes.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import SourceFingerprint

from experiments.post_training.task_curation.catalog_identity import (
    canonical_identity_survey,
    identity_survey_payload,
)
from experiments.post_training.task_curation.lexical_scan import population_key
from experiments.post_training.task_curation.source_fingerprints import (
    SourceStanding,
    duplicate_groups,
    fingerprint_source,
)
from experiments.post_training.task_curation.source_neighbor_survey import SAMPLE_SIZE, rerun, write_json
from experiments.post_training.task_curation.source_population_cache import (
    CONTENT_SCHEMA,
    PopulationLocations,
    PreparedPopulation,
    read_locator_manifest,
)
from experiments.post_training.task_curation.task_neighbors import content_rows
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

logger = logging.getLogger(__name__)

STAGED_POPULATION_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text"


def spec_documents(paths: list[Path]):
    """Stream parsed task specs from downloaded spec shards."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["task_spec"]):
            yield from (json.loads(encoded) for encoded in batch.column("task_spec").to_pylist())


def staged_shards(staged_root: str, source_id: str, population_version: str) -> list[StoragePath]:
    key = population_key(source_id)
    shards = (StoragePath(staged_root) / key / population_version / "specs" / "*.parquet").glob()
    if not shards:
        raise ValueError(f"No spec shards staged for {source_id} at {staged_root}/{key}/{population_version}/specs/")
    return sorted(shards, key=lambda shard: shard.name)


def durable_copy(payload: bytes, destination: StoragePath) -> None:
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Immutable survey input already differs: {destination}")
        return
    with destination.open("wb", auto_mkdir=True) as stream:
        stream.write(payload)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"Uploaded bytes differ after write: {destination}")


def measure_population(
    source_id: str,
    staged_root: str,
    population_version: str,
    survey_root: StoragePath,
    work_dir: Path,
    expected_tasks: int,
) -> tuple[SourceFingerprint, dict[str, str], dict[str, str], PreparedPopulation]:
    """Copy, fingerprint and prepare one population; returns its survey artifacts."""
    key = population_key(source_id)
    source_work = work_dir / key
    source_work.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, str] = {}
    files: dict[str, str] = {}
    local_paths: list[Path] = []
    for shard in staged_shards(staged_root, source_id, population_version):
        payload = shard.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = survey_root / "populations" / key / "specs" / shard.name
        durable_copy(payload, destination)
        inputs[str(destination)] = digest
        files[digest] = str(destination)
        local = source_work / shard.name
        local.write_bytes(payload)
        local_paths.append(local)
    logger.info("Copied %d shards for %s", len(local_paths), original_source_id(source_id))

    documents = spec_documents(local_paths)
    fingerprint = fingerprint_source(documents, inputs)
    if fingerprint.task_count != expected_tasks:
        raise ValueError(
            f"Fingerprint does not cover the declared population for {source_id}: "
            f"{fingerprint.task_count} != {expected_tasks}"
        )
    logger.info("Fingerprinted %s: %d tasks", original_source_id(source_id), fingerprint.task_count)

    prepared_path = source_work / "prepared-population.parquet"
    count = 0
    with pq.ParquetWriter(prepared_path, CONTENT_SCHEMA, compression="zstd") as writer:
        for batch in prepared_batches(spec_documents(local_paths)):
            writer.write_table(batch)
            count += batch.num_rows
    if count != expected_tasks:
        raise ValueError(f"Prepared population count differs for {source_id}: {count} != {expected_tasks}")
    prepared_payload = prepared_path.read_bytes()
    prepared_uri = survey_root / "populations" / key / "prepared-population.parquet"
    durable_copy(prepared_payload, prepared_uri)
    prepared = PreparedPopulation(
        uri=str(prepared_uri),
        sha256=hashlib.sha256(prepared_payload).hexdigest(),
        inputs_sha256=canonical_sha256(inputs),
        task_count=expected_tasks,
    )
    return fingerprint, inputs, files, prepared


def prepared_batches(documents, batch_rows: int = 256):
    """Write prepared-population batches from streamed task specs."""
    pending: list[dict] = []
    for row in content_rows(documents):
        pending.append(asdict(row))
        if len(pending) >= batch_rows:
            yield pa.Table.from_pylist(pending, schema=CONTENT_SCHEMA)
            pending = []
    if pending:
        yield pa.Table.from_pylist(pending, schema=CONTENT_SCHEMA)


def merge_survey_input(
    served_uri: str, served_sha256: str, fingerprints: dict[str, SourceFingerprint], survey_root: StoragePath
) -> tuple[str, str]:
    payload = StoragePath(served_uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != served_sha256:
        raise ValueError("Served identity survey SHA-256 differs")
    survey = canonical_identity_survey(json.loads(payload))
    replaced = {
        source_id for source_id, fingerprint in fingerprints.items() if survey.fingerprints.get(source_id) != fingerprint
    }
    merged_fingerprints = {**survey.fingerprints, **fingerprints}
    reviews = [
        review
        for review in survey.reviews
        if review.similarity.left not in replaced and review.similarity.right not in replaced
    ]
    standings = {
        source_id: SourceStanding(fingerprint.task_count)
        for source_id, fingerprint in merged_fingerprints.items()
        if fingerprint is not None
    }
    survey = survey.model_copy(
        update={
            "fingerprints": merged_fingerprints,
            "reviews": reviews,
            "groups": duplicate_groups(reviews, standings),
        }
    )
    uri = str(survey_root / "source-identities-input.json")
    sha = write_json(uri, identity_survey_payload(survey))
    logger.info("Staged survey input %s SHA256 %s", uri, sha)
    return uri, sha


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="source_id=expected_tasks")
    parser.add_argument("--population-version", required=True)
    parser.add_argument("--staged-population-root", default=STAGED_POPULATION_ROOT)
    parser.add_argument("--survey-root", required=True)
    parser.add_argument("--served-survey-uri", required=True)
    parser.add_argument("--served-survey-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/native-source-survey"))
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    configure_coreweave_s3()
    survey_root = StoragePath(args.survey_root)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    fingerprints: dict[str, SourceFingerprint] = {}
    inputs: dict[str, dict[str, str]] = {}
    files: dict[str, str] = {}
    prepared: dict[str, PreparedPopulation] = {}
    for assignment in args.source:
        source_id, _, expected = assignment.partition("=")
        if not expected.isdigit():
            raise ValueError(f"--source needs source_id=expected_tasks: {assignment}")
        canonical = canonical_source_id(source_id)
        fingerprint, source_inputs, source_files, source_prepared = measure_population(
            canonical,
            args.staged_population_root,
            args.population_version,
            survey_root,
            args.work_dir,
            int(expected),
        )
        fingerprints[canonical] = fingerprint
        inputs[canonical] = source_inputs
        files.update(source_files)
        prepared[canonical] = source_prepared

    survey_input_uri, survey_input_sha = merge_survey_input(
        args.served_survey_uri, args.served_survey_sha256, fingerprints, survey_root
    )
    survey_work = args.work_dir / "survey"
    prior_locations = read_locator_manifest(
        Path("experiments/post_training/task_curation/source_population_locators.json")
    )
    locations = PopulationLocations(
        files={**prior_locations.files, **files},
        prepared={**prior_locations.prepared, **prepared},
    )
    summary = rerun(
        canonical_identity_survey(json.loads(StoragePath(survey_input_uri).read_bytes())),
        args.survey_root,
        survey_work,
        args.workers,
        SAMPLE_SIZE,
        False,
        args.allocated_cpus,
        locations,
        source_survey_uri=survey_input_uri,
        source_survey_sha256=survey_input_sha,
        source_ids=set(fingerprints),
    )
    result = {
        "survey_input_uri": survey_input_uri,
        "survey_input_sha256": survey_input_sha,
        "survey_uri": summary["survey_uri"],
        "survey_sha256": summary["survey_sha256"],
        "task_counts": {original_source_id(source_id): fingerprints[source_id].task_count for source_id in fingerprints},
        "input_sha256": {
            original_source_id(source_id): canonical_sha256(inputs[source_id]) for source_id in fingerprints
        },
        "files": files,
        "prepared": {
            original_source_id(source_id): prepared[source_id].model_dump(mode="json") for source_id in prepared
        },
        "pairs": summary["pairs"],
        "labels": summary["labels"],
        "changed_pairs": summary["changed_pairs"],
    }
    print("SURVEY RESULT " + json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
