# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the source fingerprints for the requirement-backfilled variant populations.

Issue #11976 step 3: the cross-source gate and the atlas fingerprint index need a
deterministic ``SourceFingerprint`` per variant population, computed over the same
normalized texts the parent sources were measured on. Reads the frozen backfilled
populations in region, fingerprints each through the shared builder with the variant
declaration's ``FingerprintInput``, and writes the fingerprint documents beside the
populations. Nothing crosses regions; the index registration happens separately under
the registry lock.
"""

import argparse
import hashlib
import json
import logging

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.source_fingerprints import fingerprint_source
from experiments.post_training.task_curation.sources import all_sources

LOGGER = logging.getLogger(__name__)


def population_inputs(population: str) -> dict[str, str]:
    """The population files that determine the fingerprint, as {uri: sha256}."""
    inputs = {}
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    for shard in shards:
        data = StoragePath(shard).read_bytes()
        inputs[shard] = hashlib.sha256(data).hexdigest()
    return inputs


def documents(population: str) -> list[dict]:
    rows = []
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["task_spec"])
        rows.extend(json.loads(encoded) for encoded in table.column("task_spec").to_pylist())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="The fingerprint documents' output prefix")
    parser.add_argument("--source", action="append", required=True, help="source_id=population_prefix")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    sources = all_sources()
    fingerprints = {}
    for assignment in args.source:
        source_id, _, population = assignment.partition("=")
        sources[source_id].declaration.model_dump(mode="json")
        inputs = population_inputs(population)
        LOGGER.info("Fingerprinting %s (%s) over %d input files", source_id, population, len(inputs))
        fingerprint = fingerprint_source(documents(population), inputs)
        fingerprints[source_id] = fingerprint.model_dump(mode="json")
        LOGGER.info(
            "Fingerprinted %s: %d tasks, %d duplicates, sample sha %s",
            source_id,
            fingerprint.task_count,
            fingerprint.within_source_duplicates,
            fingerprint.sample_sha256[:16],
        )

    output = StoragePath(args.output)
    payload = json.dumps(fingerprints, indent=2, sort_keys=True) + "\n"
    (output / "fingerprints.json").write_text(payload)
    print(json.dumps({key: {"task_count": value["task_count"]} for key, value in fingerprints.items()}, indent=2))


if __name__ == "__main__":
    main()
