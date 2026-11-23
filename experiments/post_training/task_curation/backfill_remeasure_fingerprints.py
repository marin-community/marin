# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-measure stale source fingerprints over their current prepared populations.

The population preflight compares every surveyed source's fingerprint inputs against the
registry's prepared-population cache. Sources whose prepared cache moved since their last
fingerprint measurement fail the preflight. This job re-measures each stale source's
fingerprint over its CURRENT population — recovered from the prepared cache's recorded
inputs digest against the raw population files — with the shared fingerprint builder, so
the survey payload matches the current populations by measurement, never by copying.
"""

import argparse
import json
import logging
import subprocess

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.source_fingerprints import fingerprint_source

LOGGER = logging.getLogger(__name__)


def raw_documents(raw_uri: str) -> list[dict]:
    rows = []
    with StoragePath(raw_uri).open("rb") as stream:
        table = pq.read_table(stream)
        column = "lowered_task_spec" if "lowered_task_spec" in table.schema.names else "task_spec"
        for encoded in table.column(column).to_pylist():
            rows.append(json.loads(encoded))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="The raw population directory holding the sources' input files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", action="append", required=True, help="source_id=prepared_inputs_sha256")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    out = subprocess.run(["uv", "run", "fsutil", "ls", args.raw_dir + "/"], capture_output=True, text=True)
    files = [line.split()[-1] for line in out.stdout.splitlines() if line.endswith(".parquet")]
    LOGGER.info("Raw population files: %d", len(files))

    remeasured = {}
    for assignment in args.source:
        source_id, _, expected = assignment.partition("=")
        recovered = None
        for name in files:
            sha = name.removesuffix(".parquet")
            uri = f"{args.raw_dir}/{name}"
            if canonical_sha256({uri: sha}) == expected:
                recovered = (uri, sha)
                break
        if recovered is None:
            raise ValueError(f"No raw file matches the prepared inputs digest for {source_id}: {expected}")
        uri, sha = recovered
        LOGGER.info("Recovered %s inputs: %s", source_id, uri)
        fingerprint = fingerprint_source(raw_documents(uri), {uri: sha})
        remeasured[source_id] = fingerprint.model_dump(mode="json")
        LOGGER.info("Re-measured %s: %d tasks", source_id, fingerprint.task_count)

    output = StoragePath(args.output)
    payload = json.dumps(remeasured, indent=2, sort_keys=True) + "\n"
    (output / "fingerprints.json").write_text(payload)
    print(json.dumps({key: value["task_count"] for key, value in remeasured.items()}, indent=2))


if __name__ == "__main__":
    main()
