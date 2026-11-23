# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the backfill survey: the current composed survey + the backfilled variants.

Merges the variants' computed fingerprints and the re-measured fingerprint for a source
whose prepared cache moved, marks sources whose prepared cache no longer matches any
survey document as pending (``None`` — their standing index entries remain the evidence),
and seals the composed survey at a fresh output root with its SHA-256.
"""

import argparse
import hashlib
import json
import logging

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--survey-sha256", required=True)
    parser.add_argument("--fingerprints", required=True, help="The backfill fingerprints.json URI")
    parser.add_argument("--remeasured", required=True, help="The re-measured stale fingerprints.json URI")
    parser.add_argument("--pending", action="append", required=True, help="Source ids whose fingerprints are stale")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    payload = StoragePath(args.survey).read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.survey_sha256:
        raise ValueError("Survey input SHA-256 differs")
    document = json.loads(payload)
    fingerprints = document["fingerprints"]
    added = json.loads(StoragePath(args.fingerprints).read_text())
    remeasured = json.loads(StoragePath(args.remeasured).read_text())
    for source_id, fingerprint in added.items():
        if source_id in fingerprints:
            raise ValueError(f"The survey already carries the source: {source_id}")
        fingerprints[source_id] = fingerprint
    for source_id, fingerprint in remeasured.items():
        if source_id not in fingerprints:
            raise ValueError(f"The re-measured source is absent from the survey: {source_id}")
        fingerprints[source_id] = fingerprint
    for source_id in args.pending:
        if source_id not in fingerprints:
            raise ValueError(f"The pending source is absent from the survey: {source_id}")
        fingerprints[source_id] = None
    document["fingerprints"] = dict(sorted(fingerprints.items()))

    output = StoragePath(args.output)
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    name = f"source-identities-{digest}.json"
    (output / name).write_text(encoded)
    LOGGER.info("Sealed the composed survey: %s (%d entries)", name, len(document["fingerprints"]))
    print(json.dumps({"survey": str(output / name), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
