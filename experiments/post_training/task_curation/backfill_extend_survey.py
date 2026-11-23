# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the settled identity survey with the backfilled variants' fingerprints.

Issue #11976 step 3: the neighbor survey's focused remeasure requires the new sources'
fingerprints inside the survey payload. Fetches the settled full survey the parents were
measured in, merges the variants' computed fingerprints, and seals the extended survey at
a fresh output root with its SHA-256 for the survey run's pinned input.
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
    parser.add_argument("--survey", required=True, help="The settled identity survey document URI")
    parser.add_argument("--survey-sha256", required=True)
    parser.add_argument("--fingerprints", required=True, help="The backfill fingerprints.json URI")
    parser.add_argument("--output", required=True, help="The extended survey's output prefix")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    payload = StoragePath(args.survey).read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.survey_sha256:
        raise ValueError("Survey input SHA-256 differs")
    document = json.loads(payload)
    fingerprints = json.loads(StoragePath(args.fingerprints).read_text())
    survey_fingerprints = document["fingerprints"]
    for source_id, fingerprint in fingerprints.items():
        if source_id in survey_fingerprints:
            raise ValueError(f"The survey already carries the source: {source_id}")
        survey_fingerprints[source_id] = fingerprint
    document["fingerprints"] = dict(sorted(survey_fingerprints.items()))
    document["generated_at"] = document.get("generated_at")

    output = StoragePath(args.output)
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    name = f"source-identities-{digest}.json"
    (output / name).write_text(encoded)
    LOGGER.info("Sealed the extended survey: %s (%d fingerprints)", name, len(document["fingerprints"]))
    print(
        json.dumps(
            {"survey": str(output / name), "sha256": digest, "fingerprints": len(document["fingerprints"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
