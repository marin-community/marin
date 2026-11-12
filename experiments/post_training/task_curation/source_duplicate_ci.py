# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate changed-source fingerprints and prepare inputs for bounded overlap review."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import SourceFingerprint, SourceIdentitySurvey

EXPORT_DECLARATIONS = """import json
from experiments.post_training.task_curation.sources import all_sources
print(json.dumps({source.info.id: source.declaration.model_dump(mode='json')
                  for source in all_sources().values()}, sort_keys=True))
"""


class FingerprintReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    sha256: str
    declaration_sha256: str


def source_declarations(checkout: Path) -> dict:
    paths = [checkout, *sorted(checkout.glob("lib/*/src")), checkout / "infra/marina/src"]
    result = subprocess.check_output(
        [sys.executable, "-c", EXPORT_DECLARATIONS],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in paths)},
        text=True,
    )
    return json.loads(result)


def changed_source_declarations(before: dict, after: dict) -> set[str]:
    return {source_id for source_id in after if before.get(source_id) != after[source_id]}


def prepare_review(base_ref: str, index: Path, output: Path, changed_output: Path) -> set[str]:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    current = source_declarations(root)
    with tempfile.TemporaryDirectory(prefix="atlas-source-baseline-") as temporary:
        baseline = Path(temporary) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline), base_ref], check=True)
        try:
            previous = source_declarations(baseline)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], check=True)
    changed = changed_source_declarations(previous, current)
    references = {
        key: FingerprintReference.model_validate(value) for key, value in json.loads(index.read_text()).items()
    }
    missing = changed - references.keys()
    if missing:
        raise ValueError(f"Fingerprint every changed source before admission: {sorted(missing)}")
    fingerprints: dict[str, SourceFingerprint | None] = dict.fromkeys(current)
    artifacts = {}
    for source_id, reference in references.items():
        if source_id not in current:
            continue
        if reference.declaration_sha256 != canonical_sha256(current[source_id]):
            raise ValueError(f"Fingerprint declaration is stale: {source_id}")
        if reference.uri not in artifacts:
            payload = StoragePath(reference.uri).read_bytes()
            if hashlib.sha256(payload).hexdigest() != reference.sha256:
                raise ValueError(f"Fingerprint artifact digest differs: {reference.uri}")
            artifacts[reference.uri] = SourceIdentitySurvey.model_validate_json(payload)
        fingerprint = artifacts[reference.uri].fingerprints[source_id]
        if fingerprint is None:
            raise ValueError(f"Published fingerprint is unassessed: {source_id}")
        fingerprints[source_id] = fingerprint
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if changed and sum(value is not None for value in fingerprints.values()) < 2:
        raise ValueError("Duplicate review needs published reference-source fingerprints")
    cached = {
        (review.similarity.left, review.similarity.right): review
        for artifact in artifacts.values()
        for review in artifact.reviews
        if fingerprints.get(review.similarity.left) is not None and fingerprints.get(review.similarity.right) is not None
    }
    survey = SourceIdentitySurvey(
        catalog_root="ci:source-declarations",
        snapshot_id=head,
        generated_at=datetime.now(UTC).isoformat(),
        fingerprints=fingerprints,
        reviews=list(cached.values()),
        groups=[],
        judge=None,
    )
    output.write_text(survey.model_dump_json(indent=2) + "\n")
    changed_output.write_text(json.dumps(sorted(changed), indent=2) + "\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--changed-output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    changed = prepare_review(args.base_ref, args.index, args.output, args.changed_output)
    print(json.dumps({"changed_sources": len(changed), "source_ids": sorted(changed)}))


if __name__ == "__main__":
    main()
