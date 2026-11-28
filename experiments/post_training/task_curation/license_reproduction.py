# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce one sealed branch license receipt through main's resolver.

The sealed receipt resolves a source's primary license evidence at an
immutable pin and derives its classification. This script runs main's
ported resolver on the same pinned upstream and compares the produced
receipt with the sealed one: the resolved pin, every evidence entry's
kind, license IDs, content hash and URL, and the derived classification
the dataset rules rule on.

```bash
uv run python -m experiments.post_training.task_curation.license_reproduction \\
    --source-id ChilleD:svamp \\
    --receipt sealed_license_receipt.json \\
    --output reproduction.json --run
```
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.licenses import check_license
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.license_resolution import resolve_license
from experiments.post_training.task_curation.license_targets import LicenseTarget

logger = logging.getLogger(__name__)

EVIDENCE_FIELDS = ("kind", "license_ids", "sha256", "url")


def upstream_target(receipt: dict) -> LicenseTarget:
    """The pinned upstream the sealed receipt's evidence resolves."""
    resolved = receipt["resolved_source"]
    return LicenseTarget(
        source_id=receipt["source_id"],
        dataset_id=resolved["dataset_id"],
        revision=resolved["revision"],
        url=resolved["url"],
    )


def compare(produced: dict, sealed: dict) -> dict:
    """Compare the produced receipt with the sealed one field by field.

    The sealed receipt binds a wrapper release whose recorded ``resolved_source``
    names the pinned upstream; main re-resolves that upstream, so the pin
    comparison runs against the sealed receipt's own resolved source.
    """
    resolved = sealed["resolved_source"]
    produced_evidence = [
        {field: evidence[field] for field in EVIDENCE_FIELDS} for evidence in produced["evidence"]
    ]
    sealed_evidence = [
        {field: evidence[field] for field in EVIDENCE_FIELDS} for evidence in sealed["evidence"]
    ]
    mismatches = [
        {"index": index, "field": field, "produced": produced_evidence[index].get(field), "sealed": sealed_evidence[index][field]}
        for index in range(min(len(produced_evidence), len(sealed_evidence)))
        for field in EVIDENCE_FIELDS
        if produced_evidence[index].get(field) != sealed_evidence[index][field]
    ]
    produced_receipt = _receipt(produced)
    sealed_receipt = _receipt(sealed)
    return {
        "pins": {
            "dataset_id": {"produced": produced["dataset_id"], "sealed_resolved": resolved["dataset_id"]},
            "revision": {"produced": produced["revision"], "sealed_resolved": resolved["revision"]},
        },
        "evidence_count": {"produced": len(produced_evidence), "sealed": len(sealed_evidence)},
        "per_evidence_mismatch_count": len(mismatches),
        "per_evidence_mismatches": mismatches[:12],
        "produced": {
            "classification": produced_receipt.classification.value,
            "license_ids": list(produced_receipt.license_ids),
            "bundled_third_party": produced["bundled_third_party"],
            "model_generated_outputs": produced["model_generated_outputs"],
            "check": check_license(produced_receipt).model_dump(mode="json"),
        },
        "sealed": {
            "classification": sealed_receipt.classification.value,
            "license_ids": list(sealed_receipt.license_ids),
            "bundled_third_party": sealed["bundled_third_party"],
            "model_generated_outputs": sealed["model_generated_outputs"],
            "check": check_license(sealed_receipt).model_dump(mode="json"),
        },
        "resolver_version": {"produced": produced["resolver_version"], "sealed": sealed["resolver_version"]},
    }


def _receipt(document: dict):
    from finestore.rl.licenses import LicenseReceipt

    return LicenseReceipt.model_validate(document)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--receipt", required=True, help="Sealed license receipt JSON for this source")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    sealed = json.loads(StoragePath(args.receipt).read_text())
    if sealed["source_id"] != args.source_id:
        raise ValueError(f"The receipt belongs to {sealed['source_id']}, not {args.source_id}")
    target = upstream_target(sealed)
    if not args.run:
        args.output.write_text(json.dumps({"source_id": args.source_id, "plan": "planned"}, indent=2) + "\n")
        return

    produced_receipt = resolve_license(target)
    produced = json.loads(produced_receipt.model_dump_json())
    comparison = compare(produced, sealed)
    comparison.update(
        {
            "source_id": args.source_id,
            "upstream": {"dataset_id": target.dataset_id, "revision": target.revision, "url": target.url},
            "reproduced": (
                comparison["pins"]["dataset_id"]["produced"] == comparison["pins"]["dataset_id"]["sealed_resolved"]
                and comparison["pins"]["revision"]["produced"] == comparison["pins"]["revision"]["sealed_resolved"]
                and comparison["per_evidence_mismatch_count"] == 0
                and comparison["produced"]["check"]["verdict"] == comparison["sealed"]["check"]["verdict"]
                and comparison["produced"]["classification"] == comparison["sealed"]["classification"]
                and comparison["resolver_version"]["produced"] == comparison["resolver_version"]["sealed"]
            ),
            "produced_at": datetime.now(UTC).isoformat(),
        }
    )
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    per_run = StoragePath(
        f"{args.artifact_root.rstrip('/')}/runs/{args.source_id.replace(':', '-')}/{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}/reproduction.json"
    )
    with per_run.open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    with StoragePath(f"{args.artifact_root.rstrip('/')}/reproduction.json").open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Reproduction result written to %s and %s: reproduced=%s",
        args.output,
        per_run,
        comparison["reproduced"],
    )


if __name__ == "__main__":
    main()
