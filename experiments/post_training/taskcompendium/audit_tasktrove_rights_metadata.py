# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit TaskTrove candidate proofs, ordered tags, and source rights metadata."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.taskcompendium.export_tasktrove_accepted import export_accepted_records


def audit_rights_metadata(
    ingestion_uri: str,
    audit_uri: str,
    output_dir: Path,
    *,
    builder_revision: str,
) -> dict[str, Any]:
    """Verify an ingestion's metadata joins and inventory terms without clearing rows."""
    configure_coreweave_s3()
    ingestion_prefix = StoragePath(ingestion_uri)
    ingestion_manifest = json.loads((ingestion_prefix / "ingestion-manifest.json").read_bytes())
    audit_prefix = StoragePath(audit_uri)
    report = export_accepted_records(
        ingestion_prefix / "ingestion-ledger.parquet",
        ingestion_prefix / "private-catalog.parquet",
        ingestion_prefix / "public-candidates.jsonl",
        ingestion_prefix / "candidate-proof.jsonl",
        audit_prefix,
        ingestion_manifest=ingestion_manifest,
        builder_revision=builder_revision,
        clearance_audit_manifest_sha256=None,
    )
    audit_manifest = audit_prefix / "manifest.json"
    audit_manifest_bytes = audit_manifest.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": report["status"],
        "format": "TaskTrove-private-rights-metadata-audit-v1",
        "builder_revision": builder_revision,
        "ingestion_manifest_uri": ingestion_manifest["manifest_uri"],
        "ingestion_source_manifest_sha256": ingestion_manifest["source_manifest_sha256"],
        "audit_manifest_uri": str(audit_manifest),
        "audit_manifest_sha256": hashlib.sha256(audit_manifest_bytes).hexdigest(),
        "input_rows_total": report["input_rows_total"],
        "input_rows_by_disposition": report["input_rows_by_disposition"],
        "rights_term_group_count": len(report["rights_terms_by_source_mode_family_converter"]),
        "rights_rejected_rows_by_source_reason": report["rights_rejected_rows_by_source_reason"],
        "ledger_sha256": report["input_ledger_sha256"],
        "catalog_sha256": report["private_catalog_sha256"],
        "candidate_sha256": report["input_candidate_sha256"],
        "proof_sha256": report["input_proof_sha256"],
        "candidate_sha256_verified": report["candidate_sha256_verified"],
        "proof_sha256_verified": report["proof_sha256_verified"],
        "archive_payload_bytes_read": report["archive_payload_bytes_read"],
        "private_task_specifications_loaded": report["private_task_specifications_loaded"],
        "accepted_public_records": sum(item["accepted_public_records"] for item in report["cohort_counts"].values()),
        "public_projection_published": False,
    }
    (output_dir / "rights-audit-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    audit_rights_metadata(
        os.environ["TASKTROVE_OUTPUT_URI"],
        os.environ["TASKTROVE_RIGHTS_AUDIT_OUTPUT_URI"],
        output_dir,
        builder_revision=os.environ["TASKTROVE_AUDIT_REVISION"],
    )


if __name__ == "__main__":
    main()
