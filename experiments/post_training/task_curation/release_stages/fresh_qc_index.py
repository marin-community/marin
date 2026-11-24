# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fresh-QC-index stage: the graph's producer of the release's staged QC panel index.

The release's fresh QC index is a graph output bound to the release's target load and
snapshot (ruling 2026-11-24), not a hand-produced pin. The panels are the prior
release's current-pin SAMPLE QC panels; the browse stage copied their review tables at
the target load, and this stage re-stages each panel document byte-identically under
the release prefix (digest-verified against the prior index) and seals the index the
catalog load replays. Every panel must have been staged by the browse completion
before it enters the index.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.catalog_load import QC_PANEL_RECORDS
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

INDEX_KIND_SUFFIX = "fresh-qc-index"
STAGED_DIR = "staged-v1"


@dataclass(frozen=True)
class FreshQcIndexInputs:
    """Typed inputs for the release's fresh QC index; every value is a graph output."""

    release_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    output_root: str
    prior_index: ArtifactRef
    qc_completion: ArtifactRef

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.target_load_id or not self.snapshot_id:
            raise ValueError("release_id, target_load_id and snapshot_id are required")
        if not self.source_revision or not self.output_root:
            raise ValueError("source_revision and output_root are required")


def _read_index(ref: ArtifactRef, expected_load_id: str | None) -> dict:
    """Digest-verify and read a fresh-QC-index document."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"fresh QC index digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if not isinstance(document.get("receipts"), list) or not document["receipts"]:
        raise ValueError(f"fresh QC index at {ref.uri} carries no receipts")
    if expected_load_id is not None and document.get("load_id") != expected_load_id:
        raise ValueError(f"fresh QC index at {ref.uri} names load {document.get('load_id')!r}")
    return document


def _completion_panels(ref: ArtifactRef) -> dict[str, dict]:
    """The browse completion's staged panels by source id."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"browse completion digest mismatch: {ref.uri}")
    completion = json.loads(payload)
    if completion.get("status") != "PASS":
        raise ValueError(f"the browse completion is {completion.get('status')!r}; the index needs a PASS")
    panels = {}
    for row in completion.get("results") or []:
        source_id = row.get("source_id")
        if not source_id:
            raise ValueError("the browse completion carries a result without a source_id")
        panels[source_id] = row
    if not panels:
        raise ValueError("the browse completion carries no staged panels")
    return panels


def restage_panel(inputs: FreshQcIndexInputs, row: dict, *, write: bool) -> dict:
    """Verify one prior panel document; ``write`` re-stages it byte-identically under the release prefix."""
    raw = StoragePath(row["reference"]["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["reference"]["sha256"]:
        raise ValueError(f"prior panel digest mismatch: {row['source_id']}")
    if row.get("status") != "complete" or row.get("records") != QC_PANEL_RECORDS:
        raise ValueError(f"prior panel {row['source_id']} is {row.get('status')!r} with {row.get('records')} records")
    reference = {"uri": row["reference"]["uri"], "sha256": row["reference"]["sha256"]}
    if write:
        target = StoragePath(f"{inputs.output_root}/{STAGED_DIR}/{row['identity']}.json.gz")
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"staged panel {row['identity']} exists with different content; use a new output root")
        else:
            target.write_bytes(raw, auto_mkdir=True)
        if target.read_bytes() != raw:
            raise ValueError(f"staged panel readback mismatch: {row['identity']}")
        reference = {"uri": str(target), "sha256": row["reference"]["sha256"]}
    return {
        "source_id": row["source_id"],
        "status": row["status"],
        "records": row["records"],
        "bytes": len(raw),
        "identity": row["identity"],
        "reference": reference,
    }


def _basis(prior: dict, inputs: FreshQcIndexInputs) -> str:
    """The index's factual basis: the prior panels, re-staged for this release's load."""
    prior_basis = str(prior.get("basis") or "the prior release's staged SAMPLE QC panels").rstrip(".")
    return f"{prior_basis}; review tables re-copied at load {inputs.target_load_id} by the {inputs.release_id} browse"


def run_fresh_qc_index(
    inputs: FreshQcIndexInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
) -> dict:
    """Seal the release's fresh QC index; ``dry_run`` verifies every panel with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "fresh_qc_index", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        prior = _read_index(inputs.prior_index, expected_load_id=None)
        panels = _completion_panels(inputs.qc_completion)
        index_rows = [restage_panel(inputs, row, write=not dry_run) for row in prior["receipts"]]
        index_sources = {row["source_id"] for row in index_rows}
        if index_sources != set(panels):
            raise ValueError(
                f"the prior index panels {sorted(index_sources - set(panels))} and the browse completion "
                f"panels {sorted(set(panels) - index_sources)} disagree"
            )
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "fresh_qc_index",
                {"mode": mode, "panels": len(index_rows)},
                alert_feeds,
            )
            return {
                "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
                "release_id": inputs.release_id,
                "load_id": inputs.target_load_id,
                "snapshot_id": inputs.snapshot_id,
                "panels": len(index_rows),
                "dry_run": True,
                "sql_writes": 0,
            }
        document = {
            "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "source_revision": inputs.source_revision,
            "basis": _basis(prior, inputs),
            "receipts": index_rows,
            "provenance": {
                "prior_index": {"uri": inputs.prior_index.uri, "sha256": inputs.prior_index.sha256},
                "browse_completion": {"uri": inputs.qc_completion.uri, "sha256": inputs.qc_completion.sha256},
            },
        }
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        path = StoragePath(f"{inputs.output_root}/fresh-qc-index.json")
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"fresh QC index at {path} exists with different content; use a new output root")
        else:
            path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("fresh QC index readback mismatch")
        reference = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        alerts.stage_finished(
            inputs.release_id,
            "fresh_qc_index",
            {"mode": mode, "panels": len(index_rows), **reference},
            alert_feeds,
        )
        return {
            "kind": document["kind"],
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "panels": len(index_rows),
            "index": reference,
            "sql_writes": 0,
        }
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "fresh_qc_index", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON FreshQcIndexInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    inputs = FreshQcIndexInputs(**json.loads(args.inputs.read_text()))
    proof = run_fresh_qc_index(inputs, dry_run=args.mode == "dry-run")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
