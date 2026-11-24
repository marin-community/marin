# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fresh-QC-index stage: the graph's producer of the release's staged QC panel index.

The release's fresh QC index is a graph output bound to the release's target load and
snapshot (ruling 2026-11-24), not a hand-produced pin. One panel entry per browse-staged
source: the maintained
(:mod:`infra.marina.apps.rl_data_catalog.load_reviews`) stager resolves the frozen
declaration's selected review evidence into the replayable staged document; this stage
compresses it byte-deterministically under the release prefix, seals the per-panel
receipt row, and seals the index the catalog load replays.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
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
    declarations: ArtifactRef
    qc_completion: ArtifactRef

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.target_load_id or not self.snapshot_id:
            raise ValueError("release_id, target_load_id and snapshot_id are required")
        if not self.source_revision or not self.output_root:
            raise ValueError("source_revision and output_root are required")


def _read_declarations(ref: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the frozen declarations document the panels bind."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"declarations digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("declarations document is empty or malformed")
    return document


def _completion_rows(ref: ArtifactRef) -> list[dict]:
    """The browse completion's staged panels."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"browse completion digest mismatch: {ref.uri}")
    completion = json.loads(payload)
    if completion.get("status") != "PASS":
        raise ValueError(f"the browse completion is {completion.get('status')!r}; the index needs a PASS")
    rows = completion.get("results") or []
    if not rows:
        raise ValueError("the browse completion carries no staged panels")
    return rows


def _stage_panel(inputs: FreshQcIndexInputs, row: dict, declarations: dict[str, dict], *, write: bool) -> dict:
    """Resolve, verify and (in write mode) stage one panel document under the release prefix."""
    from infra.marina.apps.rl_data_catalog import load_reviews  # noqa: PLC0415 — the maintained stager
    from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference  # noqa: PLC0415

    source_id = row["source_id"]
    declaration = declarations.get(source_id)
    if declaration is None:
        raise ValueError(f"no frozen declaration for staged panel source: {source_id}")
    reference = ManifestReference.model_validate(row["original_receipt"]["artifact"])
    staged = load_reviews.stage_content_reviews(reference, declaration, 8, None)
    if staged["receipt"]["artifact"] != row["original_receipt"]["artifact"]:
        raise ValueError(f"the staged review receipt differs from the browse copy: {source_id}")
    payload = gzip.compress(
        json.dumps(staged, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), mtime=0
    )
    identity = reference.sha256
    reference_out = {"uri": row["original_receipt"]["artifact"]["uri"], "sha256": identity}
    if write:
        path = StoragePath(f"{inputs.output_root}/{STAGED_DIR}/{identity}.json.gz")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"staged panel {identity} exists with different content; use a new output root")
        else:
            path.write_bytes(payload, auto_mkdir=True)
        if path.read_bytes() != payload:
            raise ValueError(f"staged panel readback mismatch: {identity}")
        reference_out = {"uri": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
    return {
        "source_id": source_id,
        "identity": identity,
        "status": "complete",
        "records": len(staged["items"]),
        "bytes": len(payload),
        "reference": reference_out,
    }


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
        declarations = _read_declarations(inputs.declarations)
        rows = _completion_rows(inputs.qc_completion)
        index_rows = [_stage_panel(inputs, row, declarations, write=not dry_run) for row in rows]
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
            "basis": (
                "The browse-staged SAMPLE QC panels at their frozen declarations' selected review evidence, "
                f"staged by this release at load {inputs.target_load_id}"
            ),
            "receipts": index_rows,
            "provenance": {
                "browse_completion": {"uri": inputs.qc_completion.uri, "sha256": inputs.qc_completion.sha256},
                "declarations": {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256},
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
