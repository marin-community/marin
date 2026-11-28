# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The main-grader-drift stage: the graph's typed advisory queue of stale stored graders.

A train publication sealed at ``converter_revision`` stores the grader and
verifier files its recipe produced; when main later fixes those recipe files,
the stored grader predates the fix and the branch's own
``pipeline_reference == original_pipeline`` checks cannot see it (they compare a
release only with its own branch's declaration, and the branch has not merged
main). This stage sits early, before compose, and compares the sealed
declarations' recipe files — measured by the plan derivation at the freeze into
the plan's ``grader_drift_reference`` — against the derivation-time origin/main
blob hashes. Every Available or Pending train source whose stored grader
predates main's fixes gets one typed finding, ``grader_stale_vs_main``, with
the main fix commits and PRs that touch the changed files.

The stage is advisory by ruling (2026-11-26 6:45 AM EST): grader fixes must not
block training. It never raises on drift — it queues each stale source for
re-seal by sealing the findings document the catalog publication and the
source-status extras carry — and fails closed only when its inputs do not bind:
the pinned reference must measure exactly the sealed declarations document.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.grader_drift import (
    DRIFT_FINDING,
    GraderDriftReference,
    compare_recipe,
    is_drift_population,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

DRIFT_KIND_SUFFIX = "grader-drift"


@dataclass(frozen=True)
class GraderDriftInputs:
    """Typed inputs for the drift stage; every value is a graph output or a plan pin."""

    release_id: str
    source_revision: str
    declarations: ArtifactRef
    reference: GraderDriftReference
    output_root: str

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if not self.reference.sources:
            raise ValueError("the grader-drift reference pins no sources")


def _read_declarations(ref: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the frozen declarations document the reference must bind."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"declarations digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("declarations document is empty or malformed")
    return document


def _verify_reference_binds(inputs: GraderDriftInputs, declared: dict[str, dict]) -> None:
    """Fail closed unless the pinned reference measured exactly this declarations document.

    The derivation measured the sealed recipe side at the freeze; the graph seals
    the declarations document independently. A pin that disagrees with the sealed
    document — a source, converter revision, converter version or file set that
    differs — means the measurement and the graph are not looking at the same
    freeze, and a drift verdict built on it would be unanchored.
    """
    pinned_ids = set(inputs.reference.sources)
    if pinned_ids != set(declared):
        raise ValueError(
            "the grader-drift reference and the sealed declarations disagree on the source set"
            f" (pinned-only {sorted(pinned_ids - set(declared))[:5]},"
            f" sealed-only {sorted(set(declared) - pinned_ids)[:5]})"
        )
    for source_id, pin in inputs.reference.sources.items():
        declaration = declared[source_id]
        binding = declaration.get("actor_census_input") or {}
        if pin.converter_revision != binding.get("converter_revision"):
            raise ValueError(
                f"grader-drift pin for {source_id} measures converter revision"
                f" {pin.converter_revision!r}, the sealed declaration binds"
                f" {binding.get('converter_revision')!r}"
            )
        if pin.converter_version != binding.get("converter_version"):
            raise ValueError(
                f"grader-drift pin for {source_id} measures converter version"
                f" {pin.converter_version!r}, the sealed declaration binds"
                f" {binding.get('converter_version')!r}"
            )
        sealed_paths = sorted((declaration.get("pipeline") or {}).get("files") or [])
        if pin.converter_revision is not None and sorted(pin.files) != sealed_paths:
            raise ValueError(
                f"grader-drift pin for {source_id} measures the file set {sorted(pin.files)},"
                f" the sealed declaration declares {sealed_paths}"
            )


def run_main_grader_drift(inputs: GraderDriftInputs, *, dry_run: bool, feeds=None) -> dict:
    """Compare the sealed recipes with main and seal the advisory per-source findings."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "main_grader_drift", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        declared = _read_declarations(inputs.declarations)
        _verify_reference_binds(inputs, declared)
        findings = []
        clean = []
        skipped = []
        for source_id, declaration in sorted(declared.items()):
            if not is_drift_population(declaration):
                skipped.append(source_id)
                continue
            sealed = inputs.reference.sources[source_id]
            finding = compare_recipe(source_id, sealed, inputs.reference.main)
            if finding is None:
                clean.append(source_id)
            else:
                findings.append(finding.model_dump(mode="json"))
        document = {
            "kind": f"atlas-{inputs.release_id}-{DRIFT_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "main_revision": inputs.reference.main.revision,
            "main_merge_base": inputs.reference.main.merge_base,
            "finding_type": DRIFT_FINDING,
            "advisory": True,
            "findings": findings,
            "clean_sources": clean,
            "skipped_sources": skipped,
        }
        proof: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{DRIFT_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "main_revision": inputs.reference.main.revision,
            "stale_sources": len(findings),
            "clean_sources": len(clean),
            "skipped_sources": len(skipped),
            "grader_drift": document["findings"],
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "main_grader_drift",
                {"mode": mode, "stale": len(findings), "clean": len(clean)},
                alert_feeds,
            )
            return proof
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        document_raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        document_path = StoragePath(f"{inputs.output_root}/grader-drift.json")
        if document_path.exists():
            raise ValueError(f"grader-drift document already exists at {document_path}")
        document_path.write_bytes(document_raw, auto_mkdir=True)
        if document_path.read_bytes() != document_raw:
            raise ValueError("grader-drift document readback mismatch")
        receipt_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
        receipt_path = StoragePath(f"{inputs.output_root}/grader-drift-receipt.json")
        if receipt_path.exists():
            raise ValueError(f"grader-drift receipt already exists at {receipt_path}")
        receipt_path.write_bytes(receipt_raw, auto_mkdir=True)
        if receipt_path.read_bytes() != receipt_raw:
            raise ValueError("grader-drift receipt readback mismatch")
        proof["receipt"] = {"uri": str(receipt_path), "sha256": hashlib.sha256(receipt_raw).hexdigest()}
        proof["document"] = {"uri": str(document_path), "sha256": hashlib.sha256(document_raw).hexdigest()}
        alerts.stage_finished(
            inputs.release_id,
            "main_grader_drift",
            {"mode": mode, "stale": len(findings), "clean": len(clean)},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id, "main_grader_drift", {"mode": mode, "error": str(error)[:500]}, alert_feeds
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="A JSON GraderDriftInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["declarations"] = ArtifactRef(**document["declarations"])
    document["reference"] = GraderDriftReference.model_validate(document["reference"])
    inputs = GraderDriftInputs(**document)
    proof = run_main_grader_drift(inputs, dry_run=args.mode == "dry-run")
    print(f"GRADER-DRIFT-{'DRY-RUN' if args.mode == 'dry-run' else 'SEALED'} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
