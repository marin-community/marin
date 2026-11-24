# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend validation receipts for a release's targets.

One receipt per release, covering exactly the release's targets: the offline
capability census (:func:`backend_capabilities.build_survey`) over each target's
converted population plus the maintained live probes
(:func:`backend_live_probes.run_pull_probe` per remote backend, the local disk-pool
probe the local checker was actually built with), sealed write-once under the
release prefix. The release graph consumes it as the identity/backend compose's
``fresh_backend_receipt``; it is never a prior release's receipt reused.
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.backend_capabilities import build_survey
from experiments.post_training.task_curation.backend_live_probes import run_pull_probe
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

RECEIPT_SUFFIX = "backend-validation"
MANIFEST_KIND = "atlas-backend-census-inputs"
# The remote backends the live pull probe qualifies (the maintained probe refuses
# anything else).
REMOTE_BACKEND_NAMES = ("daytona", "iris")


@dataclass(frozen=True)
class ValidationInputs:
    """Typed inputs for one backend validation; every value is a graph output."""

    release_id: str
    source_revision: str
    snapshot_id: str
    catalog_root: str
    output_root: str
    targets: tuple[str, ...]
    manifest: ArtifactRef
    expected_task_count: int

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.source_revision or not self.snapshot_id:
            raise ValueError("release_id, source_revision and snapshot_id are required")
        if not self.catalog_root or not self.output_root:
            raise ValueError("catalog_root and output_root are required")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be a non-empty, duplicate-free tuple")
        if self.expected_task_count <= 0:
            raise ValueError(f"expected_task_count must be positive, got {self.expected_task_count}")


def read_manifest(inputs: ValidationInputs) -> dict:
    """Digest-verify the sealed census manifest and check its release bindings."""
    payload = StoragePath(inputs.manifest.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs.manifest.sha256:
        raise ValueError(f"census manifest digest mismatch: {inputs.manifest.uri}")
    manifest = json.loads(payload)
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"census manifest kind is {manifest.get('kind')!r}, expected {MANIFEST_KIND!r}")
    if manifest.get("source_revision") != inputs.source_revision:
        raise ValueError(
            f"census manifest pins revision {manifest.get('source_revision')!r}, expected {inputs.source_revision!r}"
        )
    if manifest.get("snapshot_id") != inputs.snapshot_id:
        raise ValueError(
            f"census manifest binds snapshot {manifest.get('snapshot_id')!r}, expected {inputs.snapshot_id!r}"
        )
    if manifest.get("catalog_root") != inputs.catalog_root:
        raise ValueError("census manifest binds a different catalog root than the release plan")
    covered = {canonical_id(entry["source_id"]) for entry in manifest["inputs"]}
    targets = {canonical_id(target) for target in inputs.targets}
    if covered != targets:
        raise ValueError(
            "the census manifest must cover exactly the release targets:"
            f" missing {sorted(targets - covered)}, extra {sorted(covered - targets)}"
        )
    return manifest


def canonical_id(source_id: str) -> str:
    """The canonical registry id (keeps the manifest and targets comparable)."""
    from infra.marina.apps.rl_data_catalog.server.source_identity import (  # noqa: PLC0415 — the app import stays beside its only use
        canonical_source_id,
    )

    return canonical_source_id(source_id)


def run_validation(
    inputs: ValidationInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    survey_builder=None,
    pull_probe=None,
    disk_pool_root: Path | None = None,
    revision_command: tuple[str, ...] = ("git", "rev-parse", "HEAD"),
    feeds: tuple | None = None,
) -> dict:
    """Assemble the validation receipt; ``dry_run`` verifies with zero writes.

    ``survey_builder`` defaults to the maintained offline census over the manifest
    with the production checkers; ``pull_probe`` defaults to the maintained live
    pull probe per remote backend. Both are injectable so tests pin the contract
    without machines. The releasing checkout must be clean and at the expected
    revision; ``repo_root`` is the checkout the revision command runs in.
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "backend_validation", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        manifest = read_manifest(inputs)
        revision = subprocess.check_output(revision_command, text=True, cwd=repo_root).strip()
        if revision != inputs.source_revision:
            raise ValueError(f"checkout HEAD {revision} is not the expected release revision {inputs.source_revision}")
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "snapshot_id": inputs.snapshot_id,
            "targets": list(inputs.targets),
            "manifest": {"uri": inputs.manifest.uri, "sha256": inputs.manifest.sha256},
            "expected_task_count": inputs.expected_task_count,
            "dry_run": dry_run,
            "sql_writes": 0,
            "provider_calls": 0,
            "model_requests": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "backend_validation",
                {"mode": mode, "targets": len(inputs.targets)},
                alert_feeds,
            )
            return proof
        builder = survey_builder if survey_builder is not None else _default_survey_builder
        survey = builder(manifest, revision, disk_pool_root)
        covered = {canonical_id(source.source_id) for source in survey.sources}
        measured = sum(source.task_count for source in survey.sources)
        if covered != {canonical_id(target) for target in inputs.targets}:
            raise ValueError(f"the built survey covers {sorted(covered)}, expected exactly the release targets")
        if measured != inputs.expected_task_count:
            raise ValueError(f"the survey measures {measured} tasks, expected {inputs.expected_task_count}")
        probe = pull_probe if pull_probe is not None else run_pull_probe
        pull_records = [_probe_record(probe(backend)) for backend in REMOTE_BACKEND_NAMES]
        configure_storage()
        survey_ref = seal(f"{inputs.output_root}/backend-survey.json", _survey_document(survey))
        receipt = {
            **proof,
            "survey": {"uri": survey_ref.uri, "sha256": survey_ref.sha256},
            "sources": [source.source_id for source in survey.sources],
            "task_count": measured,
            "pull_probes": pull_records,
            "survey_builder": type(survey_builder).__name__ if survey_builder is not None else "offline-census",
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        receipt_ref = seal(f"{inputs.output_root}/backend-validation-receipt.json", receipt)
        receipt["receipt"] = {"uri": receipt_ref.uri, "sha256": receipt_ref.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "backend_validation",
            {"mode": mode, "task_count": measured, **receipt["receipt"]},
            alert_feeds,
        )
        return receipt
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id, "backend_validation", {"error": str(error)[:500], "mode": mode}, alert_feeds
        )
        raise


def _survey_document(survey) -> dict:
    """The built survey as receipt JSON, whatever shape the census builder returned."""
    return json.loads(survey.model_dump_json()) if hasattr(survey, "model_dump_json") else dataclasses.asdict(survey)


def _probe_record(record) -> dict:
    """The pull-probe record as receipt JSON, whatever shape the runner returned."""
    return dataclasses.asdict(record) if dataclasses.is_dataclass(record) else dict(record)


def _default_survey_builder(manifest: dict, revision: str, disk_pool_root: Path | None):
    """The maintained offline census over the manifest with the production checkers."""
    from shellbox.survey import production_checkers  # noqa: PLC0415 — the survey stack stays beside its only use

    from experiments.post_training.task_curation.backend_capabilities import LocalDiskPool  # noqa: PLC0415

    pool = LocalDiskPool(disk_pool_root) if disk_pool_root is not None else None
    return build_survey(manifest, production_checkers(local_disk_pool=pool), revision)


def configure_storage() -> None:
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415 — storage config beside its only use

    configure_coreweave_s3()


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON ValidationInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    parser.add_argument(
        "--local-disk-root",
        type=Path,
        help="Probe ext4 loop/overlay enforcement on this privileged worker; the local checker "
        "records its disk-pool evidence from this pool",
    )
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["manifest"] = ArtifactRef(**document["manifest"])
    inputs = ValidationInputs(**document)
    repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    receipt = run_validation(
        inputs,
        dry_run=args.mode == "dry-run",
        repo_root=repo_root,
        disk_pool_root=args.local_disk_root,
    )
    label = "BACKEND-VALIDATION-DRY-RUN" if args.mode == "dry-run" else "BACKEND-VALIDATION-SEALED"
    print(f"{label} {json.dumps(receipt)}", flush=True)


if __name__ == "__main__":
    main()
