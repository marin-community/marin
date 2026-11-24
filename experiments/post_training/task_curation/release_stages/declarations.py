# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarations: extract and seal the release's canonical declaration set.

One maintained module replaces the per-release declaration-extraction scripts (see
INVENTORY.md). The release's declarations come from the current ``all_sources()``
layer at the landed revision; non-target declarations must be fingerprint-identical
to the prior release's sealed set, and every source must pass the catalog admission
gate. All release-identifying values arrive as typed
:class:`DeclarationInputs`; the dry run verifies everything with zero writes.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.sources import all_sources
from infra.marina.apps.rl_data_catalog.server.admission import require_catalog_declaration
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

DECLARATIONS_SUFFIX = "declarations"


@dataclass(frozen=True)
class DeclarationInputs:
    """Typed inputs for one declaration extraction; every value is a graph output."""

    release_id: str
    source_revision: str
    output_root: str
    expected_source_count: int
    targets: tuple[str, ...]
    prior_declarations: ArtifactRef
    removed_sources: tuple[str, ...] = ()

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if self.expected_source_count <= 0 or not self.targets:
            raise ValueError("expected_source_count and targets must be non-empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets repeat a source id")


def read_prior_declarations(prior: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the prior release's sealed declarations document."""
    payload = StoragePath(prior.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != prior.sha256:
        raise ValueError(f"prior declarations digest mismatch: {prior.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("prior declarations document is empty or malformed")
    return document


def current_declarations(sources_fn=all_sources) -> dict[str, dict]:
    """The current canonical declarations, each passing the admission gate."""
    sources = sources_fn()
    current: dict[str, dict] = {}
    for source in sources.values():
        require_catalog_declaration(source.declaration)
        current[canonical_source_id(source.info.id)] = canonical_declaration(source.declaration).model_dump(mode="json")
    return current


def verify_repo_state(repo_root: Path, inputs: DeclarationInputs) -> str:
    """The extracting checkout must be clean and contain the plan's pinned revision.

    ``source_revision`` pins the landed registry state the plan was assembled from;
    the checkout may sit at that commit or any descendant (pipeline fixes land after
    the plan), because the checkout's HEAD sha cannot appear inside its own plan
    document. Source drift between the pin and the checkout is what the declaration
    fingerprint checks refuse, not this shape check.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"extracting checkout is dirty:\n{dirty[:400]}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    pinned = subprocess.run(
        ["git", "merge-base", "--is-ancestor", inputs.source_revision, revision],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(
            f"checkout HEAD {revision} does not contain the plan's pinned revision {inputs.source_revision}; "
            "cut from a checkout that contains it"
        )
    return revision


def verify_declarations(inputs: DeclarationInputs, current: dict[str, dict], prior: dict[str, dict]) -> list[str]:
    """Count, admission, target and drift checks; returns the sorted declaration ids."""
    if len(current) != inputs.expected_source_count:
        raise ValueError(f"{len(current)} current declarations, expected {inputs.expected_source_count}")
    missing = [target for target in inputs.targets if target not in current]
    if missing:
        raise ValueError(f"current declarations lack the release targets {sorted(missing)}")
    removed = sorted(source_id for source_id in prior if source_id not in current)
    unplanned = sorted(set(removed) - set(inputs.removed_sources))
    if unplanned:
        raise ValueError(f"prior declarations absent from the current set: {unplanned[:5]}")
    planned_but_present = sorted(set(inputs.removed_sources) & set(current))
    if planned_but_present:
        raise ValueError(f"the plan removes declarations the current set still carries: {planned_but_present[:5]}")
    drift = [
        source_id
        for source_id, row in prior.items()
        if source_id in current
        and source_id not in inputs.targets
        and fingerprint_input(row) != fingerprint_input(current[source_id])
    ]
    if drift:
        raise ValueError(f"non-target declarations drifted from the prior release: {sorted(drift)[:5]}")
    return sorted(current)


def run_declarations(
    inputs: DeclarationInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    sources_fn=all_sources,
    feeds: tuple | None = None,
) -> dict:
    """Extract the release declarations; ``dry_run`` verifies with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "declarations", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        verify_repo_state(repo_root, inputs)
        current = current_declarations(sources_fn)
        prior = read_prior_declarations(inputs.prior_declarations)
        unknown_targets = [target for target in inputs.targets if target not in prior and target not in current]
        if unknown_targets:
            raise ValueError(f"targets unknown to the prior and current declaration sets: {sorted(unknown_targets)}")
        source_ids = verify_declarations(inputs, current, prior)
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{DECLARATIONS_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "prior_declarations": {"uri": inputs.prior_declarations.uri, "sha256": inputs.prior_declarations.sha256},
            "source_count": len(source_ids),
            "targets": list(inputs.targets),
            "prior_source_count": len(prior),
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id, "declarations", {"mode": mode, "source_count": len(source_ids)}, alert_feeds
            )
            return proof
        document = {source_id: current[source_id] for source_id in source_ids}
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        path = StoragePath(f"{inputs.output_root}/declarations.json")
        if path.exists():
            raise ValueError(
                f"declarations document already exists at {path}; bump the release prefix instead of overwriting"
            )
        path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("declarations document readback mismatch")
        proof["declarations"] = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        receipt_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
        receipt_path = StoragePath(f"{inputs.output_root}/declarations-receipt.json")
        if receipt_path.exists():
            raise ValueError(f"declarations receipt already exists at {receipt_path}")
        receipt_path.write_bytes(receipt_raw, auto_mkdir=True)
        if receipt_path.read_bytes() != receipt_raw:
            raise ValueError("declarations receipt readback mismatch")
        proof["receipt"] = {"uri": str(receipt_path), "sha256": hashlib.sha256(receipt_raw).hexdigest()}
        alerts.stage_finished(
            inputs.release_id,
            "declarations",
            {"mode": mode, "source_count": len(source_ids), **proof["declarations"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "declarations", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON DeclarationInputs document")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout the declarations come from")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["prior_declarations"] = ArtifactRef(**document["prior_declarations"])
    inputs = DeclarationInputs(**document)
    proof = run_declarations(inputs, dry_run=args.mode == "dry-run", repo_root=args.repo_root)
    label = "DECLARATIONS-DRY-RUN" if args.mode == "dry-run" else "DECLARATIONS-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
