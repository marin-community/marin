# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package preview: stage the preview-only catalog package.

One maintained module replaces ``bundle_preview_dependencies.py`` and the
per-release prepare calls (see INVENTORY.md). The package bytes come from the
maintained ``prepare_preview`` — this stage is the typed envelope around it: the
extracting checkout must be clean at the release revision, the sealed
oracle-inventory and source-count inputs are digest-verified, the package path
must sit outside the checkout and must not already exist (mirroring
``prepare_preview``'s own refusal semantics), and the staged provenance is
recorded on the sealed package receipt.
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
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.prepare_preview import prepare_preview

PACKAGE_DIR_NAME = "preview-package"
PACKAGE_RECEIPT_NAME = "preview-package-receipt.json"
STAGED_INPUTS_DIR_NAME = "staged-inputs"
PROVENANCE_MEMBERS = ("dist", "bundled-model-provenance.json")
REMOTE_SCHEMES = ("gs://", "s3://", "mirror://", "memory://")


@dataclass(frozen=True)
class PackagePreviewInputs:
    """Typed inputs for one preview packaging run; every value is a graph output."""

    release_id: str
    source_revision: str
    output_root: str
    oracle_inventory: ArtifactRef | None = None
    source_counts: ArtifactRef | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")


def receipt_kind(release_id: str) -> str:
    """The sealed package-receipt kind for one release."""
    return f"atlas-{release_id}-preview-package"


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


def verify_repo_state(repo_root: Path, revision: str) -> str:
    """The packaging checkout must be clean and contain the plan's pinned revision.

    In the in-region bundle mode the submitter attests the workspace (the guard's
    landing-log check carries the landed attestation) and the git checks cannot run.
    """
    from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — avoids the guard/package cycle at module load
        attested_revision,
    )

    attested = attested_revision()
    if attested is not None:
        return attested
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"packaging checkout is dirty:\n{dirty[:400]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    pinned = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, head],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(
            f"checkout HEAD {head} does not contain the plan's pinned revision {revision}; "
            "cut from a checkout that contains it"
        )
    return head


def planned_package_root(output_root: str, repo_root: Path) -> Path:
    """The package directory: under the output root and strictly outside the checkout."""
    if any(output_root.startswith(scheme) for scheme in REMOTE_SCHEMES):
        raise ValueError(f"preview packaging writes to local disk; output_root must be a local path, got {output_root}")
    root = Path(output_root) / PACKAGE_DIR_NAME
    if root.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("Stage the preview outside the source checkout")
    return root


def staged_input(ref: ArtifactRef | None, name: str, staging_dir: Path) -> Path | None:
    """Copy one sealed input's digest-verified bytes to a local file for ``prepare_preview``.

    Plans pin these inputs at object-store URIs while ``prepare_preview`` reads local
    files, so every input is staged beside the package, never read in place.
    """
    if ref is None:
        return None
    raw = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise ValueError(f"sealed reference digest mismatch: {ref.uri}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / f"{name}.json"
    path.write_bytes(raw)
    return path


def _ref_pointer(ref: ArtifactRef | None) -> dict | None:
    return None if ref is None else {"uri": ref.uri, "sha256": ref.sha256}


def _packaged_provenance(package_root: Path) -> dict | None:
    provenance = package_root.joinpath(*PROVENANCE_MEMBERS)
    if not provenance.is_file():
        return None
    return json.loads(provenance.read_text(encoding="utf-8"))


def run_package_preview(
    inputs: PackagePreviewInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    feeds: tuple | None = None,
) -> dict:
    """Stage the preview package; ``dry_run`` verifies with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "package_preview", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        package_root = planned_package_root(inputs.output_root, repo_root)
        verify_repo_state(repo_root, inputs.source_revision)
        for _, ref in (("oracle inventory", inputs.oracle_inventory), ("source counts", inputs.source_counts)):
            if ref is not None:
                read_ref(ref)
        if package_root.exists():
            raise ValueError(
                f"preview package already exists at {package_root}; bump the release prefix instead of overwriting"
            )
        proof: dict = {
            "kind": receipt_kind(inputs.release_id),
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "package": {"uri": str(package_root)},
            "oracle_inventory": _ref_pointer(inputs.oracle_inventory),
            "source_counts": _ref_pointer(inputs.source_counts),
            "packaged": None,
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id, "package_preview", {"mode": mode, "package": str(package_root)}, alert_feeds
            )
            return proof
        staging_dir = Path(inputs.output_root) / STAGED_INPUTS_DIR_NAME
        prepare_preview(
            checkout=repo_root,
            output=package_root,
            oracle_inventory=staged_input(inputs.oracle_inventory, "oracle-inventory", staging_dir),
            source_counts=staged_input(inputs.source_counts, "source-counts", staging_dir),
        )
        provenance = _packaged_provenance(package_root)
        if provenance is not None and provenance.get("source_revision") != inputs.source_revision:
            raise ValueError(
                "packaged provenance pins revision"
                f" {provenance.get('source_revision')!r}, expected {inputs.source_revision!r}"
            )
        proof["packaged"] = provenance
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        receipt = seal(f"{inputs.output_root}/{PACKAGE_RECEIPT_NAME}", proof)
        proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "package_preview",
            {"mode": mode, "package": str(package_root), **proof["receipt"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "package_preview", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON PackagePreviewInputs document")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout the package is staged from")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in ("oracle_inventory", "source_counts"):
        if document.get(name) is not None:
            document[name] = ArtifactRef(**document[name])
    inputs = PackagePreviewInputs(**document)
    proof = run_package_preview(inputs, dry_run=args.mode == "dry-run", repo_root=args.repo_root)
    label = "PACKAGE-PREVIEW-DRY-RUN" if args.mode == "dry-run" else "PACKAGE-PREVIEW-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
