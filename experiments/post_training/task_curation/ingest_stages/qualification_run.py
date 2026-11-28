# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep qualification landing and retry state outside the attested source checkout."""

import json
import subprocess
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_registration import REGISTRATION_DIRECTORY
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.landing import commit_qualification
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


def already_landed(execution: IngestExecution, report: ArtifactRef, checkout: Path) -> str | None:
    """Recover an interrupted remote push by comparing the complete qualification."""
    subprocess.run(["git", "fetch", "origin", "atlas-env-audit"], cwd=checkout, check=True)
    source_root = Path(__file__).resolve().parents[4]
    path = REGISTRATION_DIRECTORY.relative_to(source_root) / f"{canonical_sha256(execution.policy.source_id)}.json"
    found = subprocess.check_output(["git", "ls-tree", "origin/atlas-env-audit", "--", str(path)], cwd=checkout)
    if not found:
        return None
    pointer = subprocess.check_output(["git", "show", f"origin/atlas-env-audit:{path.as_posix()}"], cwd=checkout)
    remote = read_registry_document(RegistryReference.model_validate_json(pointer))
    expected = read_ref(ArtifactRef.model_validate(read_ref(report)["outputs"]["admission"]))
    if remote != {execution.policy.source_id: expected}:
        raise ValueError("A different qualification is already registered for this source")
    return subprocess.check_output(["git", "rev-parse", "origin/atlas-env-audit"], cwd=checkout, text=True).strip()


def land_qualification(execution: IngestExecution, report: ArtifactRef, checkout: Path, landing_log: Path) -> str:
    """Land from an isolated worktree; same-execution retries recover a completed push."""
    path = StoragePath(execution.artifact_root) / "landing.json"
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved["report"] != report.model_dump(mode="json"):
            raise ValueError("Saved landing belongs to another completed worker report")
        return saved["revision"]
    revision = already_landed(execution, report, checkout)
    if revision is None:
        target = checkout.parent / f"atlas-ingest-landing-{execution.policy.digest()[:16]}"
        if target.exists():
            raise ValueError(f"An interrupted landing needs inspection in {target}; preserve its changes before retry")
        branch = f"benfeuer/atlas-ingest-landing-{execution.policy.digest()[:16]}"
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(target), execution.source_revision], cwd=checkout, check=True
        )
        (target / ".venv").symlink_to(checkout / ".venv", target_is_directory=True)
        revision = commit_qualification(execution, report, target, landing_log)
        subprocess.run(["git", "worktree", "remove", str(target)], cwd=checkout, check=True)
    seal_manifest({"revision": revision, "report": report.model_dump(mode="json")}, str(path))
    return revision
