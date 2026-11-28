# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Land immutable source qualifications with pointer checks inside atlas-push's lock."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.ingest_registration import (
    REGISTRATION_DIRECTORY,
    IngestAdmission,
    qualified_source,
)
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.registry_update import (
    RegistryPointers,
    pointers_at_revision,
    publish_update,
    stage_update,
)
from experiments.post_training.task_curation.ingest_stages.registry_update import check_landing as check_registry_landing
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.reference_data import commit_reference_documents
from experiments.post_training.task_curation.registry import RegistryReference, archive_registry_documents
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.source_duplicate_ci import FingerprintReference
from experiments.post_training.task_curation.source_population_cache import PopulationLocations


class LandingProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_proposal: str
    expected_files: dict[str, str | None]


def git_digest(checkout: Path, revision: str, path: str) -> str | None:
    entries = subprocess.check_output(["git", "ls-tree", "-z", revision, "--", path], cwd=checkout)
    if not entries:
        return None
    payload = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=checkout)
    return hashlib.sha256(payload).hexdigest()


def check_landing(checkout: Path, base_ref: str, proposal_path: Path) -> None:
    """Refuse lost license/admission updates after atlas-push fetches and rebases."""
    proposal = LandingProposal.model_validate_json((checkout / proposal_path).read_bytes())
    for path, expected in proposal.expected_files.items():
        if git_digest(checkout, base_ref, path) != expected:
            raise ValueError("Atlas qualification pointers moved; rebase and rebuild this proposal")
    check_registry_landing(checkout, base_ref, Path(proposal.registry_proposal))


def stage_qualification(execution: IngestExecution, report: ArtifactRef, checkout: Path) -> tuple[Path, list[Path]]:
    """Publish small immutable registry documents and prepare the Git selection."""
    outputs = {name: ArtifactRef.model_validate(ref) for name, ref in read_ref(report)["outputs"].items()}
    admission = IngestAdmission.model_validate(read_ref(outputs["admission"]))
    if admission.policy != execution.policy:
        raise ValueError("Worker qualification belongs to another source policy")
    if qualified_source(admission).declaration.model_dump(mode="json") != read_ref(outputs["declaration"]):
        raise ValueError("Host code projects a different declaration than the sealed worker")
    expected = RegistryPointers.model_validate(read_ref(execution.initial_artifacts["registry-pointers"]))
    if pointers_at_revision(checkout, "HEAD") != expected:
        raise ValueError("Registry pointers moved; rebase and repeat the overlap survey")
    registration = (
        REGISTRATION_DIRECTORY.relative_to(Path(__file__).resolve().parents[4])
        / f"{canonical_sha256(admission.source_id)}.json"
    )
    license_index = LICENSE_RECEIPT_INDEX.relative_to(Path(__file__).resolve().parents[4])
    expected_files = {
        path.as_posix(): git_digest(checkout, "HEAD", path.as_posix()) for path in (registration, license_index)
    }
    document = {admission.source_id: admission.model_dump(mode="json")}
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    remote_path = Path("experiments/post_training/task_curation/registry-objects") / f"{digest}.json"
    api = HfApi()
    with TemporaryDirectory(prefix="atlas-qualification-") as temporary:
        root = Path(temporary)
        staged, references = root / "staged", root / "references"
        target = staged / remote_path
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        head = api.repo_info(expected.fingerprints.repo, repo_type="dataset").sha
        archive_registry_documents(
            staged,
            references,
            repo=expected.fingerprints.repo,
            parent_revision=head,
            commit_message="Record measured Atlas source qualification",
        )
        pointer = RegistryReference.model_validate_json((references / remote_path).read_bytes())
    destination = checkout / registration
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(pointer.model_dump_json(indent=2) + "\n")
    receipt = admission.retained.original_info.license_receipt
    if receipt is None:
        raise ValueError("Source qualification has no ingest license provenance")
    current = read_license_receipts(checkout / license_index).get(admission.source_id)
    if current is not None and current != receipt:
        raise ValueError("Publication attempted to replace the original ingest license record")
    changed = [registration]
    if current is None:
        body = (
            json.dumps({admission.source_id: receipt.model_dump(mode="json")}, sort_keys=True, indent=2) + "\n"
        ).encode()
        commit_reference_documents(
            checkout / license_index,
            {f"ingest/{hashlib.sha256(body).hexdigest()}.json": body},
            message="Register source ingest license provenance",
        )
        changed.append(license_index)
    fingerprint = FingerprintReference.model_validate(read_ref(outputs["fingerprint-reference"]))
    locations = PopulationLocations.model_validate(read_ref(outputs["locators"]))
    update = publish_update(
        expected,
        {admission.source_id: fingerprint},
        locations,
        registry_head=api.repo_info(expected.fingerprints.repo, repo_type="dataset").sha,
    )
    relative_root = Path("experiments/post_training/task_curation/ingest_stages/landing_proposals")
    registry_proposal = relative_root / f"{execution.policy.digest()}-registry.json"
    stage_update(checkout, update, registry_proposal)
    proposal_path = relative_root / f"{execution.policy.digest()}.json"
    proposal = LandingProposal(registry_proposal=registry_proposal.as_posix(), expected_files=expected_files)
    (checkout / proposal_path).write_text(proposal.model_dump_json(indent=2) + "\n")
    changed.extend(
        [
            Path("experiments/post_training/task_curation/source_fingerprint_index.json"),
            Path("experiments/post_training/task_curation/source_population_locators.json"),
            registry_proposal,
            proposal_path,
        ]
    )
    return proposal_path, changed


def commit_qualification(execution: IngestExecution, report: ArtifactRef, checkout: Path, landing_log: Path) -> str:
    proposal, changed = stage_qualification(execution, report, checkout)
    subprocess.run(["./infra/pre-commit.py", "--fix", *map(str, changed)], cwd=checkout, check=True)
    subprocess.run(["git", "add", "--", *map(str, changed)], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-m", "Register a measured Atlas source qualification"], cwd=checkout, check=True)
    check = (
        "uv run --no-sync python -m experiments.post_training.task_curation.ingest_stages.landing "
        f"--check --base-ref origin/atlas-env-audit --proposal {proposal.as_posix()}"
    )
    tests = (
        "uv run --no-sync pytest -q experiments/post_training/task_curation/tests/test_catalog.py "
        "experiments/post_training/task_curation/tests/test_catalog_publication.py"
    )
    subprocess.run(
        [
            str(Path.home() / ".local/bin/atlas-push"),
            "--test",
            f"{check} && {tests}",
            "--waivers",
            "experiments/post_training/task_curation/declaration-projection-waivers.json",
            "--who",
            "atlas-ingest",
            "--what",
            "Register measured source qualification",
        ],
        cwd=checkout,
        env={**os.environ, "ATLAS_PUSH_LOG": str(landing_log)},
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args()
    check_landing(Path(__file__).resolve().parents[4], args.base_ref, args.proposal)


if __name__ == "__main__":
    main()
