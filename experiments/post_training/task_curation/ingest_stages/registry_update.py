# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish immutable index documents and compare repository pointers at landing."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.registry import (
    RegistryReference,
    archive_registry_documents,
    read_registry_document,
)
from experiments.post_training.task_curation.source_duplicate_ci import FingerprintReference
from experiments.post_training.task_curation.source_population_cache import PopulationLocations
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_map

REGISTRY_DIRECTORY = Path("experiments/post_training/task_curation")
FINGERPRINT_POINTER = REGISTRY_DIRECTORY / "source_fingerprint_index.json"
LOCATOR_POINTER = REGISTRY_DIRECTORY / "source_population_locators.json"
OBJECT_DIRECTORY = REGISTRY_DIRECTORY / "registry-objects"


class RegistryPointers(BaseModel):
    """The index and locators selected together by one repository commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprints: RegistryReference
    locators: RegistryReference


class RegistryUpdate(BaseModel):
    """Compare-and-swap proposal committed beside the replacement pointers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: RegistryPointers
    proposed: RegistryPointers


def pointers_at_revision(checkout: Path, revision: str) -> RegistryPointers:
    """Read the committed pair, so untracked or edited pointers cannot pass."""
    return RegistryPointers(
        fingerprints=RegistryReference.model_validate_json(git_file(checkout, revision, FINGERPRINT_POINTER)),
        locators=RegistryReference.model_validate_json(git_file(checkout, revision, LOCATOR_POINTER)),
    )


def git_file(checkout: Path, revision: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path.as_posix()}"], cwd=checkout)


def read_fingerprints(reference: RegistryReference) -> dict[str, FingerprintReference]:
    return canonical_source_map(
        {key: FingerprintReference.model_validate(value) for key, value in read_registry_document(reference).items()}
    )


def require_preserved_entries(previous: dict, proposed: dict, label: str) -> None:
    dropped = previous.keys() - proposed.keys()
    if dropped:
        raise ValueError(f"registry update drops {label} entries: {sorted(dropped)}")


def verify_update(update: RegistryUpdate) -> None:
    """Verify new bytes, content-addressed names and a superset of every old map."""
    for old, new in (
        (update.expected.fingerprints, update.proposed.fingerprints),
        (update.expected.locators, update.proposed.locators),
    ):
        if new.repo != old.repo:
            raise ValueError("registry update cannot change the archive repository")
        if new.path != (OBJECT_DIRECTORY / f"{new.sha256}.json").as_posix():
            raise ValueError("registry update must reference content-addressed documents")
    before = read_fingerprints(update.expected.fingerprints)
    after = read_fingerprints(update.proposed.fingerprints)
    require_preserved_entries(before, after, "fingerprint")
    old_locations = PopulationLocations.model_validate(read_registry_document(update.expected.locators))
    new_locations = PopulationLocations.model_validate(read_registry_document(update.proposed.locators))
    require_preserved_entries(old_locations.files, new_locations.files, "file locator")
    require_preserved_entries(old_locations.prepared, new_locations.prepared, "prepared population")


def publish_update(
    expected: RegistryPointers,
    fingerprints: dict[str, FingerprintReference],
    locators: PopulationLocations,
    *,
    registry_head: str,
) -> RegistryUpdate:
    """Append merged content-addressed documents without changing Git pointers.

    The caller measures the current archive head. The existing registry writer's
    parent-commit check refuses an archive race; the caller refreshes and retries.
    Git selection changes only through a separately committed landing proposal.
    """
    if expected.fingerprints.repo != expected.locators.repo:
        raise ValueError("fingerprints and locators must share an archive repository")
    merged = read_fingerprints(expected.fingerprints)
    merged.update(canonical_source_map(fingerprints))
    old_locations = PopulationLocations.model_validate(read_registry_document(expected.locators))
    merged_locations = PopulationLocations(
        files={**old_locations.files, **locators.files},
        prepared={**old_locations.prepared, **locators.prepared},
    )
    documents = (
        {key: reference.model_dump(mode="json") for key, reference in merged.items()},
        merged_locations.model_dump(mode="json"),
    )
    with TemporaryDirectory(prefix="atlas-ingest-registry-") as temporary:
        root = Path(temporary)
        staged = root / "staged"
        references = root / "references"
        published = []
        missing = []
        api = HfApi()
        if not api.repo_info(repo_id=expected.fingerprints.repo, repo_type="dataset", revision=registry_head).private:
            raise ValueError("registry archive must be private")
        for document in documents:
            payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
            digest = hashlib.sha256(payload).hexdigest()
            name = OBJECT_DIRECTORY / f"{digest}.json"
            reference = RegistryReference(
                repo=expected.fingerprints.repo,
                revision=registry_head,
                path=name.as_posix(),
                sha256=digest,
            )
            if api.file_exists(
                repo_id=reference.repo, repo_type="dataset", revision=registry_head, filename=reference.path
            ):
                read_registry_document(reference)
                published.append(reference)
                continue
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            missing.append((len(published), name))
            published.append(reference)
        if missing:
            archive_registry_documents(
                staged,
                references,
                repo=expected.fingerprints.repo,
                parent_revision=registry_head,
                commit_message="Record immutable Atlas ingest index and locators",
            )
            for index, name in missing:
                published[index] = RegistryReference.model_validate_json((references / name).read_bytes())
        proposed = RegistryPointers(fingerprints=published[0], locators=published[1])
    update = RegistryUpdate(expected=expected, proposed=proposed)
    verify_update(update)
    return update


def stage_update(checkout: Path, update: RegistryUpdate, proposal: Path) -> None:
    """Prepare a committed compare-and-swap proposal in the author's worktree."""
    proposal = (checkout / proposal).resolve().relative_to(checkout.resolve())
    actual = RegistryPointers(
        fingerprints=RegistryReference.model_validate_json((checkout / FINGERPRINT_POINTER).read_bytes()),
        locators=RegistryReference.model_validate_json((checkout / LOCATOR_POINTER).read_bytes()),
    )
    if actual != update.expected:
        raise ValueError("registry pointers moved; rebase and rebuild the update from the new selection")
    verify_update(update)
    for path, reference in (
        (FINGERPRINT_POINTER, update.proposed.fingerprints),
        (LOCATOR_POINTER, update.proposed.locators),
    ):
        (checkout / path).write_text(reference.model_dump_json(indent=2) + "\n")
    destination = checkout / proposal
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(update.model_dump_json(indent=2) + "\n")


def check_landing(checkout: Path, base_revision: str, proposal: Path) -> None:
    """Run inside atlas-push's lock after fetching and rebasing onto its target."""
    proposal = (checkout / proposal).resolve().relative_to(checkout.resolve())
    update = RegistryUpdate.model_validate_json(git_file(checkout, "HEAD", proposal))
    if pointers_at_revision(checkout, base_revision) != update.expected:
        raise ValueError("registry pointers moved; rebase and rebuild the update from the new selection")
    if pointers_at_revision(checkout, "HEAD") != update.proposed:
        raise ValueError("committed registry pointers differ from the compare-and-swap proposal")
    verify_update(update)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    check_landing(checkout, args.base_ref, args.proposal)


if __name__ == "__main__":
    main()
