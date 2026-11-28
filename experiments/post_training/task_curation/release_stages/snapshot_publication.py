# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot publication: seal derived contracts and publish the assembly snapshot.

One maintained module replaces the per-release ``publish-<release>-snapshot`` and
``seal-<release>-contracts`` scripts (see INVENTORY.md). All release-identifying
values arrive as typed :class:`SnapshotPublicationInputs`; the dry run
(``check_only``) verifies every precondition with zero writes (DESIGN.md §8b — this
stage writes object storage, not SQL, but the same pre-write discipline applies).
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.curation import CurationCatalog
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.catalog_snapshot import publish_assembly_snapshot
from experiments.post_training.task_curation.datasets.derived_contract_measurements import derived_contract_receipts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.sources import all_sources
from infra.marina.apps.rl_data_catalog.server.admission import require_catalog_declaration
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


@dataclass(frozen=True)
class SnapshotPublicationInputs:
    """Typed inputs for one snapshot-publication run; every value is a graph output."""

    release_id: str
    snapshot_id: str
    prior_snapshot_id: str
    source_revision: str
    catalog_root: str
    release_root: str
    targets: tuple[str, ...]
    removed_sources: tuple[str, ...]
    expected_source_count: int
    expected_contract_count: int
    expected_contract_tasks: int
    census_receipt: ArtifactRef
    identity_receipt: ArtifactRef
    backend_receipt: ArtifactRef
    deliberate_exclusions: tuple[str, ...] = ()

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.snapshot_id or not self.prior_snapshot_id:
            raise ValueError("release_id, snapshot_id and prior_snapshot_id are required")
        if not self.source_revision:
            raise ValueError("source_revision (the expected landed repo sha) is required")
        if self.expected_source_count <= 0 or not self.targets:
            raise ValueError("expected_source_count and targets must be non-empty")


def _read_ref(ref: ArtifactRef) -> dict:
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"sealed reference digest mismatch: {ref.uri}")
    return json.loads(payload)


def _seal(release_root: str, name: str, value: dict) -> ArtifactRef:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = StoragePath(f"{release_root}/{name}")
    if path.exists():
        raise ValueError(f"snapshot artifact already exists at {path}; bump the release prefix instead of overwriting")
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def verify_repo_state(repo_root: Path, inputs: SnapshotPublicationInputs) -> str:
    """The publishing checkout must be clean and contain the plan's pinned revision.

    In the in-region bundle mode the submitter attests the workspace (the guard's
    landing-log check carries the landed attestation) and the git checks cannot run.
    """
    from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — avoids the guard/snapshot cycle at module load
        attested_revision,
    )

    attested = attested_revision()
    if attested is not None:
        return attested
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"publishing checkout is dirty:\n{dirty[:400]}")
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


def verify_current_declarations(inputs: SnapshotPublicationInputs) -> dict[str, dict]:
    """Current declarations must cover exactly the release population with no drift."""
    sources = all_sources()
    current = {
        canonical_source_id(source.info.id): canonical_declaration(source.declaration).model_dump(mode="json")
        for source in sources.values()
    }
    if len(current) != inputs.expected_source_count:
        raise ValueError(f"{len(current)} current declarations, expected {inputs.expected_source_count}")
    for source in sources.values():
        require_catalog_declaration(source.declaration)
    return current


def carry_deliberate_exclusions(
    inputs: SnapshotPublicationInputs, current: dict[str, dict], prior_declarations_uri: str
) -> dict[str, dict]:
    """A deliberately excluded source carries the prior release's declaration verbatim.

    The frozen layer's re-pin is not adopted (the reason lives in the policy config's
    DELIBERATE_EXCLUSIONS), so the served row substitutes — the same substitution the
    declarations stage seals into the release's declarations document.
    """
    if not inputs.deliberate_exclusions:
        return current
    prior = json.loads(StoragePath(prior_declarations_uri).read_text())
    missing = sorted(set(inputs.deliberate_exclusions) - set(prior))
    if missing:
        raise ValueError(f"deliberate exclusions the prior release never declared: {missing[:5]}")
    return {
        **current,
        **{source_id: prior[source_id] for source_id in inputs.deliberate_exclusions},
    }


def verify_no_prior_drift(
    inputs: SnapshotPublicationInputs, current: dict[str, dict], prior_declarations_uri: str
) -> None:
    """Non-target declarations must be byte-identical to the prior snapshot's.

    A planned removal must be absent from the current declarations; it is the only
    prior source allowed to disappear.
    """
    prior = json.loads(StoragePath(prior_declarations_uri).read_text())
    still_declared = sorted(set(inputs.removed_sources) & set(current))
    if still_declared:
        raise ValueError(f"planned removals are still declared: {still_declared[:4]}")
    drift = []
    for source_id, row in prior.items():
        if source_id in inputs.targets or source_id in inputs.removed_sources:
            continue
        if source_id not in current or fingerprint_input(current[source_id]) != fingerprint_input(row):
            drift.append(source_id)
    if drift:
        raise ValueError(f"non-target declaration drift: {drift[:4]}")


def verify_survey_alignment(inputs: SnapshotPublicationInputs, current: dict[str, dict]) -> tuple[dict, dict]:
    """Both surveys must bind this publication and cover exactly its population."""
    identity = _read_ref(inputs.identity_receipt)
    backend = _read_ref(inputs.backend_receipt)
    if set(identity["fingerprints"]) != set(current):
        raise ValueError("identity fingerprints do not cover exactly the current declarations")
    if {row["source_id"] for row in backend["sources"]} != set(current):
        raise ValueError("backend sources do not cover exactly the current declarations")
    expected = (inputs.catalog_root, inputs.snapshot_id)
    for name, document in (("identity", identity), ("backend", backend)):
        binding = (document.get("catalog_root"), document.get("snapshot_id"))
        if binding != expected:
            raise ValueError(f"{name} survey publication binding {binding!r} differs from {expected!r}")
    return identity, backend


def seal_contracts(inputs: SnapshotPublicationInputs) -> ArtifactRef:
    """Seal every target's derived-contract receipt against its census binding.

    The census covers the whole declared population; each target must appear in it
    and hold a registered receipt, or the stage refuses.
    """
    census = _read_ref(inputs.census_receipt)
    declarations = {result["declaration"]["id"]: result["declaration"] for result in census["results"]}
    uncensused = sorted(set(inputs.targets) - set(declarations))
    if uncensused:
        raise ValueError(f"targets missing from the census: {uncensused}")
    receipts = derived_contract_receipts()
    unregistered = sorted(set(inputs.targets) - set(receipts))
    if unregistered:
        raise ValueError(f"targets without a derived-contract receipt: {unregistered}")
    rows = []
    for source_id in inputs.targets:
        receipt = receipts[source_id]
        if receipt.input.model_dump(mode="json") != declarations[source_id]["actor_census_input"]:
            raise ValueError(
                f"derived-contract receipt actor binding does not match the frozen declaration: {source_id}"
            )
        rows.append(receipt.model_dump(mode="json"))
    if len(rows) != inputs.expected_contract_count:
        raise ValueError(f"{len(rows)} sealed contracts, expected {inputs.expected_contract_count}")
    if sum(row["input"]["measured_tasks"] for row in rows) != inputs.expected_contract_tasks:
        raise ValueError("sealed contract measured tasks disagree with the release plan")
    return _seal(inputs.release_root, "fresh-contracts.json", {"contracts": rows})


def run_snapshot_publication(
    inputs: SnapshotPublicationInputs,
    *,
    check_only: bool,
    repo_root: Path,
    prior_declarations_uri: str,
    feeds: tuple[Path, ...] | None = None,
    publish: Callable | None = None,
) -> dict:
    """Run one snapshot-publication pass; ``check_only`` verifies with zero writes."""
    inputs.validate()
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "check-only" if check_only else "write"
    alerts.stage_started(inputs.release_id, "snapshot_publication", {"mode": mode}, alert_feeds)
    try:
        revision = verify_repo_state(repo_root, inputs)
        current = carry_deliberate_exclusions(inputs, verify_current_declarations(inputs), prior_declarations_uri)
        verify_no_prior_drift(inputs, current, prior_declarations_uri)
        _identity, _backend = verify_survey_alignment(inputs, current)
        proof: dict[str, object] = {
            "kind": "snapshot-publication",
            "release_id": inputs.release_id,
            "snapshot_id": inputs.snapshot_id,
            "prior_snapshot": inputs.prior_snapshot_id,
            "source_revision": revision,
            "source_count": len(current),
            "new_admissions": sorted(inputs.targets),
            "identity": {"uri": inputs.identity_receipt.uri, "sha256": inputs.identity_receipt.sha256},
            "backend": {"uri": inputs.backend_receipt.uri, "sha256": inputs.backend_receipt.sha256},
            "declarations_sha256": (
                hashlib.sha256((json.dumps(current, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
            ),
            "recorded_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        if check_only:
            proof["sql_writes"] = 0
            alerts.stage_finished(
                inputs.release_id,
                "snapshot_publication",
                {"mode": "check-only", "snapshot_id": inputs.snapshot_id},
                alert_feeds,
            )
            return proof
        configure_coreweave_s3()
        contracts_ref = seal_contracts(inputs)
        binding = _seal(inputs.release_root, "snapshot-binding-receipt.json", proof)
        catalog = CurationCatalog(inputs.catalog_root)
        if catalog.current().snapshot_id != inputs.prior_snapshot_id:
            raise ValueError(
                f"catalog pointer is {catalog.current().snapshot_id!r}, expected the prior snapshot"
                f" {inputs.prior_snapshot_id!r}"
            )
        publish = publish if publish is not None else publish_assembly_snapshot
        # The publication writes the carried declarations: an excluded source's served
        # row publishes verbatim, not the frozen layer's re-pin.
        sources = []
        for source in all_sources().values():
            source_id = canonical_source_id(source.info.id)
            if source_id in inputs.deliberate_exclusions and source_id in current:
                sources.append(
                    dataclasses.replace(source, declaration=type(source.declaration).model_validate(current[source_id]))
                )
            else:
                sources.append(source)
        result = publish(inputs.catalog_root, inputs.snapshot_id, sources, assembly_commit=revision)
        if CurationCatalog(inputs.catalog_root).current().snapshot_id != inputs.snapshot_id:
            raise ValueError(f"catalog pointer did not move to {inputs.snapshot_id!r}")
        final = {
            **proof,
            "contracts": {"uri": contracts_ref.uri, "sha256": contracts_ref.sha256},
            "binding_receipt": {"uri": binding.uri, "sha256": binding.sha256},
            "publication": result.model_dump(mode="json"),
        }
        publication_ref = _seal(inputs.release_root, "snapshot-publication.json", final)
        alerts.stage_finished(
            inputs.release_id,
            "snapshot_publication",
            {
                "mode": "write",
                "snapshot_id": inputs.snapshot_id,
                "uri": publication_ref.uri,
                "sha256": publication_ref.sha256,
            },
            alert_feeds,
        )
        return {**final, "receipt": {"uri": publication_ref.uri, "sha256": publication_ref.sha256}}
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "snapshot_publication", {"error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON SnapshotPublicationInputs document")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prior-declarations", required=True, help="URI of the prior snapshot's declarations document")
    parser.add_argument("--mode", choices=("check-only", "write"), default="check-only")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for field in ("census_receipt", "identity_receipt", "backend_receipt"):
        document[field] = ArtifactRef(**document[field])
    inputs = SnapshotPublicationInputs(**document)
    configure_coreweave_s3()
    proof = run_snapshot_publication(
        inputs,
        check_only=args.mode == "check-only",
        repo_root=args.repo_root,
        prior_declarations_uri=args.prior_declarations,
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
