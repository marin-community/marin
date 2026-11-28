# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identity/backend compose: extend the retained surveys with the fresh admissions.

One maintained module replaces the per-release compose scripts (see INVENTORY.md);
every release-identifying value arrives as typed :class:`IdentityBackendComposeInputs`
and the stage source carries none of them. The retained identity survey
and backend measurements are preserved, minus the plan's removed or replaced sources (their
fingerprints, backend rows and the pair reviews that touch them), and the fresh targets'
registry fingerprints, current-input pair reviews and backend survey rows are appended.
Both output surveys bind the planned publication snapshot; their input references
and measurement revisions remain intact.
Missing reason labels are projected from stored typed blockers without changing their measurements.

The stage verifies before it composes: the fresh backend survey must cover exactly the
targets under the current requirement schema, prior and fresh runtime settings must
agree, no retained pair review may touch a target, and every retained measurement must
match its fingerprint inputs; duplicate groups are recomputed through the maintained
folder instead of being copied. ``dry_run`` composes in memory with zero writes; write
mode seals both documents and receipts write-once with byte-identical readback.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import (
    BackendSourceMap,
    BackendSurvey,
    backend_schema_sha256,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import DuplicateReview, SourceFingerprint, SourceIdentitySurvey

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.backend_capabilities import project_backend_blocker_reasons
from experiments.post_training.task_curation.catalog_identity import (
    canonical_backend_survey,
    canonical_identity_survey,
)
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server.duplicate_folding import SourceStanding, fold_groups
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, canonical_source_map

STAGE = "identity_backend_compose"
RegistryReader = Callable[[RegistryReference], dict[str, Any]]


@dataclass(frozen=True)
class IdentityBackendComposeInputs:
    """Typed inputs for one compose run; every value is a graph output."""

    release_id: str
    snapshot_id: str
    source_revision: str
    output_root: str
    targets: tuple[str, ...]
    prior_identity: ArtifactRef
    prior_backend: ArtifactRef
    fresh_backend_receipt: ArtifactRef
    registry_index: ArtifactRef
    expected_target_count: int
    removed_sources: tuple[str, ...] = ()

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.snapshot_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, snapshot_id, source_revision and output_root are required")
        if not self.targets:
            raise ValueError("targets must name the fresh admissions")
        if self.expected_target_count <= 0:
            raise ValueError(f"expected_target_count must be positive, got {self.expected_target_count}")
        if len(self.targets) != self.expected_target_count:
            raise ValueError(f"targets carry {len(self.targets)} ids, expected {self.expected_target_count}")


REGISTRY_INDEX_RELPATH = Path("experiments/post_training/task_curation/release_stages/plans/registry-index.json")
_REPO_ROOT = Path(__file__).resolve().parents[4]


def read_bundled_registry_index(pin: ArtifactRef) -> dict:
    """The registry index travels as a digest, not a path: when the pin matches the
    bundled repo-relative file, the stage reads that file — a deriving machine's
    absolute path never becomes a stage's read path (the in-region bundle has no such
    filesystem). A pin that does not match the bundled bytes reads through the storage
    layer (the tests' injected indexes; a genuinely divergent pin fails there)."""
    payload = (_REPO_ROOT / REGISTRY_INDEX_RELPATH).read_bytes()
    if hashlib.sha256(payload).hexdigest() == pin.sha256:
        return json.loads(payload)
    return read_ref(pin)


def read_ref(ref: ArtifactRef) -> dict[str, Any]:
    """Read and digest-verify one sealed JSON reference."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"sealed reference digest mismatch: {ref.uri}")
    return json.loads(payload)


def seal(path: str, value: dict[str, Any]) -> ArtifactRef:
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


def verify_backend_alignment(
    prior: BackendSurvey, fresh: BackendSurvey, targets: tuple[str, ...], removed: tuple[str, ...] = ()
) -> list[BackendSourceMap]:
    """The fresh survey must cover exactly the targets and agree with the retained map."""
    fresh_ids = {row.source_id for row in fresh.sources}
    if fresh_ids != set(targets):
        raise ValueError(f"fresh survey sources and targets disagree: {sorted(fresh_ids ^ set(targets))}")
    if fresh.schema_sha256 != backend_schema_sha256():
        raise ValueError(
            f"fresh survey schema {fresh.schema_sha256} differs from the current requirement schema;"
            " re-measure the fresh survey before composing"
        )
    for name, retained, fresh_value in (
        ("runtime_defaults", prior.runtime_defaults, fresh.runtime_defaults),
        ("deployment_classes", prior.deployment_classes, fresh.deployment_classes),
        ("deployment_limits", prior.deployment_limits, fresh.deployment_limits),
    ):
        if retained != fresh_value:
            raise ValueError(f"prior and fresh {name} disagree; the retained settings must be carried unchanged")
    retained = [row for row in prior.sources if row.source_id not in set(removed) | set(targets)]
    composed = sorted([*retained, *fresh.sources], key=lambda row: row.source_id)
    if len(composed) != len(retained) + len(targets):
        raise ValueError("composed backend rows do not equal the retained rows plus the targets")
    return composed


def compose_backend_survey(
    prior: BackendSurvey, fresh: BackendSurvey, composed_sources: list[BackendSourceMap], *, snapshot_id: str
) -> BackendSurvey:
    """Bind verified rows to the publication snapshot, retaining measurement provenance."""
    composed = prior.model_dump(mode="json")
    composed.update(
        snapshot_id=snapshot_id,
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        git_revision=fresh.git_revision,
        schema_sha256=fresh.schema_sha256,
        sources=[row.model_dump(mode="json") for row in composed_sources],
    )
    return project_backend_blocker_reasons(BackendSurvey.model_validate(composed))


@dataclass(frozen=True)
class TargetIdentities:
    fingerprints: dict[str, SourceFingerprint]
    reviews: tuple[DuplicateReview, ...]


def fetch_target_identities(index: dict[str, Any], targets: tuple[str, ...], reader: RegistryReader) -> TargetIdentities:
    """Fetch targets' fingerprints and pair reviews from their pinned registry documents.

    Two registry-entry generations exist: the pinned Hugging Face reference
    (``{repo, revision, path, sha256}``) and the digest-bound storage reference
    (``{uri, sha256}``) the current registry shards stage. Both are accepted —
    each pins the identity document's bytes against the entry's digest.
    """
    sources = index.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("registry index carries no sources mapping")
    fingerprints: dict[str, SourceFingerprint] = {}
    reviews: dict[tuple[str, ...], DuplicateReview] = {}
    for target in targets:
        entry = sources.get(target)
        if not isinstance(entry, dict):
            raise ValueError(f"registry index holds no entry for target {target!r}")
        document = _registry_document(entry, reader)
        # Registry entries point at full identity surveys; a bare fingerprint map is also accepted.
        rows = document["fingerprints"] if "fingerprints" in document else document
        if not isinstance(rows, dict):
            raise ValueError(f"registry identity document for {target!r} carries no fingerprint mapping")
        rows = canonical_source_map(rows)
        if rows.get(target) is None:
            raise ValueError(f"registry identity document for {target!r} carries no fingerprint for it")
        fingerprint = SourceFingerprint.model_validate(rows[target])
        if fingerprint.task_count <= 0:
            raise ValueError(
                f"registry fingerprint for {target!r} measures {fingerprint.task_count} tasks;"
                " targets must carry measured populations"
            )
        fingerprints[target] = fingerprint
        for payload in document.get("reviews", []):
            review = DuplicateReview.model_validate(payload)
            pair = review.similarity.model_copy(
                update={
                    "left": canonical_source_id(review.similarity.left),
                    "right": canonical_source_id(review.similarity.right),
                }
            )
            if target not in {pair.left, pair.right}:
                continue
            review = review.model_copy(update={"similarity": pair})
            key = tuple(sorted((pair.left, pair.right)))
            if key in reviews and reviews[key] != review:
                raise ValueError(f"conflicting fresh pair reviews for {pair.left}/{pair.right}")
            reviews[key] = review
    return TargetIdentities(fingerprints, tuple(reviews.values()))


def _registry_document(entry: dict[str, Any], reader: RegistryReader) -> dict[str, Any]:
    """Read one registry entry's identity document, digest-verified against the entry."""
    try:
        return reader(RegistryReference.model_validate(entry))
    except ValidationError:
        pass
    uri = entry.get("uri")
    sha256 = entry.get("sha256")
    if not isinstance(uri, str) or not isinstance(sha256, str) or not uri or not sha256:
        raise ValueError(
            f"registry entry is neither a pinned reference nor a digest-bound storage reference: {sorted(entry)}"
        )
    payload = StoragePath(uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"registry identity document digest mismatch: {uri}")
    return json.loads(payload)


def compose_identity_survey(
    prior: SourceIdentitySurvey,
    targets: tuple[str, ...],
    removed: tuple[str, ...],
    fresh: TargetIdentities,
    snapshot_id: str,
) -> SourceIdentitySurvey:
    """Drop the planned removals, extend with the targets' fingerprints, recheck reviews and groups."""
    unknown = sorted(set(removed) - set(prior.fingerprints))
    if unknown:
        raise ValueError(f"planned removals are absent from the retained identity: {unknown[:4]}")
    replaced = set(prior.fingerprints) & set(targets)
    omitted = set(removed) | replaced
    retained = {source_id: fp for source_id, fp in prior.fingerprints.items() if source_id not in omitted}
    fingerprints: dict[str, SourceFingerprint | None] = {**retained, **fresh.fingerprints}
    retained_reviews = [
        review for review in prior.reviews if not {review.similarity.left, review.similarity.right} & omitted
    ]
    for review in retained_reviews:
        pair = review.similarity
        touching = sorted({pair.left, pair.right} & set(targets))
        if touching:
            raise ValueError(
                f"retained pair review {pair.left}/{pair.right} touches target source {touching[0]};"
                " retained reviews must not cover the fresh admissions"
            )
        if review.measurement is None:
            continue
        for side, recorded in (
            (pair.left, review.measurement.left_inputs_sha256),
            (pair.right, review.measurement.right_inputs_sha256),
        ):
            fingerprint = fingerprints.get(side)
            if fingerprint is None:
                continue
            if recorded != canonical_sha256(fingerprint.inputs):
                raise ValueError(f"retained review measurement inputs disagree with the fingerprint for {side}")
    if len(fingerprints) != len(retained) + len(targets):
        raise ValueError("composed fingerprints do not equal the retained fingerprints plus the targets")
    for review in fresh.reviews:
        pair = review.similarity
        measurement = review.measurement
        if measurement is None:
            # The registry's identity documents carry judge verdicts without the
            # content-nn overlap record — no identity document in the system attaches
            # a measurement (the served survey holds none, the admissions' documents
            # hold none). The review's provenance is its digest-pinned registry
            # document, so freshness binds only when a measurement is present.
            continue
        for side, recorded in (
            (pair.left, measurement.left_inputs_sha256),
            (pair.right, measurement.right_inputs_sha256),
        ):
            fingerprint = fingerprints.get(side)
            if fingerprint is None or recorded != canonical_sha256(fingerprint.inputs):
                raise ValueError(f"fresh review measurement inputs disagree with the fingerprint for {side}")
    served_reviews = [*retained_reviews, *fresh.reviews]
    standings = {
        source_id: SourceStanding(fingerprint.task_count)
        for source_id, fingerprint in fingerprints.items()
        if fingerprint is not None
    }
    return SourceIdentitySurvey.model_validate(
        {
            **prior.model_dump(mode="json"),
            "snapshot_id": snapshot_id,
            "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "fingerprints": {
                source_id: fingerprint.model_dump(mode="json") if fingerprint is not None else None
                for source_id, fingerprint in sorted(fingerprints.items())
            },
            "reviews": [review.model_dump(mode="json") for review in served_reviews],
            "groups": [group.model_dump(mode="json") for group in fold_groups(served_reviews, standings)],
        }
    )


def _compose(
    inputs: IdentityBackendComposeInputs,
    targets: tuple[str, ...],
    *,
    dry_run: bool,
    registry_reader: RegistryReader,
) -> dict[str, Any]:
    """Verify, compose in memory and — in write mode — seal both documents and receipts."""
    receipt = read_ref(inputs.fresh_backend_receipt)
    survey_ref = receipt.get("survey")
    if not isinstance(survey_ref, dict) or "uri" not in survey_ref or "sha256" not in survey_ref:
        raise ValueError("fresh backend receipt carries no survey reference")
    fresh = canonical_backend_survey(read_ref(ArtifactRef(**survey_ref)))
    prior_backend = canonical_backend_survey(read_ref(inputs.prior_backend))
    composed_sources = verify_backend_alignment(prior_backend, fresh, targets, inputs.removed_sources)
    backend_survey = compose_backend_survey(prior_backend, fresh, composed_sources, snapshot_id=inputs.snapshot_id)

    index = read_bundled_registry_index(inputs.registry_index)
    registry_revision = index.get("revision")
    if not isinstance(registry_revision, str) or not registry_revision:
        raise ValueError("registry index carries no revision")
    fresh_identities = fetch_target_identities(index, targets, registry_reader)
    prior_identity = canonical_identity_survey(read_ref(inputs.prior_identity))
    identity_survey = compose_identity_survey(
        prior_identity, targets, inputs.removed_sources, fresh_identities, inputs.snapshot_id
    )

    # Composed counts shared by the dry-run proof and the sealed receipts.
    prior_rows = {row.source_id: row for row in prior_backend.sources}
    retained_rows = [row for row in backend_survey.sources if row.source_id not in targets]
    backend_counts = {
        "retained_rows_unchanged": sum(row == prior_rows[row.source_id] for row in retained_rows),
        "retained_measurements_unchanged": len(retained_rows),
        "blocker_reason_labels_added": (
            sum(len(fit.blocker_reasons) for row in backend_survey.sources for fit in row.backends.values())
            - sum(len(fit.blocker_reasons) for row in composed_sources for fit in row.backends.values())
        ),
        "added_rows": sorted(targets),
        "source_count": len(backend_survey.sources),
        "prior_schema_sha256": prior_backend.schema_sha256,
        "composed_schema_sha256": fresh.schema_sha256,
    }
    identity_counts = {
        "retained_fingerprints_unchanged": len(identity_survey.fingerprints) - len(targets),
        "removed_fingerprints": sorted(inputs.removed_sources),
        "added_fingerprints": sorted(targets),
        "fingerprint_count": len(identity_survey.fingerprints),
        "retained_review_count": len(identity_survey.reviews) - len(fresh_identities.reviews),
        "pair_reviews_touching_targets": len(fresh_identities.reviews),
    }
    replaced = set(targets) & set(prior_identity.fingerprints)
    if replaced:
        identity_counts["replaced_fingerprints"] = sorted(replaced)
        identity_counts["superseded_review_count"] = sum(
            bool({row.similarity.left, row.similarity.right} & replaced) for row in prior_identity.reviews
        )
    completed_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    receipt_base = {
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "snapshot_id": inputs.snapshot_id,
        "registry_revision": registry_revision,
    }
    proof: dict[str, Any] = {
        "kind": f"atlas-{inputs.release_id}-identity-backend-compose",
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "snapshot_id": inputs.snapshot_id,
        "registry_revision": registry_revision,
        "backend": {"kind": f"atlas-{inputs.release_id}-backend-composition", **backend_counts},
        "identity": {"kind": f"atlas-{inputs.release_id}-identity-composition", **identity_counts},
        "dry_run": dry_run,
        "sql_writes": 0,
        "model_calls": 0,
        "completed_at": completed_at,
    }
    if dry_run:
        return proof
    backend_ref = seal(f"{inputs.output_root}/backend-map.json", backend_survey.model_dump(mode="json"))
    backend_receipt = seal(
        f"{inputs.output_root}/backend-composition-receipt.json",
        {
            "kind": f"atlas-{inputs.release_id}-backend-composition",
            **receipt_base,
            "prior": {"uri": inputs.prior_backend.uri, "sha256": inputs.prior_backend.sha256},
            "fresh_survey": {"uri": survey_ref["uri"], "sha256": survey_ref["sha256"]},
            **backend_counts,
            "backend": {"uri": backend_ref.uri, "sha256": backend_ref.sha256},
            "sql_writes": 0,
            "model_calls": 0,
            "completed_at": completed_at,
        },
    )
    identity_ref = seal(f"{inputs.output_root}/source-identities.json", identity_survey.model_dump(mode="json"))
    identity_receipt = seal(
        f"{inputs.output_root}/identity-composition-receipt.json",
        {
            "kind": f"atlas-{inputs.release_id}-identity-composition",
            **receipt_base,
            "prior": {"uri": inputs.prior_identity.uri, "sha256": inputs.prior_identity.sha256},
            **identity_counts,
            "identity": {"uri": identity_ref.uri, "sha256": identity_ref.sha256},
            "sql_writes": 0,
            "model_calls": 0,
            "completed_at": completed_at,
        },
    )
    proof["documents"] = {
        "backend_map": {"uri": backend_ref.uri, "sha256": backend_ref.sha256},
        "backend_receipt": {"uri": backend_receipt.uri, "sha256": backend_receipt.sha256},
        "identity_survey": {"uri": identity_ref.uri, "sha256": identity_ref.sha256},
        "identity_receipt": {"uri": identity_receipt.uri, "sha256": identity_receipt.sha256},
    }
    return proof


def run_identity_backend_compose(
    inputs: IdentityBackendComposeInputs,
    *,
    dry_run: bool,
    registry_reader: RegistryReader = read_registry_document,
    feeds: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Run one compose pass; ``dry_run`` verifies and composes with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, STAGE, {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        targets = tuple(sorted({canonical_source_id(target) for target in inputs.targets}))
        if len(targets) != len(inputs.targets):
            raise ValueError("targets repeat or alias one canonical source id")
        proof = _compose(inputs, targets, dry_run=dry_run, registry_reader=registry_reader)
        alerts.stage_finished(
            inputs.release_id,
            STAGE,
            {
                "mode": mode,
                "backend_source_count": proof["backend"]["source_count"],
                "identity_fingerprint_count": proof["identity"]["fingerprint_count"],
                **({} if dry_run else proof["documents"]),
            },
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, STAGE, {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, required=True, help="Path to a JSON IdentityBackendComposeInputs document"
    )
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for field in ("prior_identity", "prior_backend", "fresh_backend_receipt", "registry_index"):
        document[field] = ArtifactRef(**document[field])
    document["targets"] = tuple(document["targets"])
    inputs = IdentityBackendComposeInputs(**document)
    proof = run_identity_backend_compose(inputs, dry_run=args.mode == "dry-run")
    label = "IDENTITY-BACKEND-COMPOSE-DRY-RUN" if args.mode == "dry-run" else "IDENTITY-BACKEND-COMPOSE-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
