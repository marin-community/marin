# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained identity/backend compose stage (release slice).

The fixtures build minimal valid sealed survey documents (retained identity, retained
backend map, fresh backend receipt, registry index and per-source registry identity
documents) and run the real compose path. The registry reader is the injected
boundary; the release graph binds the real registry client at run time.
"""

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from shellbox.machine import Backend, NetworkPolicy
from shellbox.requirements import Requirement
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    FitSupport,
    RuntimeProfile,
    backend_schema_sha256,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import (
    DirectionalOverlap,
    DuplicateLabel,
    DuplicateReview,
    MeasuredOverlap,
    PairSimilarity,
    PairVerdict,
    SourceFingerprint,
    SourceIdentitySurvey,
)

from experiments.post_training.task_curation.backend_capabilities import verify_backend_blocker_reasons
from experiments.post_training.task_curation.registry import RegistryReference
from experiments.post_training.task_curation.release_stages import identity_backend_compose as compose_stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server.duplicate_folding import SourceStanding, fold_groups

BACKENDS = tuple(backend.value for backend in Backend)
PRIOR_IDS = ("Test:Prior-One", "Test:Prior-Two", "Test:Prior-Three")
TARGETS = ("Test:Fresh-One", "Test:Fresh-Two")
REGISTRY_REVISION = "c" * 40


def _fingerprint(source_id: str, task_count: int = 3) -> SourceFingerprint:
    return SourceFingerprint(
        seed=0,
        task_count=task_count,
        within_source_duplicates=0,
        sample_sha256=hashlib.sha256(f"sample:{source_id}".encode()).hexdigest(),
        sample=[],
        instruction_minhash=[],
        instruction_hashes=[],
        reference_hashes=[],
        test_hashes=[],
        task_id_hashes=[],
        url_hashes=[],
        lineage=[],
        inputs={"task_specs_sha256": hashlib.sha256(f"inputs:{source_id}".encode()).hexdigest()},
    )


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        network=NetworkPolicy.DENY,
        cpus=1,
        memory_mb=1024,
        storage_mb=1024,
        gpus=0,
        user=None,
        startup_timeout=30.0,
        cleanup_timeout=30.0,
    )


def _measured_source(source_id: str, task_count: int = 3) -> BackendSourceMap:
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=task_count,
        inputs={},
        demands={},
        resource_ranges={},
        backends={
            backend: BackendFit(
                support=FitSupport.SUPPORTED,
                task_counts={FitSupport.SUPPORTED: task_count},
                roles={role: {FitSupport.SUPPORTED: task_count} for role in ("agent", "grader")},
                blockers={},
                validation_errors={},
            )
            for backend in BACKENDS
        },
        role_backend_assignments={},
        grading_environment=None,
    )


def _backend_survey(source_rows: list[BackendSourceMap], **overrides) -> BackendSurvey:
    profile = _runtime_profile()
    document = {
        "catalog_root": "test:catalog",
        "snapshot_id": "assembly-compose-test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "git_revision": "b" * 40,
        "schema_sha256": backend_schema_sha256(),
        "runtime_defaults": {"agent": profile, "grader": profile},
        "matrix": [
            {
                "backend": backend,
                "requirement": requirement,
                "support": "supported",
                "evidence": ["fixture"],
                "note": "fixture",
            }
            for backend in BACKENDS
            for requirement in Requirement
        ],
        "deployment_limits": {"daytona": {"max_snapshots": 1}},
        "deployment_classes": {"iris": "production", "daytona": "production", "local": "production"},
        "sources": [row.model_dump(mode="json") for row in source_rows],
    }
    return BackendSurvey.model_validate({**document, **overrides})


def _directional(fraction: float) -> DirectionalOverlap:
    return DirectionalOverlap(
        population=0,
        sampled=0,
        matched=0,
        fraction=fraction,
        ci_low=0.0,
        ci_high=0.0,
        exact_matched=0,
        exact_fraction=0.0,
        neighbors=[],
    )


def _review(
    left: str,
    right: str,
    left_fp: SourceFingerprint,
    right_fp: SourceFingerprint,
    *,
    label: DuplicateLabel = DuplicateLabel.DISTINCT,
) -> DuplicateReview:
    fraction = {DuplicateLabel.DUPLICATE: 1.0, DuplicateLabel.PARTIAL_OVERLAP: 0.5, DuplicateLabel.DISTINCT: 0.0}[label]
    return DuplicateReview(
        similarity=PairSimilarity(
            left=left,
            right=right,
            rank_score=fraction,
            instruction_similarity=0.0,
            instruction_overlap=0.0,
            reference_overlap=0.0,
            test_overlap=0.0,
            task_id_overlap=0.0,
            url_overlap=0.0,
            shared_lineage=False,
        ),
        verdict=PairVerdict(
            label=label,
            overlap_fraction=fraction,
            reason="fixture ruling",
        ),
        prompt_sha256=hashlib.sha256(f"prompt:{left}:{right}".encode()).hexdigest(),
        model="fixture-model",
        completion_id="fixture-completion",
        prompt_tokens=1,
        completion_tokens=1,
        measurement=MeasuredOverlap(
            seed=0,
            sample_size=0,
            left_inputs_sha256=canonical_sha256(left_fp.inputs),
            right_inputs_sha256=canonical_sha256(right_fp.inputs),
            similarity_threshold=1.0,
            duplicate_threshold=1.0,
            partial_threshold=0.5,
            left_to_right=_directional(fraction),
            right_to_left=_directional(0.0),
        ),
    )


def _identity_survey(
    fingerprints: dict[str, SourceFingerprint | None], reviews: list[DuplicateReview], **overrides
) -> SourceIdentitySurvey:
    # Groups are deliberately stale (empty): the compose must recompute them.
    document = {
        "catalog_root": "test:catalog",
        "snapshot_id": "assembly-prior-test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "fingerprints": {
            source_id: fp.model_dump(mode="json") if fp is not None else None for source_id, fp in fingerprints.items()
        },
        "reviews": [review.model_dump(mode="json") for review in reviews],
        "groups": [],
        "judge": None,
    }
    return SourceIdentitySurvey.model_validate({**document, **overrides})


def _write_json(tmp_path: Path, name: str, document: dict) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _document_bytes(document: dict) -> bytes:
    return (json.dumps(document, sort_keys=True) + "\n").encode()


def _registry_stack(
    tmp_path: Path, fingerprints: dict[str, SourceFingerprint], reviews: list[DuplicateReview]
) -> tuple[ArtifactRef, Callable[[RegistryReference], dict]]:
    """Seal per-source registry identity documents; return the index ref and fake reader."""
    documents: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    for source_id, fingerprint in fingerprints.items():
        path = f"identities/{source_id}.json"
        document = {
            "fingerprints": {source_id: fingerprint.model_dump(mode="json")},
            "reviews": [
                review.model_dump(mode="json")
                for review in reviews
                if source_id in {review.similarity.left, review.similarity.right}
            ],
        }
        documents[path] = document
        sources[source_id] = {
            "repo": "fixture/registry",
            "revision": "d" * 40,
            "path": path,
            "sha256": hashlib.sha256(_document_bytes(document)).hexdigest(),
        }
    index = {"kind": "source-fingerprint-index", "revision": REGISTRY_REVISION, "sources": sources}

    def reader(reference: RegistryReference) -> dict:
        raw = _document_bytes(documents[reference.path])
        if hashlib.sha256(raw).hexdigest() != reference.sha256:
            raise ValueError("registry digest mismatch")
        return json.loads(raw)

    return _write_json(tmp_path, "registry-index.json", index), reader


def _build(
    tmp_path: Path,
    *,
    targets: tuple[str, ...] = TARGETS,
    prior_backend: BackendSurvey | None = None,
    fresh_backend: BackendSurvey | None = None,
    prior_identity: SourceIdentitySurvey | None = None,
    registry_fingerprints: dict[str, SourceFingerprint] | None = None,
    registry_reviews: list[DuplicateReview] | None = None,
) -> tuple[compose_stage.IdentityBackendComposeInputs, Callable[[RegistryReference], dict]]:
    """Standard inputs over three retained sources and the typed targets; override any piece."""
    prior_fps = {source_id: _fingerprint(source_id) for source_id in PRIOR_IDS}
    reviews = [
        _review(
            PRIOR_IDS[0], PRIOR_IDS[1], prior_fps[PRIOR_IDS[0]], prior_fps[PRIOR_IDS[1]], label=DuplicateLabel.DUPLICATE
        ),
        _review(PRIOR_IDS[2], PRIOR_IDS[0], prior_fps[PRIOR_IDS[2]], prior_fps[PRIOR_IDS[0]]),
    ]
    identity = prior_identity or _identity_survey(prior_fps, reviews)
    retained = prior_backend or _backend_survey([_measured_source(source_id) for source_id in PRIOR_IDS])
    fresh = fresh_backend or _backend_survey([_measured_source(source_id) for source_id in targets])
    registry = (
        registry_fingerprints
        if registry_fingerprints is not None
        else {source_id: _fingerprint(source_id) for source_id in targets}
    )
    index_ref, reader = _registry_stack(tmp_path, registry, registry_reviews or [])
    survey_ref = _write_json(tmp_path, "fresh-backend.json", fresh.model_dump(mode="json"))
    receipt = _write_json(
        tmp_path,
        "fresh-backend-receipt.json",
        {"kind": "backend-survey-receipt", "survey": survey_ref.model_dump()},
    )
    inputs = compose_stage.IdentityBackendComposeInputs(
        release_id="testrelease",
        snapshot_id=fresh.snapshot_id,
        source_revision="e" * 40,
        output_root=str(tmp_path / "release"),
        targets=targets,
        prior_identity=_write_json(tmp_path, "prior-identity.json", identity.model_dump(mode="json")),
        prior_backend=_write_json(tmp_path, "prior-backend.json", retained.model_dump(mode="json")),
        fresh_backend_receipt=receipt,
        registry_index=index_ref,
        expected_target_count=len(targets),
    )
    return inputs, reader


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def _feed_rows(tmp_path: Path) -> list[dict]:
    return [json.loads(row) for row in (tmp_path / "alerts.jsonl").read_text().splitlines()]


def test_dry_run_verifies_with_zero_writes(tmp_path):
    inputs, reader = _build(tmp_path)
    proof = compose_stage.run_identity_backend_compose(
        inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    assert proof["kind"] == "atlas-testrelease-identity-backend-compose"
    assert proof["backend"]["source_count"] == 5 and proof["backend"]["added_rows"] == sorted(TARGETS)
    assert proof["identity"]["fingerprint_count"] == 5
    assert proof["identity"]["retained_review_count"] == 2
    assert proof["registry_revision"] == REGISTRY_REVISION
    assert proof["sql_writes"] == 0 and proof["model_calls"] == 0
    assert not (tmp_path / "release").exists()
    assert [row["kind"] for row in _feed_rows(tmp_path)] == ["STAGE_START", "STAGE_FINISH"]


def test_planned_removals_leave_both_surveys(tmp_path):
    """A removed source drops from the backend map, the identity and the reviews touching it."""
    inputs, reader = _build(tmp_path)
    removing = dataclasses.replace(inputs, removed_sources=(PRIOR_IDS[2],))
    proof = compose_stage.run_identity_backend_compose(
        removing, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    assert proof["backend"]["source_count"] == 4
    assert proof["identity"]["fingerprint_count"] == 4
    assert proof["identity"]["removed_fingerprints"] == [PRIOR_IDS[2]]
    assert proof["identity"]["retained_review_count"] == 1

    unknown = dataclasses.replace(inputs, removed_sources=("Test:Never-Retained",))
    with pytest.raises(ValueError, match="planned removals are absent from the retained identity"):
        compose_stage.run_identity_backend_compose(unknown, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_metadata_snapshot_rebind_preserves_measurements_and_their_provenance(tmp_path):
    fingerprints = {source_id: _fingerprint(source_id) for source_id in PRIOR_IDS}
    inputs, reader = _build(tmp_path, targets=(PRIOR_IDS[0],), prior_identity=_identity_survey(fingerprints, []))
    destination = dataclasses.replace(inputs, snapshot_id="assembly-metadata-refresh")
    prior_identity = compose_stage.read_ref(inputs.prior_identity)
    prior_backend = compose_stage.read_ref(inputs.prior_backend)
    fresh_receipt = compose_stage.read_ref(inputs.fresh_backend_receipt)
    fresh_ref = ArtifactRef(**fresh_receipt["survey"])
    fresh = compose_stage.read_ref(fresh_ref)

    proof = compose_stage.run_identity_backend_compose(
        destination, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    identity = compose_stage.read_ref(ArtifactRef(**proof["documents"]["identity_survey"]))
    backend = compose_stage.read_ref(ArtifactRef(**proof["documents"]["backend_map"]))
    assert identity["snapshot_id"] == backend["snapshot_id"] == destination.snapshot_id
    assert proof["snapshot_id"] == destination.snapshot_id
    assert identity["fingerprints"] == prior_identity["fingerprints"]
    assert {row["source_id"]: row for row in backend["sources"]} == {
        row["source_id"]: row for row in prior_backend["sources"]
    }
    assert backend["git_revision"] == fresh["git_revision"]
    for key in ("identity_receipt", "backend_receipt"):
        receipt = compose_stage.read_ref(ArtifactRef(**proof["documents"][key]))
        assert receipt["snapshot_id"] == destination.snapshot_id
    backend_receipt = compose_stage.read_ref(ArtifactRef(**proof["documents"]["backend_receipt"]))
    assert backend_receipt["fresh_survey"] == fresh_receipt["survey"]
    assert compose_stage.read_ref(inputs.prior_identity) == prior_identity
    assert compose_stage.read_ref(inputs.prior_backend) == prior_backend
    assert compose_stage.read_ref(fresh_ref) == fresh


def test_composition_labels_retained_blockers_without_changing_measurements(tmp_path):
    retained = _measured_source(PRIOR_IDS[0])
    fits = dict(retained.backends)
    fits["daytona"] = BackendFit(
        support=FitSupport.NOT_APPLICABLE,
        task_counts={FitSupport.NOT_APPLICABLE: 3},
        roles={"agent": {FitSupport.ROLE_ABSENT: 3}, "grader": {FitSupport.NOT_APPLICABLE: 3}},
        blockers={},
        not_applicable={"grader:host_image": 3},
        validation_errors={},
    )
    fits["qemu"] = BackendFit(
        support=FitSupport.UNSUPPORTED,
        task_counts={FitSupport.UNSUPPORTED: 3},
        roles={role: {FitSupport.UNSUPPORTED: 3} for role in ("agent", "grader")},
        blockers={"agent:required_backend": 3, "grader:network": 3},
        validation_errors={"grader: QEMU does not provide storage resizing or GPU allocation": 3},
    )
    retained = retained.model_copy(update={"backends": fits})
    prior = _backend_survey([retained, *(_measured_source(source_id) for source_id in PRIOR_IDS[1:])])
    inputs, reader = _build(tmp_path, prior_backend=prior)
    proof = compose_stage.run_identity_backend_compose(
        inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    output = compose_stage.read_ref(ArtifactRef(**proof["documents"]["backend_map"]))
    verify_backend_blocker_reasons(BackendSurvey.model_validate(output))
    rows = {row["source_id"]: row for row in output["sources"]}
    projected = rows[retained.source_id]
    assert projected["backends"]["daytona"]["blocker_reasons"] == {"grader:host_image": "unsupported-environment-kind"}
    assert projected["backends"]["qemu"]["blocker_reasons"] == {
        "agent:required_backend": "required-backend",
        "grader:network": "network",
    }
    for name, fit in projected["backends"].items():
        fit["blocker_reasons"] = retained.backends[name].blocker_reasons
    assert projected == retained.model_dump(mode="json")
    assert compose_stage.read_ref(inputs.prior_backend) == prior.model_dump(mode="json")
    assert proof["backend"]["blocker_reason_labels_added"] == 3
    assert proof["backend"]["retained_measurements_unchanged"] == 3
    assert proof["backend"]["retained_rows_unchanged"] == 2


@pytest.mark.parametrize(
    ("field", "key"),
    [("not_applicable", "grader:network"), ("blockers", "grader:unknown-capability")],
)
def test_composition_refuses_unclassifiable_retained_blockers(tmp_path, field, key):
    retained = _measured_source(PRIOR_IDS[0])
    fit = retained.backends["daytona"].model_copy(update={field: {key: 3}})
    retained = retained.model_copy(update={"backends": {**retained.backends, "daytona": fit}})
    prior = _backend_survey([retained, *(_measured_source(source_id) for source_id in PRIOR_IDS[1:])])
    inputs, reader = _build(tmp_path, prior_backend=prior)
    with pytest.raises(ValueError, match="Cannot project backend reason"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path))
    assert not (tmp_path / "release").exists()


def test_write_seals_documents_and_receipts_then_refuses_a_rerun(tmp_path):
    inputs, reader = _build(tmp_path)
    compose_stage.run_identity_backend_compose(inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path))
    root = tmp_path / "release"
    backend_map = json.loads((root / "backend-map.json").read_text())
    identity = json.loads((root / "source-identities.json").read_text())
    backend_receipt = json.loads((root / "backend-composition-receipt.json").read_text())
    identity_receipt = json.loads((root / "identity-composition-receipt.json").read_text())
    assert backend_receipt["kind"] == "atlas-testrelease-backend-composition"
    assert backend_receipt["source_count"] == 5 and backend_receipt["retained_rows_unchanged"] == 3
    assert backend_receipt["added_rows"] == sorted(TARGETS)
    assert backend_receipt["sql_writes"] == 0 and backend_receipt["model_calls"] == 0
    assert backend_receipt["registry_revision"] == REGISTRY_REVISION
    assert backend_receipt["backend"] == {
        "uri": str(root / "backend-map.json"),
        "sha256": hashlib.sha256((root / "backend-map.json").read_bytes()).hexdigest(),
    }
    assert identity_receipt["kind"] == "atlas-testrelease-identity-composition"
    assert identity_receipt["registry_revision"] == REGISTRY_REVISION
    assert identity_receipt["retained_fingerprints_unchanged"] == 3
    assert identity_receipt["added_fingerprints"] == sorted(TARGETS)
    assert identity_receipt["pair_reviews_touching_targets"] == 0
    assert {row["source_id"] for row in backend_map["sources"]} == set(PRIOR_IDS) | set(TARGETS)
    assert set(backend_map["sources"][0]) >= {"source_id", "task_count", "backends"}
    # Groups are recomputed through the maintained folder, not copied from the stale prior.
    survey = SourceIdentitySurvey.model_validate(identity)
    standings = {
        source_id: SourceStanding(fp.task_count) for source_id, fp in survey.fingerprints.items() if fp is not None
    }
    recomputed = [group.model_dump(mode="json") for group in fold_groups(list(survey.reviews), standings)]
    assert [group.model_dump(mode="json") for group in survey.groups] == recomputed
    assert len(survey.groups) == 1
    assert [row["kind"] for row in _feed_rows(tmp_path)] == ["STAGE_START", "STAGE_FINISH"]
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    # The output root is write-once: recomposing different content into it must refuse.
    other = dataclasses.replace(inputs, release_id="testrelease-next")
    with pytest.raises(ValueError, match="use a new output root"):
        compose_stage.run_identity_backend_compose(other, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path))
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_inputs_validate_refuses_target_count_mismatch(tmp_path):
    inputs, _reader = _build(tmp_path)
    with pytest.raises(ValueError, match="expected 3"):
        dataclasses.replace(inputs, expected_target_count=3).validate()


def test_refuses_digest_mismatch(tmp_path):
    inputs, reader = _build(tmp_path)
    (tmp_path / "prior-backend.json").write_text(json.dumps({"drifted": True}))
    with pytest.raises(ValueError, match="digest mismatch"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_refuses_fresh_survey_source_mismatch(tmp_path):
    inputs, reader = _build(tmp_path, fresh_backend=_backend_survey([_measured_source(TARGETS[0])]))
    with pytest.raises(ValueError, match="disagree"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_refuses_fresh_schema_drift(tmp_path):
    fresh = _backend_survey([_measured_source(source_id) for source_id in TARGETS], schema_sha256="f" * 64)
    inputs, reader = _build(tmp_path, fresh_backend=fresh)
    with pytest.raises(ValueError, match="schema"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_refuses_runtime_defaults_drift(tmp_path):
    profile = _runtime_profile().model_copy(update={"cpus": 2})
    fresh = _backend_survey(
        [_measured_source(source_id) for source_id in TARGETS],
        runtime_defaults={"agent": profile, "grader": profile},
    )
    inputs, reader = _build(tmp_path, fresh_backend=fresh)
    with pytest.raises(ValueError, match="runtime_defaults"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_replaces_backend_and_identity_without_reusing_old_population_reviews(tmp_path):
    prior_backend = _backend_survey(
        [
            *[_measured_source(source_id) for source_id in PRIOR_IDS],
            _measured_source(TARGETS[0], task_count=7),
        ]
    )
    fingerprints = {source_id: _fingerprint(source_id) for source_id in PRIOR_IDS}
    fingerprints[TARGETS[0]] = _fingerprint(TARGETS[0], task_count=7)
    fingerprints[TARGETS[0]] = fingerprints[TARGETS[0]].model_copy(update={"inputs": {"tasks": "9" * 64}})
    stale = _review(PRIOR_IDS[0], TARGETS[0], fingerprints[PRIOR_IDS[0]], fingerprints[TARGETS[0]])
    retained = _review(PRIOR_IDS[1], PRIOR_IDS[2], fingerprints[PRIOR_IDS[1]], fingerprints[PRIOR_IDS[2]])
    fresh = _review(
        PRIOR_IDS[0],
        TARGETS[0],
        fingerprints[PRIOR_IDS[0]],
        _fingerprint(TARGETS[0]),
        label=DuplicateLabel.PARTIAL_OVERLAP,
    )
    between_targets = _review(TARGETS[0], TARGETS[1], _fingerprint(TARGETS[0]), _fingerprint(TARGETS[1]))
    inputs, reader = _build(
        tmp_path,
        prior_backend=prior_backend,
        prior_identity=_identity_survey(fingerprints, [stale, retained]),
        registry_reviews=[fresh, between_targets],
    )
    before = Path(inputs.prior_identity.uri).read_bytes()
    proof = compose_stage.run_identity_backend_compose(
        inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    backend = json.loads(Path(proof["documents"]["backend_map"]["uri"]).read_text())
    identity = json.loads(Path(proof["documents"]["identity_survey"]["uri"]).read_text())
    assert {row["source_id"]: row["task_count"] for row in backend["sources"]} == {
        source_id: 3 for source_id in (*PRIOR_IDS, *TARGETS)
    }
    assert identity["fingerprints"][TARGETS[0]]["task_count"] == 3
    assert identity["fingerprints"][PRIOR_IDS[0]] == fingerprints[PRIOR_IDS[0]].model_dump(mode="json")
    assert identity["reviews"] == [row.model_dump(mode="json") for row in (retained, fresh, between_targets)]
    assert Path(inputs.prior_identity.uri).read_bytes() == before
    assert proof["identity"]["superseded_review_count"] == 1
    assert proof["identity"]["retained_review_count"] == 1
    assert proof["identity"]["pair_reviews_touching_targets"] == 2


@pytest.mark.parametrize("stale_side", ["left_inputs_sha256", "right_inputs_sha256"])
def test_a_fresh_target_review_must_measure_both_composed_populations(tmp_path, stale_side):
    review = _review(PRIOR_IDS[0], TARGETS[0], _fingerprint(PRIOR_IDS[0]), _fingerprint(TARGETS[0]))
    assert review.measurement is not None
    review = review.model_copy(update={"measurement": review.measurement.model_copy(update={stale_side: "9" * 64})})
    inputs, reader = _build(tmp_path, registry_reviews=[review])
    with pytest.raises(ValueError, match="fresh review measurement inputs disagree"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path))
    assert not (tmp_path / "release" / "source-identities.json").exists()


def test_conflicting_target_review_documents_cannot_be_composed(tmp_path):
    inputs, _ = _build(tmp_path)
    review = _review(TARGETS[0], TARGETS[1], _fingerprint(TARGETS[0]), _fingerprint(TARGETS[1]))
    other = _review(
        TARGETS[0], TARGETS[1], _fingerprint(TARGETS[0]), _fingerprint(TARGETS[1]), label=DuplicateLabel.PARTIAL_OVERLAP
    )
    sources = {}
    for source_id, row in zip(TARGETS, [review, other], strict=True):
        sources[source_id] = _write_json(
            tmp_path,
            f"{source_id}.json",
            {
                "fingerprints": {source_id: _fingerprint(source_id).model_dump(mode="json")},
                "reviews": [row.model_dump(mode="json")],
            },
        ).model_dump(mode="json")
    inputs = dataclasses.replace(
        inputs,
        registry_index=_write_json(
            tmp_path, "conflicting-index.json", {"revision": REGISTRY_REVISION, "sources": sources}
        ),
    )
    with pytest.raises(ValueError, match="conflicting fresh pair reviews"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=False, feeds=_feeds(tmp_path))
    assert not (tmp_path / "release" / "source-identities.json").exists()


def test_refuses_registry_fingerprint_with_no_tasks(tmp_path):
    registry = {source_id: _fingerprint(source_id, task_count=0) for source_id in TARGETS}
    inputs, reader = _build(tmp_path, registry_fingerprints=registry)
    with pytest.raises(ValueError, match="measures 0 tasks"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


def test_refuses_registry_index_missing_a_target(tmp_path):
    registry = {TARGETS[0]: _fingerprint(TARGETS[0])}
    inputs, reader = _build(tmp_path, registry_fingerprints=registry)
    with pytest.raises(ValueError, match="holds no entry"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))


@pytest.mark.parametrize("duplicate_alias", [False, True])
def test_storage_fingerprint_uses_canonical_source_identity(tmp_path, duplicate_alias):
    canonical = "TaskTrove:SWE-Gym__SWE-Gym"
    original = "Task Trove:SWE-Gym__SWE-Gym"
    inputs, _ = _build(tmp_path, targets=(canonical,))
    fingerprint = _fingerprint(canonical)
    rows = {original: fingerprint.model_dump(mode="json")}
    if duplicate_alias:
        rows[canonical] = fingerprint.model_dump(mode="json")
    review = _review(original, PRIOR_IDS[0], fingerprint, _fingerprint(PRIOR_IDS[0]))
    unrelated = _review(PRIOR_IDS[1], PRIOR_IDS[2], _fingerprint(PRIOR_IDS[1]), _fingerprint(PRIOR_IDS[2]))
    survey_ref = _write_json(
        tmp_path,
        "storage-survey.json",
        {"fingerprints": rows, "reviews": [row.model_dump(mode="json") for row in (review, unrelated)]},
    )
    index = {"revision": REGISTRY_REVISION, "sources": {canonical: survey_ref.model_dump(mode="json")}}
    inputs = dataclasses.replace(inputs, registry_index=_write_json(tmp_path, "storage-index.json", index))
    if duplicate_alias:
        with pytest.raises(ValueError, match="Duplicate aliased source identity"):
            compose_stage.run_identity_backend_compose(inputs, dry_run=False, feeds=_feeds(tmp_path))
        assert not (tmp_path / "release" / "source-identities.json").exists()
        return
    compose_stage.run_identity_backend_compose(inputs, dry_run=False, feeds=_feeds(tmp_path))
    composed = json.loads((tmp_path / "release" / "source-identities.json").read_text())
    assert composed["fingerprints"][canonical] == fingerprint.model_dump(mode="json")
    assert original not in composed["fingerprints"]
    assert composed["reviews"][-1]["similarity"] == {**review.similarity.model_dump(mode="json"), "left": canonical}
    assert len(composed["reviews"]) == 3


def test_failing_run_emits_a_fail_row(tmp_path):
    fresh = _backend_survey([_measured_source(source_id) for source_id in TARGETS], schema_sha256="f" * 64)
    inputs, reader = _build(tmp_path, fresh_backend=fresh)
    with pytest.raises(ValueError, match="schema"):
        compose_stage.run_identity_backend_compose(inputs, dry_run=True, registry_reader=reader, feeds=_feeds(tmp_path))
    rows = _feed_rows(tmp_path)
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FAIL"]
    assert rows[-1]["state"] == "FAILED"
    assert "schema" in rows[-1]["evidence"]["error"]


def test_registry_index_reads_the_bundled_file_by_digest(monkeypatch):
    """The registry index travels as a digest, not a path: the compose stage reads the
    bundled repo-relative file and verifies the plan pin's sha256 — a deriving
    machine's absolute path must never become a stage's read path (the v45 child
    remote_job-885fdbf2 refused on exactly that)."""
    payload = (compose_stage._REPO_ROOT / compose_stage.REGISTRY_INDEX_RELPATH).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    index = compose_stage.read_bundled_registry_index(
        ArtifactRef(uri="experiments/ignored/by/the/reader.json", sha256=digest)
    )
    assert isinstance(index.get("revision"), str) and index["revision"]

    # a pin that does not match the bundled bytes reads through the storage layer
    sentinel = {"revision": "fixture-revision"}
    monkeypatch.setattr(compose_stage, "read_ref", lambda pin: sentinel)
    assert (
        compose_stage.read_bundled_registry_index(ArtifactRef(uri="s3://bucket/fixture-index.json", sha256="0" * 64))
        is sentinel
    )


def test_a_fresh_review_without_a_measured_overlap_composes(tmp_path):
    """The registry's identity documents carry judge verdicts without the content-nn
    overlap record — no identity document in the system attaches a measurement (the
    served survey holds none; the admissions' documents hold none) — so an unmeasured
    fresh review composes like a retained one, its provenance being its digest-pinned
    registry document. A review that DOES carry a measurement still binds both sides'
    fingerprint inputs."""
    unmeasured = _review(PRIOR_IDS[0], TARGETS[0], _fingerprint(PRIOR_IDS[0]), _fingerprint(TARGETS[0]))
    unmeasured = unmeasured.model_copy(update={"measurement": None})
    inputs, reader = _build(tmp_path, registry_reviews=[unmeasured])
    proof = compose_stage.run_identity_backend_compose(
        inputs, dry_run=False, registry_reader=reader, feeds=_feeds(tmp_path)
    )
    identity = json.loads(Path(proof["documents"]["identity_survey"]["uri"]).read_text())
    composed = [DuplicateReview.model_validate(row) for row in identity["reviews"]]
    assert unmeasured in composed
    assert all(row.measurement is not None for row in composed if row != unmeasured)
