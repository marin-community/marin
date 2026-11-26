# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-shaped sealed documents for the strict audit and the hosted check: declarations,
backend survey and identity survey for two fixture sources."""

from finestore.rl.curation import QualifiedSourceDeclaration, SourceOrigin, source_display_name, source_identifier
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, NetworkPolicy, UnsupportedMachineSpec
from taskcompendium.backend_requirements import BackendFit, BackendSourceMap, BackendSurvey, FitSupport, RuntimeProfile
from taskcompendium.source_fingerprints import DuplicateGroup, DuplicateGroupMember, SourceIdentitySurvey

from experiments.post_training.task_curation.backend_capabilities import production_matrix
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

SOURCE_A = "Marin:fixture-a"
SOURCE_B = "Marin:fixture-b"
SOURCE_IDS = (SOURCE_A, SOURCE_B)
REVISION = "c" * 40
SNAPSHOT_ID = "assembly-test-v1"
RUNTIME = RuntimeProfile(
    network=NetworkPolicy.DENY,
    cpus=1,
    memory_mb=512,
    storage_mb=1024,
    gpus=0,
    user=None,
    startup_timeout=60.0,
    cleanup_timeout=30.0,
)
BACKEND_VALUES = frozenset(backend.value for backend in Backend) | {"iris"}


def fixture_declaration(identifier: str) -> QualifiedSourceDeclaration:
    return QualifiedSourceDeclaration(
        id=(source_id := source_identifier(SourceOrigin.MARIN, identifier, None, None)),
        display_name=source_display_name(source_id),
        origin="Marin",
        build_path="fixture",
        upstream_dataset=identifier,
        upstream_name=identifier,
        current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=2,
        within_source_dedup=deduplication_evidence("fixture", "1", "1", count=2),
        notes="",
        dataset_id="fixture",
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
    )


def declarations_document() -> dict[str, dict]:
    return {source_id: fixture_declaration(source_id.split(":")[1]).model_dump(mode="json") for source_id in SOURCE_IDS}


def _rejecting(requirements, spec) -> None:
    raise UnsupportedMachineSpec("The unit lane installs no optional backends")


def _checkers() -> dict:
    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "docker-gvisor": _rejecting,
        "qemu": _rejecting,
        "shellsim": _rejecting,
        "daytona": _rejecting,
        "iris": _rejecting,
    }


def _fit(support: FitSupport) -> BackendFit:
    counts = {support: 2}
    return BackendFit(
        support=support,
        task_counts=counts,
        roles={"agent": dict(counts), "grader": dict(counts)},
        blockers={},
        validation_errors={},
    )


def backend_row(source_id: str) -> BackendSourceMap:
    fits = {backend: _fit(FitSupport.SUPPORTED) for backend in BACKEND_VALUES}
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=2,
        inputs={},
        demands={"agent": {}, "grader": {}},
        resource_ranges={"agent": {}, "grader": {}},
        backends=fits,
        role_backend_assignments={"agent": {}, "grader": {}},
        grading_environment=None,
    )


def backend_survey(git_revision: str = REVISION) -> BackendSurvey:
    return BackendSurvey(
        catalog_root="test:catalog",
        snapshot_id=SNAPSHOT_ID,
        generated_at="2026-11-23T00:00:00Z",
        git_revision=git_revision,
        schema_sha256="0" * 64,
        runtime_defaults={"agent": RUNTIME, "grader": RUNTIME},
        matrix=production_matrix(_checkers()),
        deployment_limits={},
        deployment_classes={},
        sources=[backend_row(source_id) for source_id in SOURCE_IDS],
    )


def identity_survey() -> SourceIdentitySurvey:
    group = DuplicateGroup(
        group_id="fixture-duplicate-group",
        canonical_source_id=SOURCE_A,
        members=[
            DuplicateGroupMember(source_id=SOURCE_A, overlap_with_canonical=1.0),
            DuplicateGroupMember(source_id=SOURCE_B, overlap_with_canonical=None),
        ],
    )
    return SourceIdentitySurvey(
        catalog_root="test:catalog",
        snapshot_id=SNAPSHOT_ID,
        generated_at="2026-11-23T00:00:00Z",
        fingerprints={source_id: None for source_id in SOURCE_IDS},
        reviews=[],
        groups=[group],
        judge=None,
    )
