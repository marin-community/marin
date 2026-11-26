# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the derived release plan (R41 brief).

The fixtures build a real frozen layer — real population artifacts, real
census-binding chains, real digest-verified prior pins — and run the real
derivation, so the tests exercise the same code path the release runs. The
applet SQL, the prior catalog, the contract receipts and the freeze revision are
the injected boundaries (the release resolves the real ones at run time).
"""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput, QualifiedSourceDeclaration, SourceOrigin, source_identifier
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.backend_requirements import backend_schema_sha256

from experiments.post_training.task_curation import atlas_deployment
from experiments.post_training.task_curation.grader_drift import (
    GraderDriftReference,
    MainRecipeReference,
    SealedRecipeReference,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.release_stages import declarations as declarations_stage
from experiments.post_training.task_curation.release_stages import plan_derivation
from experiments.post_training.task_curation.release_stages import (
    snapshot_publication as snapshot_publication_stage,
)
from experiments.post_training.task_curation.release_stages.graph import (
    ReleasePlan,
    graph_nodes,
    release_graph,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.plan_derivation import (
    derive_release_plan,
    read_prior_catalog_state,
    seal_derivation,
)
from experiments.post_training.task_curation.release_stages.preflight_slices import (
    SliceSelection,
    _snapshot_publication_witness,
)
from experiments.post_training.task_curation.release_stages.release_config import (
    ExcludedSource,
    PriorRelease,
    ReleaseConfig,
    RemovedSource,
)
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

CLUSTER = "derivation-test-region"
RETAINED_ID = "Marin:fixture"
TARGET_ID = "Marin:fixture-target"
LATE_ID = "Marin:late-admission"
REMOVED_ID = "Marin:removed-fixture"
PRIOR_LOAD_ID = "b" * 32
PRIOR_CURRICULUM_LOAD_ID = "2" * 32
FREEZE_REVISION = "a" * 40
FREEZE_ARTIFACT_VERSION = "2026.11.25.1010"
HEAD = "d" * 40
REPO_ROOT = Path(plan_derivation.__file__).parents[4]


def _task_spec(name: str, index: int) -> dict:
    """A minimal task-spec document; the derivation only counts population rows."""
    return {"task_id": f"{name}-{index}"}


def _write_population(tmp_path: Path, source_id: str, name: str, task_count: int) -> str:
    """A frozen population artifact over ``task_count`` tasks; returns its directory.

    The artifacts live under the deployment's region prefix, as real populations do.
    """
    population_dir = tmp_path / "region" / "populations" / name
    specs_dir = tmp_path / "region" / "specs" / name
    specs_dir.mkdir(parents=True, exist_ok=True)
    population_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"task_spec": json.dumps(_task_spec(name, i)), "runtime": "{}"} for i in range(task_count)]
    pq.write_table(pa.Table.from_pylist(rows), specs_dir / "part-00000.parquet", compression="zstd")
    population = SourceTextPopulation(
        path=str(population_dir),
        source_id=source_id,
        dataset_id=name,
        dataset_revision="1",
        source_version="1",
        converter_revision="1",
        text_renderer_version="1",
        task_count=task_count,
        text_sha256="e" * 64,
        task_specs_sha256="f" * 64,
        main_output_dir=str(tmp_path / "main" / name),
        specs_output_dir=str(specs_dir),
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=str(population_dir),
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return str(population_dir)


def _declaration(source_id: str, population_dir: str) -> QualifiedSourceDeclaration:
    short = source_id.removeprefix("Marin:")
    return QualifiedSourceDeclaration(
        id=source_identifier(SourceOrigin.MARIN, short, None, None),
        display_name=f"Marin · {short}",
        origin="Marin",
        build_path=short,
        upstream_dataset=short,
        upstream_name=short,
        current_release={"repo": short, "revision": "1", "url": "https://example.org/pin"},
        family=short,
        tags=[],
        task_count=2,
        within_source_dedup=deduplication_evidence(short, "1", "1", count=2),
        notes="",
        dataset_id=short,
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": short, "version": "1", "source": short, "revision": "1", "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
        actor_census_input=ActorCensusInput(
            dataset_id=short,
            dataset_revision="1",
            converter_version="1",
            converter_revision="1",
            measured_tasks=2,
            task_specs_sha256="f" * 64,
            evidence_url=str(Path(population_dir) / ".artifact.json"),
        ),
    )


def _source_layer(tmp_path: Path, source_ids: tuple[str, ...]) -> dict:
    """The fixture sources layer: canonical-id keyed sources with real populations."""
    sources = {}
    for source_id in source_ids:
        population_dir = _write_population(tmp_path, source_id, source_id.removeprefix("Marin:"), task_count=2)
        sources[source_id] = SimpleNamespace(
            info=SimpleNamespace(id=source_id, oracle_filtration=None),
            declaration=_declaration(source_id, population_dir),
        )
    return sources


def _declaration_dump(layer: dict) -> dict:
    return {source_id: source.declaration.model_dump(mode="json") for source_id, source in layer.items()}


def _prior_pins(tmp_path: Path, declared: dict) -> PriorRelease:
    """Digest-bound prior-state pins over local fixture documents."""

    def pin(name: str, document) -> dict:
        path = tmp_path / "prior" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, sort_keys=True))
        return {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return PriorRelease(
        release_id="priorrelease",
        declarations=ArtifactRef(**pin("declarations", declared)),
        identity=ArtifactRef(**pin("identity", {"fingerprints": {}, "reviews": []})),
        backend=ArtifactRef(**pin("backend", {"sources": [], "snapshot_id": "assembly-prior-v1"})),
        composed_release=ArtifactRef(**pin("composed_release", {"release_id": "priorrelease"})),
        composed_coverage=ArtifactRef(**pin("composed_coverage", {"coverage": {}})),
        oracle_inventory=ArtifactRef(**pin("oracle_inventory", {"sources": 2})),
        source_counts=ArtifactRef(**pin("source_counts", {"sources": 2})),
    )


def _applet_query(prior_load_id: str, curriculum_load_id: str, panels: list[str], available: list[str]):
    """The applet SQL the prior-state reads run, faked at the row level."""

    def query(sql: str, parameters: dict) -> dict:
        if "FROM atlas_current" in sql:
            return {"rows": [{"load_id": prior_load_id}]}
        if "FROM curriculum_current" in sql:
            return {"rows": [{"load_id": curriculum_load_id}]}
        if "FROM atlas_review_current" in sql:
            after = parameters.get("after")
            rows = [{"source_id": source_id} for source_id in sorted(panels) if after is None or source_id > after][:500]
            return {"rows": rows}
        if "FROM atlas_sources" in sql:
            return {"rows": [{"source_id": source_id} for source_id in sorted(available)]}
        raise AssertionError(f"unexpected derivation query: {sql}")

    return query


def _catalog_factory(source_ids: tuple[str, ...]):
    catalog = SimpleNamespace(
        current=lambda: SimpleNamespace(snapshot_id="assembly-prior-v1"),
        sources=lambda snapshot_id, pending_stage_of=None: [SimpleNamespace(id=source_id) for source_id in source_ids],
    )
    return lambda root: catalog


@pytest.fixture
def frozen_layer(tmp_path, monkeypatch):
    """The fixture frozen layer: real populations, injected boundaries."""
    monkeypatch.setattr(declarations_stage, "require_catalog_declaration", lambda declaration: None)
    monkeypatch.setattr(plan_derivation, "_verify_freeze", lambda repo_root, revision: HEAD)
    monkeypatch.setattr(plan_derivation, "_freeze_artifact_version", lambda repo_root, revision: FREEZE_ARTIFACT_VERSION)
    registry_index = tmp_path / "registry-index.json"
    registry_index.write_text(
        json.dumps({"revision": "registryrev", "sources": {TARGET_ID: {"repo": "r"}, LATE_ID: {"repo": "r"}}})
    )
    monkeypatch.setattr(plan_derivation, "REGISTRY_INDEX_PATH", registry_index)
    return tmp_path


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    deployment = atlas_deployment.RegionDeployment(
        cluster=CLUSTER,
        region_label="alpha",
        region_prefix=f"{tmp_path}/region/",
        catalog_root=f"{tmp_path}/catalog/production",
    )
    monkeypatch.setitem(atlas_deployment.REGION_DEPLOYMENTS, CLUSTER, deployment)
    return deployment


def _config(prior: PriorRelease, **overrides) -> ReleaseConfig:
    defaults = dict(
        release_id="testrelease",
        cluster=CLUSTER,
        freeze_source_revision=FREEZE_REVISION,
        prior_release=prior,
        production_version=41,
        preview_base_version=30,
    )
    defaults.update(overrides)
    return ReleaseConfig(**defaults)


def _derive(
    tmp_path: Path,
    layer: dict,
    config: ReleaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    panels: list[str],
    available: list[str],
    catalog_source_ids: tuple[str, ...] = (RETAINED_ID,),
    contract_tasks: dict[str, int] | None = None,
) -> plan_derivation.DerivedRelease:
    """Run the real derivation over the fixture layer with injected boundaries."""
    receipts = contract_tasks if contract_tasks is not None else {TARGET_ID: 2}
    monkeypatch.setattr(plan_derivation, "derived_contract_receipts", _contract_receipts_factory(receipts))
    return derive_release_plan(
        config,
        repo_root=REPO_ROOT,
        query=_applet_query(PRIOR_LOAD_ID, PRIOR_CURRICULUM_LOAD_ID, panels, available),
        sources_fn=lambda: layer,
        drift_reference_fn=lambda root, revision, declared: GraderDriftReference(
            main=MainRecipeReference(revision="e" * 40, merge_base=revision),
            sources={
                source_id: SealedRecipeReference(source_id=source_id, converter_revision=revision)
                for source_id in declared
            },
        ),
        prior_catalog_reader=lambda root, q: read_prior_catalog_state(
            root,
            q,
            catalog_factory=_catalog_factory(catalog_source_ids),
            receipt_keys_fn=lambda: {},
        ),
    )


def _contract_receipts_factory(measured_tasks: dict[str, int]):
    receipts = {
        source_id: SimpleNamespace(input=SimpleNamespace(measured_tasks=tasks))
        for source_id, tasks in measured_tasks.items()
    }
    return lambda: receipts


def test_derive_release_plan_from_fixture_state(tmp_path, frozen_layer, deployment, monkeypatch):
    """Every expected_* is measured from the fixture state; nothing is hand-typed."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    prior = _prior_pins(tmp_path, _declaration_dump(layer))
    config = _config(prior)
    derived = _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])

    plan = ReleasePlan.model_validate(derived.plan_document)
    plan.validate_shape()
    measured = derived.receipt["measured"]
    assert measured["declared_source_count"] == 2
    assert measured["prior_source_count"] == 1
    assert plan.targets == (TARGET_ID,)
    assert plan.expected_source_count == 2
    assert plan.expected_prior_sources == 1
    assert plan.expected_population_tasks == 4
    assert plan.populations == {
        RETAINED_ID: str(tmp_path / "region" / "populations" / "fixture"),
        TARGET_ID: str(tmp_path / "region" / "populations" / "fixture-target"),
    }
    assert plan.expected_panels == 1 and plan.expected_browse_source_count == 1
    assert plan.expected_selected_panels == 1
    assert plan.expected_contract_count == 1 and plan.expected_contract_tasks == 2
    assert plan.prior.load_id == PRIOR_LOAD_ID
    assert plan.prior_curriculum_load_id == PRIOR_CURRICULUM_LOAD_ID
    assert plan.prior_snapshot_id == "assembly-prior-v1"
    assert plan.previously_available == (RETAINED_ID,)
    assert plan.qc_exclusions == ()
    assert plan.artifact_version == FREEZE_ARTIFACT_VERSION
    assert plan.release_root == f"{deployment.region_prefix}preview-release/testrelease-{FREEZE_ARTIFACT_VERSION}"
    assert plan.catalog_target_load_id != plan.curriculum_target_load_id
    assert len(plan.catalog_target_load_id) == 32
    assert plan.catalog_target_load_id == _rederived_catalog_load_id(plan)
    # The measured dependency is computed from the real requirement module.
    module = REPO_ROOT / plan.measured.module
    assert module.name == "models.py"
    assert plan.measured.schema_sha256 == backend_schema_sha256()
    assert plan.measured.sha256 == hashlib.sha256(module.read_bytes()).hexdigest()
    assert plan.measured.revision == FREEZE_REVISION
    assert plan.validation_integrity_contracts == {}


def _rederived_catalog_load_id(plan: ReleasePlan) -> str:
    identity = {
        "release_id": plan.release_id,
        "snapshot_id": plan.snapshot_id,
        "prior_load_id": plan.prior.load_id,
        "artifact_version": plan.artifact_version,
        "source_revision": plan.source_revision,
    }
    document = {"kind": "atlas-catalog-target-load", **identity}
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:32]


def test_derivation_refuses_unresolvable_populations(tmp_path, frozen_layer, deployment, monkeypatch):
    """A declared source without census evidence fails the derivation, itemized."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    broken_input = ActorCensusInput(
        dataset_id="x",
        dataset_revision="1",
        converter_version="1",
        converter_revision="1",
        measured_tasks=2,
        task_specs_sha256="f" * 64,
        evidence_url=str(tmp_path / "absent" / ".artifact.json"),
    )
    layer[RETAINED_ID].declaration = layer[RETAINED_ID].declaration.model_copy(
        update={"actor_census_input": broken_input}
    )
    config = _config(_prior_pins(tmp_path, _declaration_dump(layer)))
    with pytest.raises(ValueError, match="no resolvable frozen population"):
        _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])


def test_derivation_refuses_prior_drift(tmp_path, frozen_layer, deployment, monkeypatch):
    """A drifted retained declaration fails the derivation before any stage runs."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    declared = _declaration_dump(layer)
    drifted = json.loads(json.dumps(declared[RETAINED_ID]))
    drifted["dataset_revision"] = "2"
    prior = _prior_pins(tmp_path, {RETAINED_ID: drifted, TARGET_ID: declared[TARGET_ID]})
    with pytest.raises(ValueError, match="drifted from the prior release"):
        _derive(tmp_path, layer, _config(prior), monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])


def test_derivation_applies_config_policy(tmp_path, frozen_layer, deployment, monkeypatch):
    """Removals and QC exclusions ride from the config into the derived plan.

    The removed source is already staged out of the frozen layer (the pending
    mechanism): it stays in the prior state, is named in removed_sources, and its
    QC exclusion drops the prior panel.
    """
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    prior = _prior_pins(tmp_path, _declaration_dump(layer))
    config = _config(
        prior,
        removed_sources=(RemovedSource(source_id=REMOVED_ID, reason="staged for intake"),),
        qc_exclusions=(ExcludedSource(source_id=REMOVED_ID, reason="its rebuilt panel is stale"),),
    )
    derived = _derive(
        tmp_path,
        layer,
        config,
        monkeypatch,
        panels=[RETAINED_ID, REMOVED_ID],
        available=[RETAINED_ID],
        catalog_source_ids=(RETAINED_ID, REMOVED_ID),
    )
    plan = ReleasePlan.model_validate(derived.plan_document)
    plan.validate_shape()
    assert plan.removed_sources == (REMOVED_ID,)
    assert REMOVED_ID not in plan.populations
    assert plan.targets == (TARGET_ID,)
    assert plan.qc_exclusions == (REMOVED_ID,)
    assert plan.expected_panels == 1 and plan.expected_browse_source_count == 1


def test_derivation_refuses_a_removal_the_layer_still_declares(tmp_path, frozen_layer, deployment, monkeypatch):
    """A removal the frozen layer still carries refuses with the itemized list."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    config = _config(
        _prior_pins(tmp_path, _declaration_dump(layer)),
        removed_sources=(RemovedSource(source_id=RETAINED_ID, reason="fixture"),),
    )
    with pytest.raises(ValueError, match="planned removals"):
        _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])


def test_mid_cut_admission_does_not_move_the_frozen_plan(tmp_path, frozen_layer, deployment, monkeypatch):
    """The sealed plan survives a post-freeze admission unchanged and refuses a moved one."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    config = _config(_prior_pins(tmp_path, _declaration_dump(layer)))
    derived = _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])
    release_root = derived.plan_document["release_root"]
    seal_derivation(release_root, derived)
    sealed = (Path(release_root) / "release-plan.json").read_bytes()

    # A resume re-derives at the same freeze: byte-identical, the seal is idempotent.
    derived_again = _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])
    assert derived_again.plan_document == derived.plan_document
    seal_derivation(release_root, derived_again)
    assert (Path(release_root) / "release-plan.json").read_bytes() == sealed

    # A mid-cut admission lands on the branch after the freeze: the frozen layer the
    # derivation reads never contains it, and a deliberately re-derived plan cannot
    # silently replace the sealed one.
    late_layer = dict(layer)
    late_layer[LATE_ID] = _source_layer(tmp_path, (LATE_ID,))[LATE_ID]
    late_config = _config(_prior_pins(tmp_path, _declaration_dump(late_layer)))
    late_derived = _derive(
        tmp_path,
        late_layer,
        late_config,
        monkeypatch,
        panels=[RETAINED_ID],
        available=[RETAINED_ID],
        catalog_source_ids=(RETAINED_ID,),
        contract_tasks={TARGET_ID: 2, LATE_ID: 2},
    )
    assert LATE_ID in late_derived.plan_document["populations"]
    assert LATE_ID not in derived.plan_document["populations"]
    with pytest.raises(ValueError, match="different content"):
        seal_derivation(release_root, late_derived)
    assert (Path(release_root) / "release-plan.json").read_bytes() == sealed


def test_interrupted_derivation_resumes_at_the_same_version(tmp_path, frozen_layer, deployment, monkeypatch):
    """An interrupted release resumes: same sealed plan, same graph fingerprints."""
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    config = _config(_prior_pins(tmp_path, _declaration_dump(layer)))
    derived = _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])
    release_root = derived.plan_document["release_root"]
    seal_derivation(release_root, derived)
    sealed = (Path(release_root) / "release-plan.json").read_bytes()
    plan = ReleasePlan.model_validate(derived.plan_document)
    fingerprints_before = {node["name"]: node["fingerprint"] for node in graph_nodes(release_graph(plan))}

    # The interruption sealed nothing beyond the plan; the resume re-derives, and
    # every graph fingerprint matches so the runner skips sealed nodes instead of
    # rerunning them.
    derived_again = _derive(tmp_path, layer, config, monkeypatch, panels=[RETAINED_ID], available=[RETAINED_ID])
    seal_derivation(release_root, derived_again)
    resumed_plan = ReleasePlan.model_validate(derived_again.plan_document)
    fingerprints_after = {node["name"]: node["fingerprint"] for node in graph_nodes(release_graph(resumed_plan))}
    assert fingerprints_before == fingerprints_after
    assert (Path(release_root) / "release-plan.json").read_bytes() == sealed
    assert (Path(release_root) / "derivation-receipt.json").exists()


def test_read_prior_catalog_state_measurements(tmp_path, frozen_layer, deployment):
    """The prior-state reader measures the load, curriculum, panels and availability."""
    query = _applet_query(
        PRIOR_LOAD_ID,
        PRIOR_CURRICULUM_LOAD_ID,
        panels=[RETAINED_ID, TARGET_ID],
        available=[RETAINED_ID],
    )
    state = read_prior_catalog_state(
        deployment.catalog_root,
        query,
        catalog_factory=_catalog_factory((RETAINED_ID, TARGET_ID)),
        receipt_keys_fn=lambda: {},
    )
    assert state.snapshot_id == "assembly-prior-v1"
    assert state.source_ids == (RETAINED_ID, TARGET_ID)
    assert state.load_id == PRIOR_LOAD_ID
    assert state.curriculum_load_id == PRIOR_CURRICULUM_LOAD_ID
    assert state.review_panel_source_ids == (RETAINED_ID, TARGET_ID)
    assert state.available_source_ids == (RETAINED_ID,)
    assert state.snapshot_uri.endswith("w=assembly-prior-v1-catalog")


def test_read_prior_catalog_state_refuses_a_moving_pointer(tmp_path, frozen_layer, deployment):
    """A catalog pointer without a selected load refuses."""
    query = _applet_query("", PRIOR_CURRICULUM_LOAD_ID, panels=[], available=[])
    with pytest.raises(ValueError, match="no selected load"):
        read_prior_catalog_state(
            deployment.catalog_root,
            query,
            catalog_factory=_catalog_factory((RETAINED_ID,)),
            receipt_keys_fn=lambda: {},
        )


def test_snapshot_publication_witness_runs_the_real_check_path(tmp_path, frozen_layer, deployment, monkeypatch):
    """The prior witness runs snapshot publication's real pre-upstream check path.

    It verifies the repo state (the git-less-bundle bug class), extracts the
    current declarations and checks prior drift against the digest-verified pin
    — and it refuses drifted priors, the failure the release would hit later.
    """
    layer = _source_layer(tmp_path, (RETAINED_ID, TARGET_ID))
    monkeypatch.setattr(snapshot_publication_stage, "all_sources", lambda: layer)
    monkeypatch.setattr(snapshot_publication_stage, "require_catalog_declaration", lambda declaration: None)
    plan = fixture_plan(release_root=str(tmp_path / "release"), slice_output_root=str(tmp_path / "slice"))
    plan = plan.model_copy(
        update={
            # verify_repo_state is the real path: the pinned revision must be a
            # revision the deriving checkout actually contains.
            "source_revision": (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
            "targets": (TARGET_ID,),
            "expected_source_count": 2,
            "pins": plan.pins.model_copy(
                update={"prior_declarations": _prior_pins(tmp_path, _declaration_dump(layer)).declarations}
            ),
        }
    )
    evidence = _snapshot_publication_witness(plan, SliceSelection(source=RETAINED_ID))
    assert evidence["source_count"] == 2
    assert evidence["witness"] == "check-only against the prior release's sealed state"

    drifted = _declaration_dump(layer)
    drifted[RETAINED_ID] = {**drifted[RETAINED_ID], "dataset_revision": "2"}
    plan = plan.model_copy(
        update={"pins": plan.pins.model_copy(update={"prior_declarations": _prior_pins(tmp_path, drifted).declarations})}
    )
    with pytest.raises(ValueError, match="non-target declaration drift"):
        _snapshot_publication_witness(plan, SliceSelection(source=RETAINED_ID))
