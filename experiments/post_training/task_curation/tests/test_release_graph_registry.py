# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the release stage-graph registry (DESIGN.md §1-§2, M2c).

The graph is built for a fixture plan with no I/O; per-stage typed inputs are
materialized against fake dependency records so the tests pin how receipts flow:
every downstream input comes from ``ctx.resolved`` of its declared dependency.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from fray.types import CpuConfig, GpuConfig
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.lazy import StepContext
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import census as census_stage
from experiments.post_training.task_curation.release_stages import graph as release_graph_module
from experiments.post_training.task_curation.release_stages import release_check
from experiments.post_training.task_curation.release_stages.exposure_probe import exposure_lists
from experiments.post_training.task_curation.release_stages.graph import (
    _STAGE_SPECS,
    GRAPH_RECEIPT_KIND,
    REPO_ROOT,
    ComposeManifestDocument,
    StageReceipt,
    _execute_stage_node,
    _plan_document_data,
    _seal_document,
    _stage_receipt,
    graph_nodes,
    load_plan,
    release_graph,
    typed_inputs,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan
from infra.marina.apps.rl_data_catalog.server.publication_scope import PUBLICATION_DECISIONS

EXPECTED_NODES = (
    "declarations",
    "main_source_sync",
    "main_grader_drift",
    "admission_check",
    "census_inputs",
    "census",
    "corpus_assembly",
    "embed",
    "compose_manifest",
    "curriculum_compose",
    "identity_backend_compose",
    "history_staging",
    "text_scan_region[alpha]",
    "text_scans",
    "backend_validation",
    "fresh_qc_index",
    "qc_browse_plan",
    "qc_browse",
    "snapshot_publication",
    "catalog_stage",
    "catalog_load",
    "curriculum_load",
    "strict_audit_before",
    "strict_audit_after",
    "package_preview",
    "release_check_preflight",
    "release_check_hosted",
    "release_check_postpublish",
    "marina_release",
    "hosted_check",
    "exposure_probe",
)

# The two configuration-document nodes take dash-separated artifact names.
DOC_NODE_DASH_NAMES = {"census_inputs": "census-inputs", "compose_manifest": "compose-manifest"}


def _receipt_ref(tmp_path: Path, name: str, document: dict) -> dict:
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    path = tmp_path / name
    path.write_bytes(raw)
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_dep(handle, prefix: str, stage: str, release_id: str, receipt: dict, **proof_extra) -> None:
    receipt_artifact = StageReceipt(
        stage=stage,
        release_id=release_id,
        receipt=ArtifactRef(**receipt),
        proof={"receipt": receipt, **proof_extra},
    )
    write_record(
        ArtifactRecord(
            output_path=handle.path(prefix),
            result_type=result_type_name(StageReceipt),
            result=receipt_artifact.result_payload(),
        )
    )


def _write_dep_artifact(handle, prefix: str, value) -> None:
    write_record(
        ArtifactRecord(
            output_path=handle.path(prefix),
            result_type=result_type_name(type(value)),
            result=value.result_payload(),
        )
    )


def _materialized_inputs(handle, prefix: str) -> dict:
    ctx = StepContext.for_run(output_path=handle.path(prefix), prefix=prefix, deps=handle.deps)
    return handle.build_config(ctx)


def _declarations_dep(built, prefix: str, plan, receipt: dict):
    _write_dep(built.stages["declarations"], prefix, "declarations", plan.release_id, receipt)


def test_registry_covers_every_stage_and_the_terminal_closure():
    built = release_graph(fixture_plan())
    assert sorted(built.stages) == sorted(EXPECTED_NODES)
    expected_names = {f"atlas-release-testrelease-{DOC_NODE_DASH_NAMES.get(key, key)}" for key in EXPECTED_NODES}
    assert {handle.name for handle in built.stages.values()} == expected_names
    # The lowered graph adds the sealed plan document ahead of every stage.
    assert {node["name"] for node in graph_nodes(built)} == expected_names | {"atlas-release-testrelease-plan"}
    assert built.terminal is built.stages["exposure_probe"]


def test_independent_stages_run_as_siblings():
    """The parallel stages share no edges: the runner schedules them concurrently."""
    built = release_graph(fixture_plan())
    names = {handle.name for handle in built.stages.values()}
    for key in ("identity_backend_compose", "history_staging", "qc_browse_plan", "census"):
        handle = built.stages[key]
        for dep in handle.deps:
            assert dep.name in names
    # census, identity/backend compose and history staging depend on disjoint nodes.
    assert not set(built.stages["census"].deps) & set(built.stages["identity_backend_compose"].deps)
    assert not set(built.stages["census"].deps) & set(built.stages["history_staging"].deps)


def test_fingerprint_configs_are_io_free_identity_payloads():
    """Every stage's fingerprint config reads nothing and carries the plan identity."""
    plan = fixture_plan()
    built = release_graph(plan)
    for key, handle in built.stages.items():
        payload = handle.build_config(StepContext.for_fingerprint())
        assert payload["node"] == key
        assert payload["release_root"] == plan.release_root


def test_expected_fingerprint_pins_are_self_consistent():
    built = release_graph(fixture_plan())
    for handle in built.stages.values():
        assert handle.expected_fingerprint == handle.fingerprint()


def test_cluster_stages_carry_sizing_resources():
    plan = fixture_plan()
    release_graph(plan)
    # Census sizing derives the job resources from the declared population.
    config = release_graph_module._resources_config(release_graph_module.census_sizing(plan.expected_population_tasks))
    assert config.device == CpuConfig()
    assert config.cpu > 0
    # The GPU rule: a parity receipt plus a typed runtime selects the accelerator.
    gpu_plan = fixture_plan(
        pins={
            **fixture_plan().pins.model_dump(mode="json"),
            "parity_receipt": {"uri": "s3://bucket/parity.json", "sha256": "c" * 64},
        },
        gpu_runtime="b200",
    )
    sizing = release_graph_module.embedding_sizing(gpu_plan.expected_population_tasks)
    base = release_graph_module._resources_config(sizing)
    gpu = dataclasses.replace(base, device=GpuConfig(variant="b200"))
    assert gpu.device == GpuConfig(variant="b200")


def test_census_inputs_bind_the_declarations_receipt(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    declarations_receipt = _receipt_ref(tmp_path, "declarations.json", {"Marin:fixture": {}})
    _declarations_dep(built, prefix, plan, declarations_receipt)
    inputs = _materialized_inputs(built.stages["census_inputs"], prefix)
    assert inputs["declarations"] == declarations_receipt
    assert inputs["populations"] == plan.populations
    assert inputs["source_revision"] == plan.source_revision


def test_declarations_step_binds_the_document_consumers_read(tmp_path, monkeypatch):
    """The declarations step's receipt reference is the sealed document itself.

    Every downstream consumer (census inputs, admission check, QC plan, catalog
    staging, exposure) reads the reference as the {source_id: declaration} mapping;
    binding the stage's receipt envelope instead feeds them the wrong document.
    """
    plan = fixture_plan()
    built = release_graph(plan)
    document_ref = {"uri": str(tmp_path / "declarations.json"), "sha256": "c" * 64}
    envelope_ref = {"uri": str(tmp_path / "declarations-receipt.json"), "sha256": "d" * 64}

    def fake_execute(stage, config):
        assert stage == "declarations"
        return {
            "kind": f"atlas-{plan.release_id}-declarations",
            "release_id": plan.release_id,
            "declarations": document_ref,
            "receipt": envelope_ref,
            "dry_run": False,
            "sql_writes": 0,
        }

    monkeypatch.setattr(release_graph_module, "_execute_stage", fake_execute)
    receipt = built.stages["declarations"].run({})
    assert receipt.receipt == ArtifactRef(**document_ref)


def test_stage_step_fingerprints_cover_the_receipt_contract():
    """The receipt key is part of the step's output contract and its fingerprint.

    Regression: the declarations spec's receipt key changed what the step binds,
    but the stored record from an earlier attempt still matched the fingerprint,
    so consumers kept reading the stale envelope reference.
    """
    plan = fixture_plan()
    built = release_graph(plan)
    ctx = StepContext.for_fingerprint()
    config = built.stages["declarations"].build_config(ctx)
    assert config["receipt_key"] == "declarations"


def test_text_scan_regions_read_the_compose_handoff(tmp_path):
    """The scan handoff is the composition handoff document reference.

    The scan region and merge builders read the compose proof's handoff key (the
    pattern curriculum_load uses); the compose proof carries no 'receipt' key, so
    the default receipt extraction could never satisfy this input.
    """
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    handoff = {"uri": "s3://bucket/handoff.json", "sha256": "c" * 64}
    _write_dep(
        built.stages["curriculum_compose"],
        prefix,
        "curriculum_compose",
        plan.release_id,
        handoff,
        handoff=handoff,
    )
    inputs = _materialized_inputs(built.stages["text_scan_region[alpha]"], prefix)
    assert inputs["handoff"] == handoff
    region_receipt = {"uri": "s3://bucket/region-alpha.json", "sha256": "d" * 64}
    _write_dep(
        built.stages["text_scan_region[alpha]"],
        prefix,
        "text_scan_region",
        plan.release_id,
        region_receipt,
    )
    merge_inputs = _materialized_inputs(built.stages["text_scans"], prefix)
    assert merge_inputs["handoff"] == handoff
    assert merge_inputs["region_receipts"] == [region_receipt]


def test_census_inputs_node_seals_one_row_per_declared_source(tmp_path):
    release_root = str(tmp_path / "release")
    plan = fixture_plan(
        release_root=release_root,
        populations={"Marin:fixture": "s3://bucket/populations/fixture", "Marin:other": "s3://bucket/populations/other"},
    )
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    declarations_document = {"Marin:fixture": {"id": "Marin:fixture"}, "Marin:other": {"id": "Marin:other"}}
    declarations_receipt = _receipt_ref(tmp_path, "declarations.json", declarations_document)
    _declarations_dep(built, prefix, plan, declarations_receipt)
    inputs = _materialized_inputs(built.stages["census_inputs"], prefix)
    document = json.loads(StoragePath(built.census_inputs_handle.run(inputs).uri).read_text())
    assert [row["source_id"] for row in document["rows"]] == ["Marin:fixture", "Marin:other"]
    assert document["rows"][0]["population_path"] == "s3://bucket/populations/fixture"
    assert document["kind"] == census_stage.INPUTS_KIND
    with pytest.raises(ValueError, match="pins no population"):
        release_graph_module._seal_census_inputs(
            declarations_document,
            {"Marin:fixture": "s3://bucket/populations/fixture"},
            release_root,
            plan.source_revision,
        )


def test_curriculum_compose_fresh_total_comes_from_the_census_receipt(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    census_results = [
        {"source_id": "Marin:fixture", "inventory": {"retained_tasks": 12}},
        {"source_id": "Marin:other", "inventory": {"retained_tasks": 5}},
    ]
    census_receipt = _receipt_ref(
        tmp_path, "census-receipt.json", {"kind": "census", "tasks": 17, "results": census_results}
    )
    _write_dep(
        built.stages["census"],
        prefix,
        "census",
        plan.release_id,
        census_receipt,
        tasks=17,
        results=census_results,
    )
    manifest = {"uri": "s3://bucket/manifest.json", "sha256": "c" * 64}
    _write_dep_artifact(
        built.compose_manifest_handle,
        prefix,
        ComposeManifestDocument(uri=manifest["uri"], sha256=manifest["sha256"], targets=("Marin:fixture",)),
    )
    inputs = _materialized_inputs(built.stages["curriculum_compose"], prefix)
    assert inputs["expected_fresh_task_count"] == 12
    assert inputs["inputs"] == manifest
    with pytest.raises(ValueError, match="carry no rows for targets"):
        missing = [row for row in census_results if row["source_id"] != "Marin:fixture"]
        census_receipt_missing = _receipt_ref(
            tmp_path, "census-missing.json", {"kind": "census", "tasks": 5, "results": missing}
        )
        built_missing = release_graph(plan)
        _write_dep(
            built_missing.stages["census"],
            prefix,
            "census",
            plan.release_id,
            census_receipt_missing,
            tasks=5,
            results=missing,
        )
        _write_dep_artifact(
            built_missing.compose_manifest_handle,
            prefix,
            ComposeManifestDocument(uri=manifest["uri"], sha256=manifest["sha256"], targets=("Marin:fixture",)),
        )
        _materialized_inputs(built_missing.stages["curriculum_compose"], prefix)


def test_catalog_load_staging_flows_from_the_catalog_stage_proof(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    staged_refs = {
        name: {"uri": f"s3://bucket/{name}.json", "sha256": "c" * 64}
        for name in (
            "declarations",
            "publication",
            "history_receipt",
            "prior_qc_completion",
            "fresh_qc_index",
            "contracts",
        )
    }
    _write_dep(
        built.stages["catalog_stage"],
        prefix,
        "catalog_stage",
        plan.release_id,
        staged_refs["publication"],
        refs=staged_refs,
    )
    inputs = _materialized_inputs(built.stages["catalog_load"], prefix)
    assert inputs["publication"] == staged_refs["publication"]
    assert sorted(inputs["staging"]) == ["contracts", "fresh_qc_index", "history_receipt", "prior_qc_completion"]
    assert inputs["target_load_id"] == plan.catalog_target_load_id
    assert inputs["measured_module"] == str(release_graph_module.REPO_ROOT / plan.measured.module)
    assert inputs["previously_available"] == plan.previously_available


def test_qc_inputs_replace_changed_panels_and_preserve_policy_exclusions(tmp_path):
    def declaration(source_id, revision, evidence):
        return {
            "id": source_id,
            "dataset_id": source_id,
            "dataset_revision": revision,
            "pipeline": None,
            "data_restriction": "open",
            "independent_qc": {"evidence_url": evidence},
        }

    review = _receipt_ref(tmp_path, "review.artifact.json", {"review": "selected"})
    current = {
        sid: declaration(sid, "current", review["uri"]) for sid in ("Marin:carried", "Marin:changed", "Marin:excluded")
    }
    prior = {**current, "Marin:changed": {**current["Marin:changed"], "dataset_revision": "old"}}
    prior_ref = _receipt_ref(tmp_path, "prior-declarations.json", prior)
    declarations = _receipt_ref(tmp_path, "declarations.json", current)
    plan = fixture_plan(qc_exclusions=("Marin:excluded",))
    plan = plan.model_copy(
        update={"pins": plan.pins.model_copy(update={"prior_declarations": ArtifactRef(**prior_ref)})}
    )
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    _declarations_dep(built, prefix, plan, declarations)
    copy_inputs = _materialized_inputs(built.stages["qc_browse_plan"], prefix)
    assert set(copy_inputs["exclusions"]) == {"Marin:changed", "Marin:excluded"}

    completion = _receipt_ref(tmp_path, "completion.json", {"status": "PASS"})
    _write_dep(
        built.stages["qc_browse"],
        prefix,
        "qc_browse",
        plan.release_id,
        completion,
        results=[{"source_id": "Marin:carried"}],
    )
    fresh_inputs = _materialized_inputs(built.stages["fresh_qc_index"], prefix)
    assert fresh_inputs["fresh_panels"] == {"Marin:changed": review}
    typed = typed_inputs(release_graph_module.fresh_qc_index_stage.FreshQcIndexInputs, fresh_inputs)
    assert typed.fresh_panels["Marin:changed"].sha256 == review["sha256"]


def test_curriculum_load_expected_counts_come_from_the_compose_handoff(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    handoff = {"uri": "s3://bucket/handoff.json", "sha256": "c" * 64}
    _write_dep(
        built.stages["curriculum_compose"],
        prefix,
        "curriculum_compose",
        plan.release_id,
        handoff,
        handoff=handoff,
        source_count=4,
        inventory_count=5,
        task_count=900,
        confidence_validated_rows=11,
    )
    scan_receipt = {"uri": "s3://bucket/scan.json", "sha256": "d" * 64}
    _write_dep(built.stages["text_scans"], prefix, "text_scans", plan.release_id, scan_receipt)
    inputs = _materialized_inputs(built.stages["curriculum_load"], prefix)
    assert inputs["handoff"] == handoff
    assert inputs["text_scan"] == scan_receipt
    assert inputs["expected_inventory_count"] == 5
    assert inputs["expected_aligned_sources"] == 4
    assert inputs["expected_task_count"] == 900
    assert inputs["expected_confidence_rows"] == 11
    assert inputs["prior_curriculum_load_id"] == plan.prior_curriculum_load_id
    assert inputs["catalog_target_load_id"] == plan.catalog_target_load_id
    assert inputs["expected_rebound_source"] == ""


def test_release_check_phases_bind_their_receipts(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    declarations_receipt = {"uri": "s3://bucket/declarations.json", "sha256": "9" * 64}
    _write_dep(built.stages["declarations"], prefix, "declarations", plan.release_id, declarations_receipt)
    documents = {
        "backend_map": {"uri": "s3://bucket/backend-map.json", "sha256": "d" * 64},
        "identity_survey": {"uri": "s3://bucket/identity.json", "sha256": "e" * 64},
        "identity_receipt": {"uri": "s3://bucket/id-receipt.json", "sha256": "f" * 64},
    }
    _write_dep(
        built.stages["identity_backend_compose"],
        prefix,
        "identity_backend_compose",
        plan.release_id,
        documents["identity_receipt"],
        documents=documents,
    )
    package_receipt = {"uri": "s3://bucket/package.json", "sha256": "c" * 64}
    derived = {
        "oracle_inventory": {"uri": "s3://bucket/package-inputs/oracle-inventory.json", "sha256": "2" * 64},
        "source_counts": {"uri": "s3://bucket/package-inputs/source-counts.json", "sha256": "3" * 64},
    }
    _write_dep(built.stages["package_preview"], prefix, "package_preview", plan.release_id, package_receipt, **derived)
    preflight_inputs = _materialized_inputs(built.stages["release_check_preflight"], prefix)
    assert preflight_inputs["package"] == package_receipt
    # The preflight checks the census-derived inputs the package sealed, not plan pins.
    assert {name: preflight_inputs[name] for name in derived} == derived
    assert preflight_inputs["phase"] == "preflight"
    assert "published_revision" not in preflight_inputs
    assert preflight_inputs["applet_id"] == plan.service.applet_id

    preflight_receipt = {"uri": "s3://bucket/preflight.json", "sha256": "d" * 64}
    live_revision = plan.service.preview_base_version + 7
    _write_dep(
        built.stages["release_check_preflight"],
        prefix,
        "release_check_preflight",
        plan.release_id,
        preflight_receipt,
        live={"version": live_revision, "mode": "public"},
    )
    package_dep = {"uri": "s3://bucket/package.json", "sha256": "c" * 64}
    marina_inputs = _materialized_inputs(built.stages["marina_release"], prefix)
    # A stale planned base never blocks the publish: it compare-and-sets against the
    # live revision and mode the preflight check recorded (R40: planned 32, live 39, public).
    assert marina_inputs["base_version"] == live_revision != plan.service.preview_base_version
    assert marina_inputs["access_mode"] == "public"
    assert marina_inputs["package"] == package_dep
    assert marina_inputs["preflight_receipt"] == preflight_receipt
    assert "hosted_receipt" not in marina_inputs
    marina_receipt = {"uri": "s3://bucket/manifest.json", "sha256": "e" * 64}
    _write_dep(
        built.stages["marina_release"],
        prefix,
        "marina_release",
        plan.release_id,
        marina_receipt,
        manifest=marina_receipt,
        publish={
            "id": plan.service.applet_id,
            "version": 31,
            "url": "https://applets.example.test",
            "mode": "private",
        },
    )
    # RELEASE.md: the hosted check runs after the publish, on the new revision (R40 #51).
    hosted_inputs = _materialized_inputs(built.stages["release_check_hosted"], prefix)
    assert hosted_inputs["preflight_receipt"] == preflight_receipt
    assert hosted_inputs["published_revision"] == 31
    assert "hosted_receipt" not in hosted_inputs
    hosted_check_receipt = {"uri": "s3://bucket/hosted-check.json", "sha256": "1" * 64}
    _write_dep(
        built.stages["release_check_hosted"],
        prefix,
        "release_check_hosted",
        plan.release_id,
        hosted_check_receipt,
        published_revision=31,
    )
    postpublish_inputs = _materialized_inputs(built.stages["release_check_postpublish"], prefix)
    assert postpublish_inputs["published_revision"] == 31
    assert postpublish_inputs["hosted_receipt"] == hosted_check_receipt
    census_receipt = {"uri": "s3://bucket/census.json", "sha256": "c" * 64}
    _write_dep(built.stages["census"], prefix, "census", plan.release_id, census_receipt, tasks=17)
    hosted_check_inputs = _materialized_inputs(built.stages["hosted_check"], prefix)
    assert hosted_check_inputs["revision"] == 31
    assert hosted_check_inputs["root"] == "https://applets.example.test"
    assert hosted_check_inputs["release_manifest"] == marina_receipt
    assert hosted_check_inputs["retained_tasks"] == 17


def test_snapshot_publication_binds_the_composed_surveys(tmp_path):
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    census_receipt = {"uri": "s3://bucket/census.json", "sha256": "c" * 64}
    _write_dep(built.stages["census"], prefix, "census", plan.release_id, census_receipt, tasks=17, results=[])
    documents = {
        "backend_map": {"uri": "s3://bucket/backend-map.json", "sha256": "d" * 64},
        "identity_survey": {"uri": "s3://bucket/identity.json", "sha256": "e" * 64},
    }
    identity_receipt = documents["identity_receipt"] = {"uri": "s3://bucket/id-receipt.json", "sha256": "f" * 64}
    _write_dep(
        built.stages["identity_backend_compose"],
        prefix,
        "identity_backend_compose",
        plan.release_id,
        identity_receipt,
        documents=documents,
    )
    inputs = _materialized_inputs(built.stages["snapshot_publication"], prefix)
    assert inputs["census_receipt"] == census_receipt
    assert inputs["identity_receipt"] == documents["identity_survey"]
    assert inputs["backend_receipt"] == documents["backend_map"]
    assert inputs["prior_snapshot_id"] == plan.prior_snapshot_id
    assert inputs["prior_declarations_uri"] == plan.pins.prior_declarations.uri


def _exposure_declarations() -> dict[str, dict]:
    """A declaration cut with one license-withheld source and every recorded grant's release."""
    declarations = {
        decision.source_id: {
            "id": decision.source_id,
            "dataset_id": decision.dataset_id,
            "dataset_revision": decision.dataset_revision,
        }
        for decision in PUBLICATION_DECISIONS
    }
    declarations["Marin:secret"] = {"id": "Marin:secret", "exclusion_claims": [{"reason": "license-ban"}]}
    declarations["Marin:open"] = {"id": "Marin:open"}
    return declarations


def test_exposure_probe_derives_its_lists_from_the_declaration_cut(tmp_path):
    """Regression (R40 #48): the R40 plan pins no exposure input check, so the probe
    must read the withheld sources and grants from the sealed declarations."""
    plan = fixture_plan(release_root=str(tmp_path / "release"))
    assert plan.pins.exposure_input_check is None
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    _declarations_dep(built, prefix, plan, _receipt_ref(tmp_path, "declarations.json", _exposure_declarations()))
    marina_receipt = {"uri": "s3://bucket/manifest.json", "sha256": "e" * 64}
    _write_dep(
        built.stages["marina_release"],
        prefix,
        "marina_release",
        plan.release_id,
        marina_receipt,
        manifest=marina_receipt,
        publish={"id": "x", "version": 31, "url": "https://applets.example.test", "mode": "public"},
    )
    inputs = _materialized_inputs(built.stages["exposure_probe"], prefix)
    assert inputs["withheld_sources"] == ("Marin:secret",)
    assert [grant["source_id"] for grant in inputs["publication_grants"]] == [
        decision.source_id for decision in PUBLICATION_DECISIONS
    ]
    assert all(grant["upstream_url"] and grant["license_id"] for grant in inputs["publication_grants"])
    assert inputs["input_check_receipt"] is None
    assert inputs["revision"] == 31
    assert inputs["root"] == plan.service.public_root


def test_exposure_lists_refuse_a_grant_declared_at_another_release():
    declarations = _exposure_declarations()
    granted = PUBLICATION_DECISIONS[0].source_id
    declarations[granted] = {**declarations[granted], "dataset_revision": "0" * 40}
    with pytest.raises(ValueError, match="does not match its declared release"):
        exposure_lists(declarations)


def test_exposure_lists_refuse_a_withheld_grant():
    declarations = _exposure_declarations()
    granted = PUBLICATION_DECISIONS[0].source_id
    declarations[granted] = {**declarations[granted], "exclusion_claims": [{"reason": "license-ban"}]}
    with pytest.raises(ValueError, match="names a withheld source"):
        exposure_lists(declarations)


def test_cluster_stage_run_reads_back_its_graph_receipt_without_dispatch(tmp_path, monkeypatch):
    """A sealed graph receipt satisfies the step without any remote dispatch."""
    release_root = str(tmp_path / "release")
    plan = fixture_plan(release_root=release_root)
    built = release_graph(plan)
    census = built.stages["census"]
    receipt = {"uri": str(tmp_path / "census-receipt.json"), "sha256": "c" * 64}
    graph_receipt = {
        "kind": GRAPH_RECEIPT_KIND,
        "node": "census",
        "release_id": plan.release_id,
        "receipt": receipt,
        "proof": {"receipt": receipt, "release_id": plan.release_id, "tasks": 17},
    }
    _seal_document(f"{release_root}/graph/census.json", graph_receipt)

    def _no_dispatch(*args, **kwargs):
        raise AssertionError("a sealed graph receipt must not trigger a remote dispatch")

    monkeypatch.setattr(release_graph_module, "remote", _no_dispatch)
    config = {
        "release_id": plan.release_id,
        "source_revision": plan.source_revision,
        "inputs": {"uri": "s3://bucket/census-inputs.json", "sha256": "c" * 64},
        "output_root": f"{release_root}/census",
        "region_prefix": plan.region_prefix,
        "expected_sources": plan.expected_source_count,
        "slice_output_root": None,
        "graph_receipt_root": f"{release_root}/graph",
    }
    loaded = census.run(config)
    assert loaded.stage == "census" and loaded.receipt == ArtifactRef(**receipt)
    assert loaded.proof["tasks"] == 17


def test_execute_stage_node_seals_the_graph_receipt(tmp_path, monkeypatch):
    """The worker-side dispatch seals the graph receipt the driver reads back."""
    release_root = str(tmp_path / "release")
    config = {"graph_receipt_root": f"{release_root}/graph"}

    def _fake_executor(_config: dict) -> dict:
        return {
            "release_id": "testrelease",
            "receipt": {"uri": str(tmp_path / "receipt.json"), "sha256": "c" * 64},
        }

    monkeypatch.setitem(release_graph_module._STAGE_EXECUTORS, "fixture_stage", _fake_executor)
    _execute_stage_node("fixture_stage", "fixture_stage", "receipt", config)
    document = json.loads(StoragePath(f"{release_root}/graph/fixture_stage.json").read_text())
    assert document["kind"] == GRAPH_RECEIPT_KIND and document["node"] == "fixture_stage"
    assert document["receipt"]["uri"].endswith("receipt.json")

    def _nested_executor(_config: dict) -> dict:
        return {
            "release_id": "testrelease",
            "documents": {"identity_receipt": {"uri": str(tmp_path / "identity.json"), "sha256": "d" * 64}},
        }

    monkeypatch.setitem(release_graph_module._STAGE_EXECUTORS, "fixture_stage", _nested_executor)
    _execute_stage_node("fixture_identity", "fixture_stage", "documents.identity_receipt", config)
    document = json.loads(StoragePath(f"{release_root}/graph/fixture_identity.json").read_text())
    assert document["receipt"]["uri"].endswith("identity.json")


def test_cluster_dispatch_extracts_the_spec_receipt_key(tmp_path, monkeypatch):
    """The cluster dispatch reads the spec's sealed reference, not the default key.

    Regression: the dispatch defaulted to 'receipt' while the identity-backend
    compose spec seals its receipt under 'documents.identity_receipt', so the
    stage's own graph receipt could never be read back.
    """
    release_root = str(tmp_path / "release")
    plan = fixture_plan(release_root=release_root)
    built = release_graph(plan)
    spec = release_graph_module._STAGE_SPECS["identity_backend_compose"]
    receipt_ref = {
        "uri": f"{release_root}/identity-backend/identity-composition-receipt.json",
        "sha256": "a" * 64,
    }

    def fake_executor(_config: dict) -> dict:
        return {"release_id": plan.release_id, "documents": {"identity_receipt": receipt_ref}}

    captured = {}

    def fake_remote(callable_, *, resources, env_vars=None, pip_packages=None, pip_dependency_groups=None):
        captured["dispatch"] = callable_
        captured["env_vars"] = env_vars
        captured["pip_packages"] = pip_packages
        captured["pip_dependency_groups"] = pip_dependency_groups
        return lambda config: callable_(config)

    monkeypatch.setitem(release_graph_module._STAGE_EXECUTORS, spec.stage, fake_executor)
    monkeypatch.setattr(release_graph_module, "remote", fake_remote)
    receipt = built.stages["identity_backend_compose"].run({"graph_receipt_root": f"{release_root}/graph"})
    dispatch = captured["dispatch"]
    assert dispatch.func is release_graph_module._execute_stage_node
    assert dispatch.args == ("identity_backend_compose", spec.stage, "documents.identity_receipt")
    assert captured["env_vars"] is not None
    assert receipt.receipt == ArtifactRef(**receipt_ref)


def test_cpu_embed_dispatch_installs_the_torch_cpu_extra(tmp_path, monkeypatch):
    """The Harrier embedder imports torch; a CPU embed job and its workers need the cpu extra.

    Regression: R40's embed dispatched with no extras and every Zephyr worker failed
    with ModuleNotFoundError: torch.
    """
    release_root = str(tmp_path / "release")
    plan = fixture_plan(release_root=release_root)
    assert plan.pins.parity_receipt is None  # the CPU path
    spec = release_graph_module._STAGE_SPECS["embed"]
    receipt_ref = {"uri": f"{release_root}/embed/receipt.json", "sha256": "b" * 64}
    captured = {}

    def fake_remote(callable_, *, resources, env_vars=None, pip_packages=None, pip_dependency_groups=None):
        captured["pip_dependency_groups"] = pip_dependency_groups
        return lambda config: callable_(config)

    monkeypatch.setitem(
        release_graph_module._STAGE_EXECUTORS,
        spec.stage,
        lambda _config: {"release_id": plan.release_id, "receipt": receipt_ref},
    )
    monkeypatch.setattr(release_graph_module, "remote", fake_remote)
    release_graph(plan).stages["embed"].run({"graph_receipt_root": f"{release_root}/graph"})
    assert captured["pip_dependency_groups"] == [release_graph_module.EMBED_CPU_DEPENDENCY_GROUP]


def test_measured_module_path_resolves_a_dotted_module_or_a_repo_path():
    """A hand-written plan names the measured module; a derived plan records its repo path."""
    from_name = release_graph_module.measured_module_path("taskcompendium.models")
    relative = str(Path(from_name).resolve().relative_to(release_graph_module.REPO_ROOT.resolve()))
    assert Path(release_graph_module.measured_module_path(relative)).resolve() == Path(from_name).resolve()
    with pytest.raises(ValueError, match="neither a repo path nor an importable module"):
        release_graph_module.measured_module_path("not_a_module_anywhere")


def test_stage_child_env_forwards_the_attestation_and_credentials(monkeypatch):
    """Dispatched stage jobs carry the attestation and credentials, nothing else."""
    monkeypatch.setenv("ATLAS_RELEASE_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("CW_KEY_SECRET", "s3cret")
    monkeypatch.setenv("HF_TOKEN", "hf-tok")
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.setenv("UNRELATED_VAR", "noise")
    env = release_graph_module._stage_child_env()
    assert env["ATLAS_RELEASE_SOURCE_REVISION"] == "a" * 40
    assert env["CW_KEY_SECRET"] == "s3cret"
    assert env["HF_TOKEN"] == "hf-tok"
    assert "CW_KEY_ID" not in env
    assert "UNRELATED_VAR" not in env


def test_materialize_job_credentials_writes_the_marin_record(monkeypatch, tmp_path):
    raw = json.dumps({"cluster": "marin", "endpoint": "https://marina.oa.dev", "edge_refresh_token": "tok"})
    monkeypatch.setenv("MARIN_IAP_CREDENTIAL_JSON", raw)
    monkeypatch.setattr(release_graph_module.Path, "home", staticmethod(lambda: tmp_path))
    release_graph_module._materialize_job_credentials()
    record = tmp_path / ".config/marin/credentials/marin.json"
    assert record.read_text() == raw
    release_graph_module._materialize_job_credentials()  # idempotent: the record stays
    assert record.read_text() == raw


def test_typed_inputs_coerce_reference_fields():
    config = {
        "release_id": "testrelease",
        "source_revision": "a" * 40,
        "inputs": {"uri": "s3://bucket/inputs.json", "sha256": "c" * 64},
        "output_root": "/fixture/census",
        "region_prefix": "s3://bucket-region/",
        "expected_sources": 2,
        "workers": 4,
        "slice_output_root": None,
    }
    inputs = typed_inputs(census_stage.CensusInputs, config)
    assert isinstance(inputs.inputs, ArtifactRef)
    assert inputs.expected_sources == 2
    assert inputs.workers == 4


def test_in_flight_r40_plan_reseals_its_recorded_plan_document():
    """R40's sealed plan step pins its fingerprint; a plan-model edit must not move it.

    The R40 plan still carries the prior oracle-inventory pin that the package no
    longer reads; dropping the field from the model refused every R40 resume.
    """
    plan = load_plan(REPO_ROOT / "experiments/post_training/task_curation/release_stages/plans/r40.json")
    assert release_graph(plan).plan_handle.fingerprint() == "c0088866"
    without = plan.model_copy(update={"pins": plan.pins.model_copy(update={"oracle_inventory": None})})
    assert "oracle_inventory" not in _plan_document_data(without)["pins"]


def test_curriculum_compose_receipt_is_its_sealed_handoff():
    """R40: compose sealed its handoff, then the graph refused a proof without a 'receipt' key."""
    handoff = {"uri": "s3://bucket/compose/composed/attempt-1/handoff.json", "sha256": "a" * 64}
    proof = {"release_id": "r40", "handoff": handoff, "task_count": 3, "source_count": 1}
    receipt = _stage_receipt("curriculum_compose", proof, _STAGE_SPECS["curriculum_compose"].receipt_key)
    assert receipt.receipt.model_dump(mode="json") == handoff


def test_typed_inputs_keep_defaults_for_inputs_a_phase_omits(tmp_path):
    """R40: the preflight config names no preflight receipt; typed_inputs raised KeyError."""
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    package_receipt = {"uri": "s3://bucket/package.json", "sha256": "c" * 64}
    derived = {
        "oracle_inventory": {"uri": "s3://bucket/package-inputs/oracle-inventory.json", "sha256": "2" * 64},
        "source_counts": {"uri": "s3://bucket/package-inputs/source-counts.json", "sha256": "3" * 64},
    }
    _write_dep(built.stages["package_preview"], prefix, "package_preview", plan.release_id, package_receipt, **derived)
    config = _materialized_inputs(built.stages["release_check_preflight"], prefix)
    config.pop("graph_receipt_root")
    inputs = typed_inputs(release_check.ReleaseCheckInputs, config)
    assert inputs.preflight_receipt is None and inputs.published_revision is None
    inputs.validate()


def test_grader_drift_leaf_runs_only_when_the_plan_opt_in(tmp_path):
    """The drift leaf exists for every plan but refuses without the derivation-pinned reference."""
    plan = fixture_plan()
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    _declarations_dep(built, prefix, plan, {"uri": "s3://bucket/declarations.json", "sha256": "9" * 64})
    inputs = _materialized_inputs(built.stages["main_grader_drift"], prefix)
    assert inputs["reference"].main.revision == "e" * 40
    refused = fixture_plan(grader_drift_reference=None)
    unbound = release_graph(refused)
    _declarations_dep(unbound, prefix, refused, {"uri": "s3://bucket/declarations.json", "sha256": "9" * 64})
    with pytest.raises(ValueError, match="pins no grader_drift_reference"):
        _materialized_inputs(unbound.stages["main_grader_drift"], prefix)
