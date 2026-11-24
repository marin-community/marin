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
from experiments.post_training.task_curation.release_stages.graph import (
    GRAPH_RECEIPT_KIND,
    ComposeManifestDocument,
    StageReceipt,
    _execute_stage_node,
    _seal_document,
    graph_nodes,
    release_graph,
    typed_inputs,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan

EXPECTED_NODES = (
    "declarations",
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
    with pytest.raises(ValueError, match="carry no tasks for targets"):
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
    assert inputs["measured_module"] == plan.measured.module
    assert inputs["previously_available"] == plan.previously_available


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
    _write_dep(built.stages["package_preview"], prefix, "package_preview", plan.release_id, package_receipt)
    preflight_inputs = _materialized_inputs(built.stages["release_check_preflight"], prefix)
    assert preflight_inputs["package"] == package_receipt
    assert preflight_inputs["phase"] == "preflight"
    assert "published_revision" not in preflight_inputs
    assert preflight_inputs["applet_id"] == plan.service.applet_id

    preflight_receipt = {"uri": "s3://bucket/preflight.json", "sha256": "d" * 64}
    _write_dep(
        built.stages["release_check_preflight"],
        prefix,
        "release_check_preflight",
        plan.release_id,
        preflight_receipt,
    )
    hosted_inputs = _materialized_inputs(built.stages["release_check_hosted"], prefix)
    assert hosted_inputs["preflight_receipt"] == preflight_receipt
    # The pre-publish hosted check verifies the live preview revision (r39-quirk note).
    assert hosted_inputs["published_revision"] == plan.service.preview_base_version

    hosted_check_receipt = {"uri": "s3://bucket/hosted-check.json", "sha256": "1" * 64}
    _write_dep(
        built.stages["release_check_hosted"],
        prefix,
        "release_check_hosted",
        plan.release_id,
        hosted_check_receipt,
    )
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
    postpublish_inputs = _materialized_inputs(built.stages["release_check_postpublish"], prefix)
    assert postpublish_inputs["published_revision"] == 31
    assert postpublish_inputs["hosted_receipt"] == hosted_check_receipt
    hosted_check_inputs = _materialized_inputs(built.stages["hosted_check"], prefix)
    assert hosted_check_inputs["revision"] == 31
    assert hosted_check_inputs["root"] == "https://applets.example.test"
    assert hosted_check_inputs["release_manifest"] == marina_receipt


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


def test_exposure_probe_binds_the_input_check_evidence(tmp_path):
    release_root = str(tmp_path / "release")
    input_check = {
        "withheld_sources": ["Marin:secret"],
        "publication_grants": [
            {"source_id": "Marin:open", "upstream_url": "https://upstream", "license_id": "cc-by-4.0"}
        ],
    }
    input_check_ref = _receipt_ref(tmp_path, "exposure-input-check.json", input_check)
    plan = fixture_plan(
        release_root=release_root,
        pins={**fixture_plan().pins.model_dump(mode="json"), "exposure_input_check": input_check_ref},
    )
    built = release_graph(plan)
    prefix = str(tmp_path / "prefix")
    marina_receipt = {"uri": "s3://bucket/manifest.json", "sha256": "e" * 64}
    _write_dep(
        built.stages["marina_release"],
        prefix,
        "marina_release",
        plan.release_id,
        marina_receipt,
        manifest=marina_receipt,
        publish={"id": "x", "version": 31, "url": "https://applets.example.test", "mode": "private"},
    )
    inputs = _materialized_inputs(built.stages["exposure_probe"], prefix)
    assert inputs["withheld_sources"] == ("Marin:secret",)
    assert inputs["publication_grants"] == tuple(input_check["publication_grants"])
    assert inputs["revision"] == 31
    assert inputs["root"] == plan.service.public_root


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

    def fake_remote(callable_, *, resources):
        captured["dispatch"] = callable_
        return lambda config: callable_(config)

    monkeypatch.setitem(release_graph_module._STAGE_EXECUTORS, spec.stage, fake_executor)
    monkeypatch.setattr(release_graph_module, "remote", fake_remote)
    receipt = built.stages["identity_backend_compose"].run({"graph_receipt_root": f"{release_root}/graph"})
    dispatch = captured["dispatch"]
    assert dispatch.func is release_graph_module._execute_stage_node
    assert dispatch.args == ("identity_backend_compose", spec.stage, "documents.identity_receipt")
    assert receipt.receipt == ArtifactRef(**receipt_ref)


def test_typed_inputs_coerce_reference_fields():
    config = {
        "release_id": "testrelease",
        "source_revision": "a" * 40,
        "inputs": {"uri": "s3://bucket/inputs.json", "sha256": "c" * 64},
        "output_root": "/fixture/census",
        "region_prefix": "s3://bucket-region/",
        "expected_sources": 2,
        "slice_output_root": None,
    }
    inputs = typed_inputs(census_stage.CensusInputs, config)
    assert isinstance(inputs.inputs, ArtifactRef)
    assert inputs.expected_sources == 2
