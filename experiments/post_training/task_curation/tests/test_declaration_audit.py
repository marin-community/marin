# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The declaration audit pairs every Atlas receipt family with the canonical layer."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from finestore.rl.curation import ActorCensusInput, PipelineReference
from finestore.rl.status_taxonomy import SourceStatus

from experiments.post_training.task_curation import declaration_audit, ensure_current_projections
from experiments.post_training.task_curation import sources as sources_module
from experiments.post_training.task_curation.declaration_audit import (
    CanonicalLayer,
    DeclaredPin,
    ReleasePin,
    audit_families,
)


def pin(source_id, dataset_id="open-athena/example-source", revision="1" * 40, version="2026.11.13.1"):
    return DeclaredPin(source_id=source_id, dataset_id=dataset_id, dataset_revision=revision, converter_version=version)


def release(repo="open-athena/example-release", revision="2" * 40):
    return ReleasePin(repo=repo, revision=revision)


def layer(**overrides) -> CanonicalLayer:
    base = dict(
        source_ids=frozenset({"Test:declared-source"}),
        declared={"Test:declared-source": pin("Test:declared-source")},
        selected={"Test:declared-source": release()},
        retired=frozenset(),
        archived_releases={},
    )
    base.update(overrides)
    return CanonicalLayer(**base)


def test_a_receipt_at_the_declared_pin_binds():
    findings, notes = audit_families(
        {
            "census receipts": {"Test:declared-source": pin("Test:declared-source")},
            "within-source dedup": {"Test:declared-source": pin("Test:declared-source")},
            "lexical receipts": {"Test:declared-source": pin("Test:declared-source")},
        },
        layer=layer(),
        releases={"Test:declared-source": release()},
    )
    assert findings == []
    assert notes == []


SEALED_RELEASE = release(repo="open-athena/sealed-repair-cohort", revision="3" * 40)


def sealed_layer(**overrides) -> CanonicalLayer:
    """A layer whose declared pin is the licensed upstream and whose selected pin is the sealed publication."""
    base = dict(
        source_ids=frozenset({"Test:sealed-repair-cohort"}),
        declared={"Test:sealed-repair-cohort": pin("Test:sealed-repair-cohort", "open-thoughts/TaskTrove", "4" * 40)},
        selected={"Test:sealed-repair-cohort": SEALED_RELEASE},
        retired=frozenset(),
        archived_releases={},
        admitted_census_converters={"Test:sealed-repair-cohort": "2026.11.19.1"},
    )
    base.update(overrides)
    return CanonicalLayer(**base)


def sealed_pin(version="2026.11.19.1"):
    return DeclaredPin(
        source_id="Test:sealed-repair-cohort",
        dataset_id="open-athena/sealed-repair-cohort",
        dataset_revision="3" * 40,
        converter_version=version,
    )


@pytest.mark.parametrize("family", ["census receipts", "within-source dedup"])
def test_an_admitted_repair_cohorts_selected_population_pin_binds(family):
    """The sealed cohort's receipts measure its published population (ruling 66)."""
    findings, notes = audit_families(
        {family: {"Test:sealed-repair-cohort": sealed_pin()}},
        layer=sealed_layer(),
        releases={},
    )
    assert findings == []
    assert notes == []


@pytest.mark.parametrize("family", ["census receipts", "within-source dedup"])
def test_a_selected_pin_without_an_admitted_receipt_stays_a_finding(family):
    findings, _ = audit_families(
        {family: {"Test:sealed-repair-cohort": sealed_pin()}},
        layer=sealed_layer(admitted_census_converters={}),
        releases={},
    )
    assert len(findings) == 1
    assert "Test:sealed-repair-cohort" in findings[0]


@pytest.mark.parametrize("family", ["census receipts", "within-source dedup"])
def test_a_census_converter_mismatch_against_the_admission_stays_a_finding(family):
    findings, _ = audit_families(
        {family: {"Test:sealed-repair-cohort": sealed_pin(version="2026.11.20.1")}},
        layer=sealed_layer(),
        releases={},
    )
    assert len(findings) == 1
    assert "2026.11.20.1" in findings[0]


def test_a_drifted_retained_binding_reports_a_gap_not_a_family_finding(monkeypatch):
    """A retained release whose conversion pin drifts is a waivable gap, never a crash."""
    monkeypatch.setattr(sources_module, "RETAINED_BINDING_DRIFTS", {"Test:drifted-source": "3" * 40})
    # The control: an unrecorded unselected release still fails the family audit.
    findings, _ = audit_families(
        {},
        layer=layer(selected={}),
        releases={"Test:unrecorded-source": release()},
    )
    assert any("Test:unrecorded-source" in finding for finding in findings)
    # The drift-recorded source produces no family finding: the audit reports it as a
    # waivable served-catalog gap at the receipt's pinned revision instead.
    findings, _ = audit_families(
        {},
        layer=layer(selected={}),
        releases={"Test:drifted-source": release()},
    )
    assert findings == []


def test_a_registry_only_receipt_fails():
    findings, _ = audit_families(
        {"census receipts": {"Test:undeclared-source": pin("Test:undeclared-source")}},
        layer=layer(),
    )
    assert findings == [
        "census receipts: Test:undeclared-source names a source the declaration layer does not declare "
        "(registry-only receipt at " + "1" * 40 + ")"
    ]


def test_a_stale_declaration_pin_fails():
    findings, _ = audit_families(
        {
            "census receipts": {"Test:declared-source": pin("Test:declared-source", revision="9" * 40)},
            "within-source dedup": {"Test:declared-source": pin("Test:declared-source", version="2026.11.12.1")},
        },
        layer=layer(),
    )
    assert len(findings) == 2
    assert all("but the canonical layer declares" in finding for finding in findings)


def test_a_lexical_receipt_at_neither_declared_nor_selected_pin_fails():
    findings, _ = audit_families(
        {"lexical receipts": {"Test:declared-source": pin("Test:declared-source", revision="8" * 40)}},
        layer=layer(),
    )
    assert findings == [
        "lexical receipts: Test:declared-source receipt pin open-athena/example-source@"
        + "8" * 40
        + " (2026.11.13.1) matches no declared or selected pin"
    ]


def test_a_retained_receipt_must_match_the_selected_release():
    findings, _ = audit_families(
        {},
        layer=layer(),
        releases={"Test:declared-source": release(repo="open-athena/other-release")},
    )
    assert findings == [
        "retained release receipts: Test:declared-source selects open-athena/example-release@"
        + "2" * 40
        + " but its receipt publishes open-athena/other-release@"
        + "2" * 40
    ]


def test_a_receipt_for_a_retired_source_is_history_not_drift():
    retired_layer = layer(source_ids=frozenset(), retired=frozenset({"Test:retired-source"}))
    findings, _ = audit_families(
        {
            "census receipts": {"Test:retired-source": pin("Test:retired-source", revision="9" * 40)},
            "within-source dedup": {"Test:retired-source": pin("Test:retired-source")},
        },
        layer=retired_layer,
        releases={"Test:retired-source": release()},
    )
    assert findings == []


def test_a_filtration_superseded_retained_receipt_binds():
    superseded = layer(
        source_ids=frozenset({"Test:declared-source", "Test:filtered-source"}),
        declared={
            "Test:declared-source": pin("Test:declared-source"),
            "Test:filtered-source": pin("Test:filtered-source"),
        },
        selected={
            "Test:declared-source": release(),
            "Test:filtered-source": release(repo="open-athena/filtered-release", revision="3" * 40),
        },
        superseded_by_filtration=frozenset({"Test:filtered-source"}),
    )
    findings, notes = audit_families(
        {},
        layer=superseded,
        releases={"Test:filtered-source": release(revision="4" * 40)},
    )
    assert findings == []
    assert notes == []


def test_a_pending_source_retained_release_is_reported_staged_for_intake():
    pending = layer(
        source_ids=frozenset({"Test:declared-source", "Test:pending-source"}),
        declared={
            "Test:declared-source": pin("Test:declared-source"),
            "Test:pending-source": pin("Test:pending-source"),
        },
        superseded_by_filtration=frozenset(),
        pending=frozenset({"Test:pending-source"}),
    )
    findings, notes = audit_families(
        {
            "census receipts": {"Test:pending-source": pin("Test:pending-source")},
            "within-source dedup": {"Test:pending-source": pin("Test:pending-source")},
        },
        layer=pending,
        releases={"Test:pending-source": release(repo="open-athena/pending-remediated-release")},
    )
    assert findings == []
    assert notes == [
        "retained release receipts: Test:pending-source is pending intake; its receipt for "
        "open-athena/pending-remediated-release@" + "2" * 40 + " stays staged until adoption selects it"
    ]


def test_an_archived_original_record_counts_as_declared():
    archived = layer(
        source_ids=frozenset({"Test:archived-source"}),
        declared={
            "Test:archived-source": pin("Test:archived-source", dataset_id="open-athena/original-conversion"),
        },
        selected={"Test:archived-source": release(repo="open-athena/admitted-release", revision="5" * 40)},
        archived_releases={"Test:archived-source": release(repo="open-athena/archived-retained", revision="6" * 40)},
    )
    findings, notes = audit_families(
        {
            "census receipts": {
                "Test:archived-source": pin("Test:archived-source", dataset_id="open-athena/original-conversion")
            }
        },
        layer=archived,
        releases={"Test:archived-source": release(repo="open-athena/archived-retained", revision="6" * 40)},
    )
    assert findings == []
    assert notes == []


@pytest.fixture
def waived_audit_cli(monkeypatch, tmp_path):
    waivers = ensure_current_projections.ProjectionWaivers(
        marina_url="https://marina.example",
        applet_id="preview",
        rule="stale-or-missing-serving-projection-without-declaration-drift",
        reason=ensure_current_projections.PROJECTION_WAIVER_REASON,
    )
    path = tmp_path / "waivers.json"
    path.write_text(waivers.model_dump_json())
    monkeypatch.setattr(sys, "argv", ["declaration_audit", "--waivers", str(path)])
    monkeypatch.setattr(
        declaration_audit,
        "canonical_layer",
        lambda: (
            layer(),
            {
                "census receipts": {"Test:declared-source": pin("Test:declared-source", revision="9" * 40)},
                "lexical receipts": {},
                "grading census receipts": {},
                "within-source dedup": {},
            },
            {},
            {},
        ),
    )
    monkeypatch.setattr("experiments.post_training.task_curation.sources.configured_sources", lambda: [])
    monkeypatch.setattr(declaration_audit, "audit_typed_status", lambda: [])
    monkeypatch.setattr(
        declaration_audit,
        "atlas_catalog_selection",
        lambda: ensure_current_projections.CatalogSelection(
            "https://marina.example", "preview", "lagging", refresh_sealed=False
        ),
    )
    monkeypatch.setattr(declaration_audit, "atlas_contract_receipts", lambda: {})
    monkeypatch.setattr(declaration_audit, "atlas_curriculum_rows", lambda: {})
    monkeypatch.setattr(declaration_audit, "served_declaration_pins", lambda selection: [])
    return declaration_audit.main


def test_projection_waivers_cannot_clear_declaration_drift(waived_audit_cli, capsys):
    with pytest.raises(SystemExit, match="1 mismatches"):
        waived_audit_cli()
    assert "but the canonical layer declares" in capsys.readouterr().out


@pytest.mark.parametrize("gap", [None, "served-catalog", "derived-contract", "curriculum", "unmeasured"])
def test_a_remeasured_pin_clears_archived_retention_drift_only_through_current_projections(
    waived_audit_cli, monkeypatch, capsys, gap
):
    source_id = "Test:declared-source"
    actor = ActorCensusInput(
        dataset_id="open-athena/example-source",
        dataset_revision="1" * 40,
        converter_version="2026.11.13.1",
        converter_revision="2" * 40,
        measured_tasks=10,
        task_specs_sha256="c" * 64,
        evidence_url="memory://current-census",
    )
    old_actor = actor.model_copy(update={"dataset_revision": "9" * 40})
    pipeline = PipelineReference(
        name="example",
        version=actor.converter_version,
        source=actor.dataset_id,
        revision=actor.dataset_revision,
        files=[],
    )
    source = SimpleNamespace(
        info=SimpleNamespace(id=source_id, actor_census_input=None if gap == "unmeasured" else actor, count=10),
        dataset=SimpleNamespace(name=actor.dataset_id, revision=actor.dataset_revision),
        pipeline_reference=pipeline,
        status=SourceStatus.AVAILABLE,
    )
    served = ensure_current_projections.DeclarationPin(
        source_id, actor.dataset_id, actor.dataset_revision, pipeline.model_dump(mode="json"), actor
    )
    if gap == "served-catalog":
        served = replace(served, dataset_revision=old_actor.dataset_revision, actor_binding=old_actor)
    contracts = {source_id: [SimpleNamespace(input=old_actor if gap == "derived-contract" else actor, summary={})]}
    curriculum = {
        source_id: {
            "declaration": {
                "dataset_id": actor.dataset_id,
                "dataset_revision": old_actor.dataset_revision if gap == "curriculum" else actor.dataset_revision,
                "pipeline": pipeline.model_dump(mode="json"),
            },
            "has_alignment": True,
        }
    }
    monkeypatch.setattr(sys, "argv", ["declaration_audit"])
    families = {
        name: {} for name in ("census receipts", "lexical receipts", "grading census receipts", "within-source dedup")
    }
    monkeypatch.setattr(declaration_audit, "canonical_layer", lambda: (layer(), families, {}, {source_id: source}))
    monkeypatch.setattr(declaration_audit, "audit_replacement_links", lambda: [])
    monkeypatch.setattr(declaration_audit, "atlas_contract_receipts", lambda: contracts)
    monkeypatch.setattr(declaration_audit, "atlas_curriculum_rows", lambda: curriculum)
    monkeypatch.setattr(declaration_audit, "served_declaration_pins", lambda selection: [served])
    monkeypatch.setattr(sources_module, "RETAINED_BINDING_DRIFTS", {source_id: old_actor.dataset_revision})
    monkeypatch.setattr("experiments.post_training.task_curation.sample_qc_binding.SAMPLE_BINDING_DRIFTS", {})
    if gap is None:
        waived_audit_cli()
    else:
        with pytest.raises(SystemExit, match="1 mismatches"):
            waived_audit_cli()
    result = json.loads(capsys.readouterr().out)
    expected_projection = "served-catalog" if gap == "unmeasured" else gap
    assert len(result["findings"]) == (0 if gap is None else 1)
    if gap is not None:
        assert f"serving projection {expected_projection}/stale-pin:" in result["findings"][0]
    assert result["projection_waivers"] == []


def test_typed_status_audit_reports_each_violation(monkeypatch):
    """The typed status audit measures the plain-JSON extras rows.

    The registry layer validates through the typed models, so this fixture
    polices the strings the extras artifact publishes: every status, stage,
    Use, Format and Method value must come from its enum.
    """
    from experiments.post_training.task_curation import declaration_audit as audit_module  # noqa: PLC0415

    extras = {
        "pending": [
            {
                "id": "Test:intake",
                "status": "Pending",
                "stage": "Intake queued",
                "reasons": [],
                "use": "eval",
                "task_format": "taskspec",
                "method_tags": ["RL"],
            },
            {
                "id": "Test:free-stage",
                "status": "Pending",
                "stage": "Hand-entered",
                "reasons": [],
                "use": "train",
                "task_format": "taskspec",
                "method_tags": ["RL"],
            },
            {
                "id": "Test:free-use",
                "status": "Pending",
                "stage": "Intake queued",
                "reasons": [],
                "use": "unknown",
                "task_format": "taskspec",
                "method_tags": ["RL"],
            },
        ],
        "remediation_queue": [],
        "metadata_exclusions": [
            {
                "id": "Test:meta",
                "status": "Excluded",
                "stage": None,
                "reasons": [{"reason": "license-ban"}],
                "use": None,
                "task_format": None,
                "method_tags": [],
            },
            {
                "id": "Test:free-method",
                "status": "Excluded",
                "stage": None,
                "reasons": [{"reason": "license-ban"}],
                "use": None,
                "task_format": "taskspec",
                "method_tags": ["distillation"],
            },
            {
                "id": "Test:free-status",
                "status": "Not selectable",
                "stage": None,
                "reasons": [],
                "use": None,
                "task_format": None,
                "method_tags": [],
            },
        ],
    }
    monkeypatch.setattr("experiments.post_training.task_curation.source_status.status_extras", lambda: extras)
    findings = audit_module.audit_typed_status()
    assert "extras stage 'Hand-entered' is outside PendingStage: Test:free-stage" in findings
    assert "extras Use 'unknown' is outside SourceUse: Test:free-use" in findings
    assert "extras Method tag 'distillation' is outside SourceMethod: Test:free-method" in findings
    assert "extras status 'Not selectable' is outside SourceStatus: Test:free-status" in findings
    assert len(findings) == 4


def test_projection_waivers_cannot_clear_a_served_query_failure(waived_audit_cli, monkeypatch):
    def failed_query():
        raise RuntimeError("served query failed")

    monkeypatch.setattr(declaration_audit, "atlas_contract_receipts", failed_query)
    with pytest.raises(RuntimeError, match="served query failed"):
        waived_audit_cli()


def test_a_same_source_drift_leaves_its_projection_gap_blocking():
    result = declaration_audit.audit_declarations(
        {
            "census receipts": {"Test:declared-source": pin("Test:declared-source", revision="9" * 40)},
            "within-source dedup": {},
        },
        layer=layer(),
        releases={},
        configured_ids=frozenset(),
    )
    rule = ensure_current_projections.ProjectionWaivers(
        marina_url="https://marina.example",
        applet_id="preview",
        rule="stale-or-missing-serving-projection-without-declaration-drift",
        reason=ensure_current_projections.PROJECTION_WAIVER_REASON,
    )
    gap = ensure_current_projections.ProjectionGap("Test:declared-source", "curriculum", "missing", None)
    waived = ensure_current_projections.apply_projection_waivers(
        [gap],
        rule,
        ensure_current_projections.CatalogSelection(
            "https://marina.example", "preview", "lagging", refresh_sealed=False
        ),
        result.drift_sources,
    )
    assert result.findings
    assert result.drift_sources == {"Test:declared-source"}
    assert waived.blocking == [gap]
    assert waived.waived == []


def test_only_a_passing_strict_finalizer_records_the_refresh_seal(waived_audit_cli, monkeypatch, tmp_path):
    path = tmp_path / "strict.json"
    monkeypatch.setattr(sys, "argv", ["declaration_audit", "--seal-refresh", "--output", str(path)])
    monkeypatch.setattr(declaration_audit, "PROJECTION_WAIVER_CONFIG", tmp_path / "waivers.json")
    calls = []

    def record(selection, scope, audit):
        calls.append(audit)
        return replace(selection, refresh_sealed=True)

    monkeypatch.setattr(declaration_audit, "record_projection_refresh_seal", record)
    with pytest.raises(SystemExit, match="1 mismatches"):
        waived_audit_cli()
    assert calls == []
    families = {
        "census receipts": {},
        "lexical receipts": {},
        "grading census receipts": {},
        "within-source dedup": {},
    }
    monkeypatch.setattr(declaration_audit, "canonical_layer", lambda: (layer(), families, {}, {}))
    waived_audit_cli()
    receipt = json.loads(path.read_text())
    assert len(calls) == 1
    assert receipt["refresh_sealed"] is True
    assert receipt["waived_projection_count"] == 0
    assert receipt["waiver_scope"] is None


def test_the_census_demand_exemption_names_exactly_excluded_empty_sources():
    """Ruling 2026-11-26: the projection audit exempts exactly Excluded or Deprecated sources whose
    retained population is empty; any other status or a nonempty population stays
    inside the census demand."""

    class Source:
        def __init__(self, source_id, status, count):
            self.info = type("Info", (), {"id": source_id, "count": count})()
            self._status = status

        @property
        def status(self):
            return self._status

    sources = {
        "Test:zero-excluded": Source("Test:zero-excluded", SourceStatus.EXCLUDED, 0),
        "Test:none-excluded": Source("Test:none-excluded", SourceStatus.EXCLUDED, None),
        "Test:excluded-with-tasks": Source("Test:excluded-with-tasks", SourceStatus.EXCLUDED, 2574),
        "Test:pending-empty": Source("Test:pending-empty", SourceStatus.PENDING, 0),
        "Test:available-empty": Source("Test:available-empty", SourceStatus.AVAILABLE, 0),
        "Test:deprecated-empty": Source("Test:deprecated-empty", SourceStatus.DEPRECATED, 0),
        "Test:deprecated-with-tasks": Source("Test:deprecated-with-tasks", SourceStatus.DEPRECATED, 12927),
    }
    assert declaration_audit.exempt_unmeasured_sources(sources) == frozenset(
        {"Test:zero-excluded", "Test:none-excluded", "Test:deprecated-empty"}
    )


def test_registry_index_gaps_name_every_uncovered_declared_source():
    """The admission flow registers each newly declared source's identity document in
    the registry index (the compose reads its targets there); the landing gate names
    every declared source the index does not cover, and only those."""
    index = {"sources": {"Test:covered": {"uri": "s3://example/covered.json", "sha256": "a" * 64}}}
    declared = frozenset({"Test:covered", "Test:uncovered-a", "Test:uncovered-b"})
    assert declaration_audit.registry_index_gaps(declared, index) == ["Test:uncovered-a", "Test:uncovered-b"]


def test_registry_index_gaps_refuse_a_shapeless_index():
    declared = frozenset({"Test:declared-source"})
    assert declaration_audit.registry_index_gaps(declared, {}) == ["Test:declared-source"]
    assert declaration_audit.registry_index_gaps(declared, {"sources": "not-a-mapping"}) == ["Test:declared-source"]


def test_registry_index_gaps_pass_when_every_declared_source_is_covered():
    index = {
        "sources": {
            source_id: {"uri": "s3://example/x.json", "sha256": "b" * 64}
            for source_id in (
                "Test:one",
                "Test:two",
                "Test:unrelated-retained-source",
            )
        }
    }
    assert declaration_audit.registry_index_gaps(frozenset({"Test:one", "Test:two"}), index) == []
