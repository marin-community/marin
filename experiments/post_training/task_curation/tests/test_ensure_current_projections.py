# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The projection currency stage pairs every pin with its grader and curriculum receipts."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import ActorCensusInput

from experiments.post_training.task_curation import ensure_current_projections
from experiments.post_training.task_curation.ensure_current_projections import (
    CatalogSelection,
    DeclarationPin,
    ProjectionGap,
    ProjectionWaivers,
    apply_projection_waivers,
    projection_gaps,
    publication_preflight,
    summary,
)

DATASET = "open-athena/example"
REVISION = "a" * 40
PIPELINE = {"name": "example"}


def binding(revision=REVISION) -> ActorCensusInput:
    return ActorCensusInput(
        dataset_id=DATASET,
        dataset_revision=revision,
        converter_version="2026.11.13.1",
        converter_revision="b" * 40,
        measured_tasks=10,
        task_specs_sha256="c" * 40,
        evidence_url="memory://actor/.artifact.json",
    )


def declaration(source_id="Test:source", revision=REVISION, measured=True) -> DeclarationPin:
    return DeclarationPin(
        source_id=source_id,
        dataset_id=DATASET,
        dataset_revision=revision,
        pipeline=PIPELINE,
        actor_binding=binding(revision) if measured else None,
    )


class Receipt:
    """A minimal derived-contract receipt stand-in with the projection's input shape."""

    def __init__(self, source_id="Test:source", revision=REVISION):
        self.source_id = source_id
        self.summary = {"task_count": 10}
        self.input = binding(revision)


def curriculum_row(declaration: DeclarationPin, revision=None) -> dict:
    return {
        "declaration": {
            "dataset_id": declaration.dataset_id,
            "dataset_revision": revision or declaration.dataset_revision,
            "pipeline": declaration.pipeline,
        },
        "has_alignment": True,
    }


def test_a_current_pin_with_both_projections_has_no_gaps():
    source = declaration()
    gaps, notes = projection_gaps(
        [source],
        {"Test:source": [Receipt()]},
        {"Test:source": curriculum_row(source)},
    )
    assert gaps == []
    assert notes == []


def test_a_missing_derived_contract_receipt_is_a_gap():
    source = declaration()
    gaps, _ = projection_gaps([source], {}, {"Test:source": curriculum_row(source)})
    assert [(gap.projection, gap.state) for gap in gaps] == [("derived-contract", "missing")]


def test_a_receipt_at_an_older_pin_is_a_stale_pin_gap_naming_the_last_computed_revision():
    source = declaration()
    gaps, _ = projection_gaps(
        [source], {"Test:source": [Receipt(revision="9" * 40)]}, {"Test:source": curriculum_row(source)}
    )
    assert [(gap.projection, gap.state) for gap in gaps] == [("derived-contract", "stale-pin")]
    assert gaps[0].last_computed_revision == "9" * 40


def test_a_curriculum_row_at_an_older_pin_is_a_stale_pin_gap():
    source = declaration()
    gaps, _ = projection_gaps(
        [source], {"Test:source": [Receipt()]}, {"Test:source": curriculum_row(source, revision="8" * 40)}
    )
    assert [(gap.projection, gap.state) for gap in gaps] == [("curriculum", "stale-pin")]
    assert gaps[0].last_computed_revision == "8" * 40


def test_an_unmeasured_source_is_reported_not_failed():
    gaps, notes = projection_gaps([declaration(measured=False)], {}, {})
    assert gaps == []
    assert len(notes) == 2


def test_a_stale_served_catalog_is_its_own_gap():
    source = declaration()
    served = DeclarationPin(
        source_id=source.source_id,
        dataset_id=source.dataset_id,
        dataset_revision="7" * 40,
        pipeline=source.pipeline,
        actor_binding=None,
    )
    gaps, _ = projection_gaps(
        [source],
        {"Test:source": [Receipt(revision="9" * 40)]},
        {"Test:source": curriculum_row(source)},
        served_pins={"Test:source": served},
    )
    assert [(gap.projection, gap.state) for gap in gaps] == [("served-catalog", "stale-pin")]


def test_the_served_catalog_target_comes_from_config(monkeypatch, tmp_path):
    """The served-catalog checks read the applet named by the target config.

    While production is on the coordinator's hold the config points at the preview
    applet the landing lanes keep current; the production applet stays the default
    once the hold lifts.
    """
    config = tmp_path / "atlas-target-applet.json"
    config.write_text(json.dumps({"marina_url": "https://marina.example", "applet_id": "preview-applet-id"}))
    captured = {}

    def fake_query(service_url: str, applet_id: str):
        captured["url"] = service_url
        captured["applet_id"] = applet_id
        raise AssertionError("the target query must not contact the applet in this test")

    monkeypatch.setattr(ensure_current_projections, "TARGET_CONFIG", config)
    monkeypatch.setattr(ensure_current_projections, "applet_query", fake_query)
    assert ensure_current_projections.target_applet() == ("https://marina.example", "preview-applet-id")
    with pytest.raises(AssertionError):
        ensure_current_projections.atlas_applet_query()
    assert captured == {"url": "https://marina.example", "applet_id": "preview-applet-id"}


def test_the_target_config_defaults_to_production_when_the_hold_lifts(monkeypatch, tmp_path):
    """The committed config names production as the post-hold default target."""
    config = tmp_path / "atlas-target-applet.json"
    config.write_text(
        json.dumps(
            {
                "marina_url": ensure_current_projections.MARINA_URL,
                "applet_id": ensure_current_projections.ATLAS_APPLET_ID,
            }
        )
    )
    monkeypatch.setattr(ensure_current_projections, "TARGET_CONFIG", config)
    assert ensure_current_projections.target_applet() == (
        ensure_current_projections.MARINA_URL,
        ensure_current_projections.ATLAS_APPLET_ID,
    )


def test_the_preflight_refuses_and_names_the_pins_with_their_last_computed_revision():
    source = declaration()
    try:
        publication_preflight(
            [source],
            {"Test:source": [Receipt(revision="9" * 40)]},
            {"Test:source": curriculum_row(source)},
        )
        raise AssertionError("the preflight must refuse")
    except ValueError as error:
        assert "1 current pins lack a serving projection" in str(error)
        assert "Test:source" in str(error) and "last computed at " + "9" * 40 in str(error)


def projection_waivers() -> ProjectionWaivers:
    return ProjectionWaivers(
        marina_url="https://marina.example",
        applet_id="preview",
        rule="stale-or-missing-serving-projection-without-declaration-drift",
        reason=ensure_current_projections.PROJECTION_WAIVER_REASON,
    )


@pytest.mark.parametrize("load_id", ["lagging", "frozen-before-refresh", "3e3f46aebfe843ddb4c4be3eceb173dc"])
def test_new_selected_loads_remain_waived_until_the_strict_seal_event(load_id):
    source = declaration()
    notes = publication_preflight(
        [source],
        {"Test:source": [Receipt(revision="9" * 40)]},
        {"Test:source": curriculum_row(source)},
        projection_waivers(),
        drift_sources=frozenset(),
        selection=CatalogSelection("https://marina.example", "preview", load_id, refresh_sealed=False),
    )
    assert len(notes) == 1
    assert "Test:source:derived-contract/stale-pin" in notes[0]
    assert ensure_current_projections.PROJECTION_WAIVER_REASON in notes[0]


@pytest.mark.parametrize(
    "selection",
    [
        CatalogSelection("https://marina.example", "preview", "r31j-during-refresh", refresh_sealed=True),
        CatalogSelection("https://marina.example", "production", "lagging", refresh_sealed=False),
    ],
)
def test_a_sealed_refresh_or_different_target_cannot_reuse_a_projection_waiver(selection):
    source = declaration()
    with pytest.raises(ValueError, match="derived-contract/stale-pin: Test:source"):
        publication_preflight(
            [source],
            {"Test:source": [Receipt(revision="9" * 40)]},
            {"Test:source": curriculum_row(source)},
            projection_waivers(),
            selection=selection,
            drift_sources=frozenset(),
        )
    assert publication_preflight(
        [source],
        {"Test:source": [Receipt()]},
        {"Test:source": curriculum_row(source)},
        projection_waivers(),
        selection=selection,
        drift_sources=frozenset(),
    ) == ["Projection waivers expired or target differs; all projection checks remain blocking"]


def test_a_gap_created_after_the_rule_file_is_generated_is_waived(tmp_path):
    path = tmp_path / "rule.json"
    path.write_text(projection_waivers().model_dump_json())
    source = declaration(source_id="New:source-after-file-generation")
    notes = publication_preflight(
        [source],
        {},
        {},
        ProjectionWaivers.model_validate_json(path.read_bytes()),
        selection=CatalogSelection("https://marina.example", "preview", "lagging", refresh_sealed=False),
        drift_sources=frozenset(),
    )
    assert len(notes) == 2
    assert any("New:source-after-file-generation:derived-contract/missing" in note for note in notes)
    assert any("New:source-after-file-generation:curriculum/missing" in note for note in notes)


def test_the_same_source_declaration_drift_cannot_be_waived():
    with pytest.raises(ValueError, match="Declaration drift blocks publication: Test:source"):
        publication_preflight(
            [declaration()],
            {},
            {},
            projection_waivers(),
            selection=CatalogSelection("https://marina.example", "preview", "lagging", refresh_sealed=False),
            drift_sources=frozenset({"Test:source"}),
        )


def test_the_summary_counts_gaps_by_projection_and_state():
    source = declaration()
    report = summary(projection_gaps([source], {}, {})[0], [])
    assert report["counts"] == {"derived-contract.missing": 1, "curriculum.missing": 1}
    assert report["state"] == "incomplete"


@pytest.mark.parametrize("has_alignment", [True, False])
def test_curriculum_presence_uses_the_query_response_field(has_alignment):
    source = declaration()
    row = {**curriculum_row(source), "has_alignment": has_alignment}
    gaps, _ = projection_gaps([source], {"Test:source": [Receipt()]}, {"Test:source": row})
    assert [(gap.projection, gap.state) for gap in gaps] == ([] if has_alignment else [("curriculum", "stale-pin")])


@pytest.mark.parametrize("current_first", [True, False])
def test_an_archived_stale_receipt_cannot_hide_a_current_receipt(current_first):
    source = declaration()
    current = Receipt()
    current.input = current.input.model_copy(update={"evidence_url": "memory://other", "carry_forward": "proof"})
    stale = Receipt(revision="9" * 40)
    receipts = [current, stale] if current_first else [stale, current]
    gaps, _ = projection_gaps([source], {"Test:source": receipts}, {"Test:source": curriculum_row(source)})
    assert gaps == []


@pytest.mark.parametrize("field,value", [("dataset_revision", "9" * 40), ("pipeline", {"name": "other"})])
def test_served_declaration_fields_must_match_even_with_the_same_actor_binding(field, value):
    source = declaration()
    served = replace(source, **{field: value})
    gaps, _ = projection_gaps(
        [source],
        {"Test:source": [Receipt()]},
        {"Test:source": curriculum_row(source)},
        served_pins={"Test:source": served},
    )
    assert [(gap.projection, gap.state) for gap in gaps] == [("served-catalog", "stale-pin")]


@pytest.mark.parametrize("served_revision", [REVISION, "9" * 40])
def test_conflicting_current_contract_summaries_fail(served_revision):
    source = declaration()
    other = Receipt()
    other.summary = {"task_count": 10, "grader_type_counts": {"script": 10}}
    with pytest.raises(ValueError, match="Conflicting derived contract summaries"):
        publication_preflight(
            [source],
            {"Test:source": [Receipt(), other]},
            {"Test:source": curriculum_row(source)},
            projection_waivers(),
            drift_sources=frozenset(),
            selection=CatalogSelection("https://marina.example", "preview", "lagging", refresh_sealed=False),
            served_pins={"Test:source": replace(source, dataset_revision=served_revision)},
        )


def test_receipt_grouping_keeps_both_current_and_historical_inputs():
    current = Receipt()
    stale = Receipt(revision="9" * 40)
    grouped = ensure_current_projections.contract_receipt_map([current, stale])
    assert grouped["Test:source"] == [current, stale]
    assert ensure_current_projections.contract_state(declaration(), grouped) == ("current", "")


def test_served_contract_reader_applies_exact_corrections_before_parsing_legacy_json(monkeypatch):
    fixture = Path(__file__).parents[4] / "infra/marina/apps/rl_data_catalog/tests/fixtures/contract_supersessions.json"
    pair = json.loads(fixture.read_text())[0]
    historical = copy.deepcopy(pair["selected"])
    historical_revision = "f" * 40
    historical["input"]["dataset_revision"] = historical_revision
    rows = [
        {"source_id": pair["source_id"], "receipt": payload}
        for payload in [pair["previous"], pair["selected"], historical]
    ]
    monkeypatch.setattr(ensure_current_projections, "atlas_applet_query", lambda: lambda _sql, _params: {"rows": rows})
    receipts = ensure_current_projections.atlas_contract_receipts()[pair["source_id"]]
    assert len(receipts) == 2
    assert {receipt.input.dataset_revision for receipt in receipts} == {
        pair["selected"]["input"]["dataset_revision"],
        historical["input"]["dataset_revision"],
    }
    selected = next(receipt for receipt in receipts if receipt.input.dataset_revision != historical_revision)
    assert selected.summary.single_turn_count == pair["selected"]["summary"]["single_turn_count"]
    unlisted = copy.deepcopy(pair["selected"])
    unlisted["summary"]["single_turn_count"] -= 1
    rows.append({"source_id": pair["source_id"], "receipt": unlisted})
    with pytest.raises(ValueError, match="Conflicting derived contract summaries"):
        ensure_current_projections.atlas_contract_receipts()


def test_contract_receipts_read_in_bounded_key_windows(monkeypatch):
    """The archive read pages by receipt_id so no single query exceeds the response cap."""
    fixture = Path(__file__).parents[4] / "infra/marina/apps/rl_data_catalog/tests/fixtures/contract_supersessions.json"
    base = json.loads(fixture.read_text())[0]["selected"]
    rows = []
    for index in range(5):
        receipt = copy.deepcopy(base)
        receipt["input"]["dataset_revision"] = str(index)
        rows.append({"receipt_id": f"{index:064x}", "source_id": "Fixture:source", "receipt": receipt})
    seen_windows = []

    def fake_query(sql, parameters):
        seen_windows.append(parameters["after"])
        after = parameters["after"]
        page = [row for row in rows if row["receipt_id"] > after][: parameters["batch"]]
        return {"rows": page}

    monkeypatch.setattr(ensure_current_projections, "atlas_applet_query", lambda: fake_query)
    monkeypatch.setattr(ensure_current_projections, "APPLET_PAGE_SIZE", 2)
    receipts = ensure_current_projections.atlas_contract_receipts()
    # Every row crossed a window boundary exactly once, in key order.
    assert seen_windows == ["", rows[1]["receipt_id"], rows[3]["receipt_id"]]
    assert len(receipts) == 1


def post_seal_waiver(*source_ids: str) -> ProjectionWaivers:
    return ProjectionWaivers(
        marina_url="https://marina.example",
        applet_id="preview",
        rule="post-seal-declaration-missing-served-projection",
        reason="source declared after the sealed refresh; served-catalog projection pending; ruling 2026-11-20",
        source_ids=source_ids,
    )


def test_post_seal_waiver_covers_named_sources_sealed_refresh_projections():
    """Ruling 68 + ruling 2026-11-25 13:00 EST: a source declared after the sealed refresh
    lands with a waiver naming its exact ID, covering its sealed-refresh projection
    findings — the served-catalog, curriculum and derived-contract projections are the
    same pending state (they all catch up in the one projection pass the expiry forces).
    Other sources stay blocking."""
    source = declaration()
    gaps = [
        ProjectionGap(
            source_id=source.source_id, projection="served-catalog", state="missing", last_computed_revision=None
        ),
        ProjectionGap(
            source_id=source.source_id, projection="served-catalog", state="stale-pin", last_computed_revision="8" * 40
        ),
        ProjectionGap(
            source_id=source.source_id, projection="derived-contract", state="stale-pin", last_computed_revision="9" * 40
        ),
        ProjectionGap(source_id=source.source_id, projection="curriculum", state="missing", last_computed_revision=None),
    ]
    result = apply_projection_waivers(gaps, post_seal_waiver(source.source_id), None, None)
    assert not result.blocking
    assert [(record.projection, record.state) for record in result.waived] == [
        ("served-catalog", "missing"),
        ("served-catalog", "stale-pin"),
        ("derived-contract", "stale-pin"),
        ("curriculum", "missing"),
    ]


def test_post_seal_waiver_rejects_unnamed_sources():
    """Ruling 68: the waiver names the exact source IDs; an unnamed source's gap stays blocking."""
    declared = declaration()
    other_id = "Test:other-source"
    gaps = [
        ProjectionGap(
            source_id=declared.source_id, projection="served-catalog", state="missing", last_computed_revision=None
        ),
        ProjectionGap(source_id=other_id, projection="served-catalog", state="missing", last_computed_revision=None),
    ]
    result = apply_projection_waivers(gaps, post_seal_waiver(declared.source_id), None, None)
    assert [gap.source_id for gap in result.blocking] == [other_id]
    assert [record.source_id for record in result.waived] == [declared.source_id]


def test_post_seal_waiver_rejects_other_finding_classes():
    """Ruling 68 + ruling 2026-11-25 13:00 EST: the named sources' sealed-refresh
    projections ride the waiver, but other finding classes stay blocking — a
    served-catalog gap for a source the waiver does NOT name stays blocking."""
    source = declaration()
    gaps = [
        ProjectionGap(
            source_id=source.source_id, projection="served-catalog", state="stale-pin", last_computed_revision="9" * 40
        ),
        ProjectionGap(
            source_id="Test:other-source", projection="served-catalog", state="missing", last_computed_revision=None
        ),
    ]
    result = apply_projection_waivers(gaps, post_seal_waiver(source.source_id), None, None)
    assert [gap.source_id for gap in result.blocking] == ["Test:other-source"]
    assert [record.source_id for record in result.waived] == [source.source_id]


def test_post_seal_waiver_without_named_source_ids_is_rejected():
    """Ruling 68: the waiver must name the exact source IDs; an empty list is refused."""
    gaps = [
        ProjectionGap(source_id="Test:source", projection="served-catalog", state="missing", last_computed_revision=None)
    ]
    with pytest.raises(ValueError, match="must name the exact declared source IDs"):
        apply_projection_waivers(gaps, post_seal_waiver(), None, None)
