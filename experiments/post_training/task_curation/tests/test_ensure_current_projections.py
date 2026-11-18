# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The projection currency stage pairs every pin with its grader and curriculum receipts."""

import json

import pytest
from finestore.rl.curation import ActorCensusInput

from experiments.post_training.task_curation import ensure_current_projections
from experiments.post_training.task_curation.ensure_current_projections import (
    DeclarationPin,
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
        self.input = binding(revision)


def curriculum_row(declaration: DeclarationPin, revision=None) -> dict:
    return {
        "declaration": {
            "dataset_id": declaration.dataset_id,
            "dataset_revision": revision or declaration.dataset_revision,
            "pipeline": declaration.pipeline,
        },
        "alignment": {"area": "math"},
    }


def test_a_current_pin_with_both_projections_has_no_gaps():
    source = declaration()
    gaps, notes = projection_gaps(
        [source],
        {"Test:source": Receipt()},
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
        [source], {"Test:source": Receipt(revision="9" * 40)}, {"Test:source": curriculum_row(source)}
    )
    assert [(gap.projection, gap.state) for gap in gaps] == [("derived-contract", "stale-pin")]
    assert gaps[0].last_computed_revision == "9" * 40


def test_a_curriculum_row_at_an_older_pin_is_a_stale_pin_gap():
    source = declaration()
    gaps, _ = projection_gaps(
        [source], {"Test:source": Receipt()}, {"Test:source": curriculum_row(source, revision="8" * 40)}
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
        {"Test:source": Receipt(revision="9" * 40)},
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
            {"Test:source": Receipt(revision="9" * 40)},
            {"Test:source": curriculum_row(source)},
            frozenset(),
        )
        raise AssertionError("the preflight must refuse")
    except ValueError as error:
        assert "1 current pins lack a serving projection" in str(error)
        assert "Test:source" in str(error) and "last computed at " + "9" * 40 in str(error)


def test_the_preflight_honors_typed_waivers():
    source = declaration()
    notes = publication_preflight(
        [source],
        {"Test:source": Receipt(revision="9" * 40)},
        {"Test:source": curriculum_row(source)},
        frozenset({"Test:source:derived-contract"}),
    )
    assert notes == []


def test_the_summary_counts_gaps_by_projection_and_state():
    source = declaration()
    report = summary(projection_gaps([source], {}, {})[0], [])
    assert report["counts"] == {"derived-contract.missing": 1, "curriculum.missing": 1}
    assert report["state"] == "incomplete"
