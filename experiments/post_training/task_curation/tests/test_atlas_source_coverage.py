# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main declares every Atlas-admitted source at the Atlas pin.

The fixture pins one Atlas cut (origin/atlas-env-audit revision and the R40 admit
set). Regenerate it whenever the Atlas admits more sources: the coverage test then
fails until main declares the new sources, and the pin-parity test fails when a
declared pin drifts from the Atlas admitted pin. A waiver records a known mismatch
with its exact blocker; a waiver whose baseline moved, or whose mismatch is fixed,
fails the test so the record cannot rot.
"""

import json
from pathlib import Path

import pytest
from finestore.rl.source_identity import canonical_source_id

from experiments.post_training.task_curation.sources import all_sources

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "atlas_source_pins.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _durable_ref(source) -> dict:
    dataset = source.dataset
    pipeline = source.pipeline
    return {
        "dataset_repo": dataset.name if dataset else None,
        "dataset_revision": dataset.revision if dataset else None,
        "pipeline_files": list(source.files) if pipeline is not None else [],
        "pipeline_name": source.name if pipeline is not None else None,
        "pipeline_version": source.version if pipeline is not None else None,
    }


def test_every_atlas_admitted_source_is_declared_on_main():
    """An Atlas-admitted source missing from the catalog fails the coverage test."""
    fixture = _fixture()
    declared = {canonical_source_id(source.info.id) for source in all_sources().values()}
    missing = sorted(source_id for source_id in fixture["required_ids"] if canonical_source_id(source_id) not in declared)
    assert missing == []


def test_main_pins_match_the_atlas_admitted_pins():
    """A declared pin differs from the Atlas admitted pin only under a recorded waiver."""
    fixture = _fixture()
    waivers = fixture["pin_parity_waivers"]
    unknown = sorted(set(waivers) - set(fixture["required_ids"]))
    assert unknown == []

    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale_waivers, moved_waivers, unwaived = [], [], []
    for source_id, atlas_ref in fixture["pins"].items():
        key = canonical_source_id(source_id)
        current = _durable_ref(sources[key])
        if key in waivers:
            waiver = waivers[key]
            if current == waiver["atlas"]:
                stale_waivers.append(source_id)
            elif current != waiver["main"]:
                moved_waivers.append(source_id)
        elif current != atlas_ref:
            unwaived.append(source_id)

    assert stale_waivers == [], f"parity landed; drop the waivers: {stale_waivers}"
    assert moved_waivers == [], f"waived pins moved on main: {moved_waivers}"
    assert unwaived == [], f"pins differ from the Atlas admitted pins with no waiver: {unwaived}"


def test_the_fixture_covers_one_exact_atlas_cut():
    """The fixture names the Atlas revision and the R40 plan it was generated from."""
    fixture = _fixture()
    assert len(fixture["atlas_revision"]) == 40
    assert fixture["r40_plan"]["targets"]
    assert fixture["required_ids"]
    for source_id, origin in fixture["required_ids"].items():
        assert origin["source"] in {"atlas-admitted", "r40-target"}, source_id
        assert source_id in fixture["pins"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
