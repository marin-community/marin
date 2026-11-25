# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main declares every Atlas-admitted source at the Atlas pin.

The fixture pins one Atlas cut (origin/atlas-env-audit revision and the R40 admit
set). Regenerate it whenever the Atlas admits more sources: the coverage test then
fails until main declares the new sources, and the pin-parity test fails when a
declared pin drifts from the Atlas admitted pin. A waiver records a known mismatch
with its exact main-side baseline and blocker; a waiver whose baseline moved, or
whose mismatch is fixed, fails the test so the record cannot rot.
"""

import json
from pathlib import Path

import pytest
from finestore.rl.source_identity import canonical_source_id

from experiments.post_training.task_curation.sources import all_sources

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "atlas_source_pins.json"

ATLAS_REVISION = "aa413cc1aa279dfc848031b22271b87543ea9c16"
"""The origin/atlas-env-audit HEAD the fixture's admitted set was computed at."""
R40_ARTIFACT_VERSION = "2026.11.25.1"
"""The r40.json artifact version the fixture's admit set was read from."""


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
    missing = sorted(
        source_id for source_id in fixture["required_ids"] if canonical_source_id(source_id) not in declared
    )
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
            if current == atlas_ref:
                stale_waivers.append(source_id)
            elif current != waiver["main"]:
                moved_waivers.append(source_id)
        elif current != atlas_ref:
            unwaived.append(source_id)

    assert stale_waivers == [], f"parity landed; drop the waivers: {stale_waivers}"
    assert moved_waivers == [], f"waived pins moved on main: {moved_waivers}"
    assert unwaived == [], f"pins differ from the Atlas admitted pins with no waiver: {unwaived}"


def test_the_fixture_is_cut_to_one_atlas_revision():
    """The fixture carries exactly the pinned Atlas cut and R40 plan version.

    Regenerating the fixture against a newer cut updates these two expectations in
    the same change, so a silently different cut cannot pass.
    """
    fixture = _fixture()
    assert fixture["atlas_revision"] == ATLAS_REVISION
    assert fixture["r40_plan"]["artifact_version"] == R40_ARTIFACT_VERSION
    assert sorted(fixture["r40_plan"]["targets"])
    assert all(origin["source"] in {"atlas-admitted", "r40-target"} for origin in fixture["required_ids"].values())
    for source_id, atlas_ref in fixture["pins"].items():
        assert atlas_ref["dataset_repo"], source_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
