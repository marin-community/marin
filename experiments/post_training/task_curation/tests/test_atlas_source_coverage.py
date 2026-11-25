# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main declares every Atlas-admitted source at the Atlas pin.

The fixture pins one Atlas cut (origin/atlas-env-audit revision and the R40 admit
set); its ``atlas_revision`` and ``r40_plan`` fields record where each value came
from. Regenerate it whenever the Atlas admits more sources, with
``tests/generate_atlas_source_pins.py``.

Three checks, all computed from the catalog:

- coverage: every Atlas-admitted source is declared on main, or carries a
  ``declaration_blockers`` entry — the import that would run it on main is still
  pending, and per the #11903 review resolution a record lands together with its
  import, never pipeline-less;
- stale blockers: a source whose blocker entry still stands while main declares
  it — the ingestion PR that landed the import must remove the entry;
- pin parity: every declared source's durable reference (repo, revision, files,
  recipe, version) equals the Atlas admitted pin, unless a recorded waiver
  states the blocker; parity that has landed makes the waiver stale.
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


def test_every_atlas_admitted_source_is_declared_on_main_or_recorded_as_blocked():
    """An Atlas-admitted source missing from the catalog fails unless its blocker stands."""
    fixture = _fixture()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id) for source in all_sources().values()}
    missing = sorted(
        source_id
        for source_id in fixture["required_ids"]
        if canonical_source_id(source_id) not in sources and source_id not in blockers
    )
    assert missing == []


def test_a_declaration_blocker_expires_when_main_declares_the_source():
    """The ingestion PR that lands the import removes the source's blocker entry."""
    fixture = _fixture()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale = sorted(source_id for source_id in blockers if canonical_source_id(source_id) in sources)
    assert stale == []


def test_main_pins_match_the_atlas_admitted_pins():
    """A declared pin differs from the Atlas admitted pin only under a recorded waiver."""
    fixture = _fixture()
    waivers = fixture["pin_parity_waivers"]
    blockers = fixture["declaration_blockers"]
    unknown = sorted(set(waivers) - set(fixture["required_ids"]) - set(blockers))
    assert unknown == []

    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale_waivers, unwaived = [], []
    for source_id, atlas_ref in fixture["pins"].items():
        key = canonical_source_id(source_id)
        source = sources.get(key)
        if source is None or key in blockers:
            continue
        current = _durable_ref(source)
        if key in waivers:
            if current == atlas_ref:
                stale_waivers.append(source_id)
        elif current != atlas_ref:
            unwaived.append(source_id)

    assert stale_waivers == [], f"parity landed; drop the waivers: {stale_waivers}"
    assert unwaived == [], f"pins differ from the Atlas admitted pins with no waiver: {unwaived}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
