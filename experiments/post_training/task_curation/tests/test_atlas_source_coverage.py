# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check catalog coverage and pins against the Atlas cut recorded in the fixture.

Regenerate the fixture with ``generate_atlas_source_pins.py`` when the cut changes.
"""

import json
from pathlib import Path

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
    fixture = _fixture()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale = sorted(source_id for source_id in blockers if canonical_source_id(source_id) in sources)
    assert stale == []


def test_main_pins_match_the_atlas_admitted_pins():
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
