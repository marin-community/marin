# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main declares every Atlas-admitted source at the Atlas pin.

The fixture pins one Atlas cut (origin/atlas-env-audit revision and the R40 admit
set); its ``atlas_revision`` and ``r40_plan`` fields record where each value came
from. The fixture's bytes live in the pinned public Hugging Face dataset
``open-athena/atlas-source-pins``; the repository commits only the immutable
reference (revision, sha256, byte size), and the Atlas release step on
``atlas-env-audit`` republishes a refreshed cut as a new revision.

Three checks, all computed from the catalog:

- coverage: every Atlas-admitted source is declared on main, or carries a
  ``declaration_blockers`` entry — the import that would run it on main is still
  pending, so its record lands with its import;
- stale blockers: a source whose blocker entry still stands while main declares
  it — the ingestion PR that landed the import must remove the entry;
- pin parity: every declared source's durable reference (repo, revision, files,
  recipe, version) equals the Atlas admitted pin, unless a recorded waiver
  states the blocker; parity that has landed makes the waiver stale.

The gate runs in the default CI selection: the fixture is a public dataset
fetched anonymously, so the unit job can read it.
"""

import json
from functools import cache

import pytest
from finestore.rl.pinned_file import PinnedHfFile, read_pinned_hf_file
from finestore.rl.source_identity import canonical_source_id

from experiments.post_training.task_curation.sources import all_sources

ATLAS_SOURCE_PINS_REPO = "open-athena/atlas-source-pins"
ATLAS_SOURCE_PINS_REVISION = "53147ef2354b789879e2b93f7919accdd6a04c84"
ATLAS_SOURCE_PINS_SHA256 = "f0b0e08bc3e53260142323c3acd29e9da44e5e9ff8f26504d47472fafcec7f26"
ATLAS_SOURCE_PINS_BYTES = 170943


ATLAS_SOURCE_PINS_REFERENCE = PinnedHfFile(
    repo=ATLAS_SOURCE_PINS_REPO,
    revision=ATLAS_SOURCE_PINS_REVISION,
    filename="atlas_source_pins.json",
    sha256=ATLAS_SOURCE_PINS_SHA256,
    byte_size=ATLAS_SOURCE_PINS_BYTES,
)


@cache
def read_atlas_source_pins() -> dict:
    """Read the pinned fixture, verify its bytes, and return the decoded mapping."""
    return json.loads(read_pinned_hf_file(ATLAS_SOURCE_PINS_REFERENCE))


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
    fixture = read_atlas_source_pins()
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
    fixture = read_atlas_source_pins()
    blockers = fixture["declaration_blockers"]
    sources = {canonical_source_id(source.info.id): source for source in all_sources().values()}
    stale = sorted(source_id for source_id in blockers if canonical_source_id(source_id) in sources)
    assert stale == []


def test_main_pins_match_the_atlas_admitted_pins():
    """A declared pin differs from the Atlas admitted pin only under a recorded waiver."""
    fixture = read_atlas_source_pins()
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


def test_dapo_math_binds_to_its_retained_release():
    """The proof case: main's DAPO declaration selects the Atlas's deduplicated release.

    The raw pin (BytedTsinghua-SIA/DAPO-Math-17k, 1,791,700 raw rows) is superseded by
    the retained release the Atlas admitted (16,901 retained of the deduplicated
    population), and the durable reference, recipe and count follow it.
    """
    source = all_sources()["BytedTsinghua-SIA:dapo_math"]
    dataset = source.dataset
    assert (dataset.name, dataset.revision) == (
        "open-athena/marinskyrl-dapo-math-251da0b4-retained-20261113-v1",
        "36fa57456e941732aa805200cda98389734b5399",
    )
    assert (source.name, source.version, list(source.files)) == (
        "dapo_math-retained",
        "2026.11.13.1",
        ["tasks.parquet"],
    )
    assert source.info.count == 16901
    assert source.declaration.status == "Available"
    source.require_license()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
