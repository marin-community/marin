# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The catalog binding for RL curricula."""

from collections import Counter
from pathlib import Path

import pytest
from marin.rl.curriculum import PoolRow

from experiments.post_training.task_curation.curriculum import (
    curriculum_entries,
    ordered_curriculum,
    served_catalog_sources,
    weighted_curriculum,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source

_DUMMY_INPUT = Path("/unused/numbers.txt")


def test_served_catalog_admission_and_pins_drive_curriculum_entries():
    source = {
        "status": "Available",
        "intended_use": "train",
        "dataset_id": "open-athena/retained",
        "dataset_revision": "a" * 40,
        "current_release": {"repo": "open-athena/retained", "revision": "a" * 40, "url": "https://example.org/release"},
        "pipeline": {
            "name": "retained-recipe",
            "version": "2026.11.13.1",
            "source": "open-athena/retained",
            "revision": "a" * 40,
        },
    }
    document = {
        "catalog_load": {"load_id": "r40-fixture"},
        "sources": {
            "available": source,
            "pending": {**source, "status": "Pending"},
            "excluded": {**source, "status": "Excluded", "license_exception": True},
        },
    }
    catalog = served_catalog_sources(document)
    (entry,) = curriculum_entries("mix", "1", weights={"available": 1.0}, catalog=catalog)
    assert (entry.source, entry.version, entry.pin.name, entry.pin.revision) == (
        "retained-recipe",
        "2026.11.13.1",
        "open-athena/retained",
        "a" * 40,
    )
    assert entry.recipe == "atlas:r40-fixture/retained-recipe@2026.11.13.1"
    for label in ("pending", "excluded"):
        with pytest.raises(ValueError, match="unknown RL source"):
            curriculum_entries("mix", "1", weights={label: 1.0}, catalog=catalog)
    campaign = served_catalog_sources(document, admitted_statuses=("Available", "Pending", "Excluded"))
    entries = curriculum_entries("campaign", "1", weights={label: 1 / 3 for label in campaign}, catalog=campaign)
    assert len(entries) == 3 and {entry.pin.revision for entry in entries} == {"a" * 40}
    assert document["sources"]["pending"]["status"] == "Pending"
    assert set(served_catalog_sources(document)) == {"available"}
    document["sources"]["available"] = {**source, "pipeline": {**source["pipeline"], "revision": "b" * 40}}
    with pytest.raises(ValueError, match="release and pipeline disagree"):
        served_catalog_sources(document)


def _catalog() -> dict[str, RlDataSource]:
    return {
        "numbers": number_source(_DUMMY_INPUT, "numbers"),
        "letters": number_source(_DUMMY_INPUT, "letters"),
    }


def _pool_rows(label: str, count: int) -> tuple[PoolRow, ...]:
    return tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))


def test_weighted_curriculum_records_pinned_source_identities():
    curriculum = weighted_curriculum(
        "users/tester/curricula/numbers-mix",
        "2026.11.18",
        weights={"numbers": 0.6, "letters": 0.4},
        seed=17,
        validation_size=200,
        pool_rows={"numbers": _pool_rows("numbers", 1000), "letters": _pool_rows("letters", 1000)},
        catalog=_catalog(),
    )
    assert curriculum.fingerprint
    assert {entry.label for entry in curriculum.entries} == {"numbers", "letters"}
    entry = curriculum.entries[0]
    assert entry.source in ("numbers", "letters")
    assert entry.version == "1"
    assert (entry.pin.name, entry.pin.revision, entry.pin.url) == (
        "local-numbers",
        "pinned",
        "https://example.org/numbers",
    )
    assert entry.recipe.endswith("numbers_pipeline.numbers_pipeline")
    assert curriculum.identity.startswith("users/tester/curricula/numbers-mix@2026.11.18:")
    # the drawn holdout follows the declared mixture and partitions the pools
    counts = Counter(row.label for row in curriculum.validation.rows)
    assert counts == {"numbers": 120, "letters": 80}
    assert curriculum.validation.seed == 17
    assert curriculum.validation.size == 200


def test_weighted_curriculum_derives_one_split_for_one_input():
    """The draw is deterministic for the seed, so record and pool data derive one split."""
    kwargs = {
        "weights": {"numbers": 0.6, "letters": 0.4},
        "seed": 17,
        "validation_size": 200,
        "pool_rows": {"numbers": _pool_rows("numbers", 1000), "letters": _pool_rows("letters", 1000)},
        "catalog": _catalog(),
    }
    assert weighted_curriculum("users/tester/curricula/numbers-mix", "2026.11.18", **kwargs).validation == (
        weighted_curriculum("users/tester/curricula/numbers-mix", "2026.11.18", **kwargs).validation
    )


def test_ordered_curriculum_declares_the_stage_sequence():
    curriculum = ordered_curriculum(
        "users/tester/curricula/numbers-ladder",
        "2026.11.18",
        stages=("numbers", "letters", "numbers"),
        quotas={"numbers": 8},
        seed=3,
        validation_size=10,
        pool_rows={"numbers": _pool_rows("numbers", 20), "letters": _pool_rows("letters", 20)},
        catalog=_catalog(),
    )
    assert {entry.label for entry in curriculum.entries} == {"numbers", "letters"}
    assert curriculum.sampling_config() == {
        "kind": "ordered",
        "stages": [
            {"source": "numbers", "quota": 8},
            {"source": "letters", "quota": None},
            {"source": "numbers", "quota": 8},
        ],
        "seed": 3,
    }
    # consumed prompts: numbers 8+8=16, letters 20; allocation of 10 follows 16:36 and 20:36
    assert Counter(row.label for row in curriculum.validation.rows) == {"numbers": 4, "letters": 6}


def test_curriculum_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown RL source 'ghost'"):
        weighted_curriculum(
            "users/tester/curricula/broken",
            "2026.11.18",
            weights={"ghost": 1.0},
            validation_size=10,
            pool_rows={"ghost": _pool_rows("ghost", 100)},
            catalog=_catalog(),
        )


def test_curriculum_rejects_unpinned_source():
    catalog = {**_catalog(), "inventory": RlDataSource(info=SourceInfo(id="fixture:inventory", title="x", origin="f"))}
    with pytest.raises(ValueError, match="no pinned dataset"):
        ordered_curriculum(
            "users/tester/curricula/unpinned",
            "2026.11.18",
            stages=("inventory",),
            validation_size=10,
            pool_rows={"inventory": _pool_rows("inventory", 100)},
            catalog=catalog,
        )


def test_ordered_curriculum_rejects_quotas_for_unstaged_sources():
    with pytest.raises(ValueError, match="no stage samples"):
        ordered_curriculum(
            "users/tester/curricula/stray-quota",
            "2026.11.18",
            stages=("numbers",),
            quotas={"letters": 4},
            validation_size=10,
            pool_rows={"numbers": _pool_rows("numbers", 100), "letters": _pool_rows("letters", 100)},
            catalog=_catalog(),
        )


def test_weighted_curriculum_rejects_pool_rows_outside_the_mixture():
    with pytest.raises(ValueError, match="does not declare"):
        weighted_curriculum(
            "users/tester/curricula/stray-rows",
            "2026.11.18",
            weights={"numbers": 1.0},
            seed=1,
            validation_size=10,
            pool_rows={"numbers": _pool_rows("numbers", 100), "letters": _pool_rows("letters", 100)},
            catalog=_catalog(),
        )
