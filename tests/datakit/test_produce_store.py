# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Graph-shape tests for the hero store entry point.

Pure StepSpec construction -- no cluster, no data. The point of
``produce_store`` is that the store reads stages that already exist, so what
these tests pin is that it *addresses* them and never offers to rebuild one: a
dependency that can execute is a dependency that can overwrite a stage the run
depends on. They also pin the two ways a subset run diverges from a full one,
because a subset is how anyone will smoke-test this before committing the fleet.
"""

import json
import pathlib

import pytest
from marin.datakit.sources import all_sources

from experiments.datakit import hero_data, produce_store

PREFIX = hero_data.MANIFEST_PREFIX
SOURCES = ["stack-v3", "nsf_awards"]


@pytest.fixture(autouse=True)
def _registered(monkeypatch, tmp_path):
    """Stand in for the two stages whose producing jobs have not landed yet."""
    monkeypatch.setenv("MARIN_PREFIX", PREFIX)
    all_sources.cache_clear()

    decon_map = tmp_path / "hero_data_decon_paths.json"
    decon_map.write_text(json.dumps({name: f"datakit/decontam/{name}_deadbeef" for name in hero_data.source_names()}))
    hero_data.decon_paths.cache_clear()
    monkeypatch.setattr(hero_data, "decon_paths_path", lambda: decon_map)
    monkeypatch.setattr(hero_data, "VERIFIED_FUZZY_DUPS_ID", "verify_fuzzy_dups_c0ffee")
    yield
    hero_data.decon_paths.cache_clear()
    all_sources.cache_clear()


def test_pending_lists_unregistered_stages(monkeypatch):
    """Both jobs are in flight at once, so one run has to name both."""
    monkeypatch.setattr(hero_data, "VERIFIED_FUZZY_DUPS_ID", None)
    monkeypatch.setattr(hero_data, "decon_paths_path", lambda: pathlib.Path("/nonexistent.json"))
    hero_data.decon_paths.cache_clear()

    stages = {item.stage for item in produce_store.pending()}
    assert stages == {"verified fuzzy duplicates", "decontamination"}


def test_nothing_registered_is_pending_once_the_pins_are_set():
    assert produce_store.pending() == []


def test_store_depends_on_every_stage_of_every_source():
    inputs = produce_store.store_inputs(SOURCES)
    step = produce_store.build_store_step(inputs, max_workers=4)

    # Four per-source stages plus the two global dedup artifacts.
    assert len(step.deps) == len(SOURCES) * 4 + 2
    assert step.name == "datakit/store"


def test_dependencies_refuse_to_run():
    """A hero dep that can execute can overwrite the stage the store reads."""
    inputs = produce_store.store_inputs(SOURCES)
    for dep in produce_store.build_store_step(inputs, max_workers=4).deps:
        with pytest.raises(AssertionError, match="must never execute"):
            dep.fn(dep.output_path)


def test_repointing_a_pin_moves_the_store(monkeypatch):
    """The store's identity is its inputs', so a new dedup run is a new store."""
    before = produce_store.build_store_step(produce_store.store_inputs(SOURCES), max_workers=4).output_path
    monkeypatch.setattr(hero_data, "VERIFIED_FUZZY_DUPS_ID", "verify_fuzzy_dups_0ther")
    after = produce_store.build_store_step(produce_store.store_inputs(SOURCES), max_workers=4).output_path

    assert before != after


def test_a_subset_is_a_different_store():
    """Otherwise a smoke run over two sources serves its cache to the full run."""
    subset = produce_store.build_store_step(produce_store.store_inputs(SOURCES), max_workers=4)
    single = produce_store.build_store_step(produce_store.store_inputs(SOURCES[:1]), max_workers=4)

    assert subset.output_path != single.output_path


def test_worker_shape_does_not_move_the_store():
    """Sizing is an execution knob. Rerunning bigger must not orphan the cache."""
    inputs = produce_store.store_inputs(SOURCES)
    small = produce_store.build_store_step(inputs, max_workers=4)
    large = produce_store.build_store_step(inputs, max_workers=256)

    assert small.output_path == large.output_path


def test_quality_must_match_the_tokenization_it_scored():
    """The store joins quality onto tokenize shards and then checks ids agree."""
    with pytest.raises(ValueError, match="not the one"):
        produce_store.store_inputs(SOURCES, tokenizer=hero_data.MARIN_TOKENIZER)


def test_unknown_source_is_rejected():
    with pytest.raises(KeyError):
        produce_store.store_inputs(["no-such-source"])
