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


def test_the_focus_crawl_reaches_the_store_through_the_exact_repack():
    """Both pinned dedup runs filed the focus crawl under its pre-#8111 extraction.

    Fuzzy verification repacks its own candidates, so only the exact marks need a
    step here, and the store has to read that step rather than the pin.
    """
    inputs = produce_store.store_inputs([*SOURCES, hero_data.FOCUS_SOURCE_NAME])

    assert inputs.repacked_sources == (hero_data.FOCUS_SOURCE_NAME,)
    assert inputs.exact_dups.name == f"datakit/repack_exact_dups/{hero_data.FOCUS_SOURCE_NAME}"
    assert inputs.exact_dups_pin.output_path.endswith(f"/datakit/{hero_data.EXACT_DUPS_ID}")
    assert inputs.exact_dups in produce_store.build_store_step(inputs, max_workers=4).deps


def test_a_run_without_the_focus_crawl_reads_the_pin_directly():
    inputs = produce_store.store_inputs(SOURCES)

    assert inputs.repacked_sources == ()
    assert inputs.exact_dups is inputs.exact_dups_pin


def test_the_repack_is_not_something_preflight_expects_to_exist():
    """It is the one input this run builds, so an absent output is not a problem."""
    inputs = produce_store.store_inputs([*SOURCES, hero_data.FOCUS_SOURCE_NAME])
    registered = {step.output_path for step in inputs.registered_steps()}

    assert inputs.exact_dups.output_path not in registered
    assert inputs.exact_dups_pin.output_path in registered


def test_the_repack_reads_the_pin_and_the_current_normalize():
    inputs = produce_store.store_inputs([hero_data.FOCUS_SOURCE_NAME])
    deps = {dep.output_path for dep in inputs.exact_dups.deps}

    assert deps == {
        hero_data.exact_dups().output_path,
        hero_data.normalized(hero_data.FOCUS_SOURCE_NAME).output_path,
    }


def test_repointing_the_exact_pin_moves_the_repack_and_the_store(monkeypatch):
    inputs = produce_store.store_inputs([*SOURCES, hero_data.FOCUS_SOURCE_NAME])
    before = produce_store.build_store_step(inputs, max_workers=4).output_path
    monkeypatch.setattr(hero_data, "EXACT_DUPS_ID", "global_exact_dedup_0ther")
    after_inputs = produce_store.store_inputs([*SOURCES, hero_data.FOCUS_SOURCE_NAME])

    assert after_inputs.exact_dups.output_path != inputs.exact_dups.output_path
    assert produce_store.build_store_step(after_inputs, max_workers=4).output_path != before


def test_running_the_repack_early_lands_where_the_store_will_look():
    """``--repack-only`` exists so the store serves it from cache later.

    That only holds if the two build the same identity, which means the sizing
    flags the early run takes must stay out of the step hash.
    """
    early = produce_store.build_focus_exact_repack_step(
        hero_data.exact_dups(), max_workers=8, worker_resources=produce_store.ResourceConfig(cpu=1, ram="2g")
    )
    from_store = produce_store.store_inputs([*SOURCES, hero_data.FOCUS_SOURCE_NAME]).exact_dups

    assert early.output_path == from_store.output_path
