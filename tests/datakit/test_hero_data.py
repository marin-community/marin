# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path-identity tests for the hero data accessors.

Pure StepSpec construction -- no cluster, no data. ``hero_data`` exists so that
callers do not have to know which artifact version or upstream hash produced a
stage, which only holds while the paths it hands out keep pointing at the data
that was verified to live there. These tests pin that: a change that moves a
hero path fails here rather than silently repointing consumers at a path with
nothing behind it, or at a tokenize output built from a different normalize.
"""

import json
from dataclasses import replace

import pytest
from marin.datakit.sources import all_sources

from experiments.datakit import hero_data
from experiments.datakit.cluster.quality.fast_transformer import run as quality_run

PREFIX = hero_data.MANIFEST_PREFIX
MANIFEST = hero_data.manifest_path()
PINNED_MAPS = {
    "harrier": hero_data.harrier_paths_path(),
    "fusion_scores": hero_data.fusion_score_paths_path(),
    "content_type": hero_data.content_type_paths_path(),
}


@pytest.fixture(autouse=True)
def _marin_prefix(monkeypatch):
    # Hero data has one CoreWeave location. Pin it instead of inheriting the test
    # process's prefix because some step hashes include resolved paths.
    monkeypatch.setenv("MARIN_PREFIX", PREFIX)
    # The registry caches prefix-resolved GHALogs steps for the life of the process.
    all_sources.cache_clear()
    yield
    all_sources.cache_clear()


def _relative_paths() -> dict[str, str]:
    return {key: path.removeprefix(f"{PREFIX}/") for key, path in hero_data.all_paths().items()}


def test_paths_match_the_checked_in_manifest():
    """Regenerate with ``uv run python -m experiments.datakit.hero_data``.

    A diff here means a hero path moved. The stored data does not move with it,
    so regenerate only once the new paths are known to hold the data.
    """
    assert _relative_paths() == json.loads(MANIFEST.read_text())


@pytest.mark.parametrize("stage", sorted(PINNED_MAPS))
def test_pinned_maps_are_complete_and_relative(stage):
    paths = json.loads(PINNED_MAPS[stage].read_text())

    assert set(paths) == set(hero_data.source_names())
    assert all("://" not in path and not path.startswith("/") for path in paths.values())


def test_pinned_leaves_resolve_under_the_prefix():
    focus = "common-crawl-focus-2026-22"
    assert hero_data.harrier(focus).output_path == f"{PREFIX}/datakit/embed/harrier-all/{focus}_fc8cffa4"
    assert hero_data.fusion_scores(focus).output_path.startswith(f"{PREFIX}/datakit/quality/{focus}_")
    assert hero_data.content_type(focus).output_path.startswith(f"{PREFIX}/datakit/content-type/{focus}_")


def test_every_registered_source_has_every_stage():
    keys = set(_relative_paths())
    missing = {
        f"{stage}/{source}"
        for source in hero_data.source_names()
        for stage in (
            "normalized",
            "minhash",
            "tokenize.marin",
            "tokenize.nemotron",
            "harrier",
            "cluster_assign",
            "fusion_scores",
            "content_type",
            "quality",
        )
    } - keys
    assert not missing


def test_steps_refuse_to_run():
    # These describe data that already exists. A runner that executed one would
    # overwrite it, so every accessor must fail instead of producing output.
    steps = [
        hero_data.normalized("stack-v3"),
        hero_data.tokenized("stack-v3", hero_data.MARIN_TOKENIZER),
        hero_data.minhash("stack-v3"),
        hero_data.exact_dups(),
        hero_data.fuzzy_dups(),
        hero_data.domain_cluster_assignment(),
        hero_data.assigned_clusters("stack-v3"),
        hero_data.harrier("stack-v3"),
        hero_data.fusion_scores("stack-v3"),
        hero_data.content_type("stack-v3"),
        hero_data.quality("stack-v3"),
    ]
    for step in steps:
        with pytest.raises(AssertionError, match="must never execute"):
            step.fn(step.output_path)


def test_repointing_a_dedup_pin_changes_dependency_identity(monkeypatch):
    # ``hash_id`` ignores ``override_output_path`` and a dependent's cache key comes
    # from its deps' ``name_with_hash``. Without the pin in ``hash_attrs`` a repoint
    # left dependents free to reuse outputs computed against the previous dedup run.
    before = hero_data.fuzzy_dups()
    monkeypatch.setattr(hero_data, "FUZZY_DUPS_ID", "dedup_ffffffff")
    after = hero_data.fuzzy_dups()

    assert before.output_path != after.output_path
    assert before.name_with_hash != after.name_with_hash


def test_unknown_source_is_rejected():
    with pytest.raises(KeyError):
        hero_data.normalized("no-such-source")


@pytest.mark.parametrize(
    "field, value",
    [
        ("calibration_sha256", "0" * 64),
        ("name", "some-other-scorer"),
    ],
)
def test_quality_path_moves_with_the_calibration(field, value):
    """The bucket step's identity is the remap it applies; a refit moves the output."""
    base = hero_data.NEMOTRON_88K
    changed = replace(base, **{field: value})
    assert hero_data.quality("stack-v3", changed).output_path != hero_data.quality("stack-v3", base).output_path


def test_quality_path_moves_with_its_pinned_inputs(monkeypatch):
    # Repointing the fusion scores or the content types is a different dataset even
    # under the same calibration: the frozen inputs carry their paths in hash_attrs.
    before = hero_data.quality("stack-v3").output_path
    monkeypatch.setitem(hero_data.content_type_paths(), "stack-v3", "datakit/content-type/stack-v3_ffffffff")
    assert hero_data.quality("stack-v3").output_path != before


def test_fusion_scores_refuse_a_pin_that_is_another_model():
    """The pinned scores were written by one model; bucketing them under another lies."""
    other = replace(hero_data.NEMOTRON_88K, model_sha256="0" * 64)
    with pytest.raises(ValueError, match="score the corpus"):
        hero_data.quality("stack-v3", other)


def test_the_bucket_driver_resolves_to_the_registered_quality_path():
    (step,) = quality_run.build_bucket_steps(["stack-v3"])
    assert step.output_path == hero_data.quality("stack-v3").output_path
    assert step.fn is not None
