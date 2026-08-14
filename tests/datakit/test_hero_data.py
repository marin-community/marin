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

import pytest

from experiments.datakit import hero_data

PREFIX = hero_data.MANIFEST_PREFIX
MANIFEST = hero_data.manifest_path()
EMBED_MANIFEST = hero_data.harrier_paths_path()


@pytest.fixture(autouse=True)
def _marin_prefix(monkeypatch):
    # Hero data has one CoreWeave location. Pin it instead of inheriting the test
    # process's prefix because some step hashes include resolved paths.
    monkeypatch.setenv("MARIN_PREFIX", PREFIX)


def _relative_paths() -> dict[str, str]:
    return {key: path.removeprefix(f"{PREFIX}/") for key, path in hero_data.all_paths().items()}


def test_paths_match_the_checked_in_manifest():
    """Regenerate with ``uv run python -m experiments.datakit.hero_data``.

    A diff here means a hero path moved. The stored data does not move with it,
    so regenerate only once the new paths are known to hold the data.
    """
    assert _relative_paths() == json.loads(MANIFEST.read_text())


def test_harrier_paths_are_complete_relative_and_include_focus():
    paths = json.loads(EMBED_MANIFEST.read_text())

    assert set(paths) == set(hero_data.source_names())
    assert all("://" not in path and not path.startswith("/") for path in paths.values())
    expected = f"{PREFIX}/datakit/embed/harrier-all/common-crawl-focus-2026-22_fc8cffa4"
    assert hero_data.harrier("common-crawl-focus-2026-22") == expected


def test_every_registered_source_has_every_stage():
    keys = set(_relative_paths())
    missing = {
        f"{stage}/{source}"
        for source in hero_data.source_names()
        for stage in ("normalized", "minhash", "tokenize.marin", "tokenize.nemotron")
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
