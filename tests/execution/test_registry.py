# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The artifact registry: build-once immutability, dev mutability, and provenance.

Drives the real lower -> StepRunner pipeline against a tmp prefix so the guard is
exercised exactly as it fires in production (before a cached SUCCESS is served).
"""

import pytest
from marin.execution.artifact import Artifact as ArtifactIO
from marin.execution.lazy import Dataset, Recipe, lower
from marin.execution.registry import ImmutableArtifactError, read_record
from marin.execution.step_runner import StepRunner


def _toy(version: str, payload: str) -> Dataset:
    """A toy artifact whose recipe (and thus fingerprint) is keyed by ``payload``."""
    return Dataset(
        name="datasets/toy",
        version=version,
        recipe=Recipe(
            fn=lambda config: config,
            build_config=lambda ctx: {"out": ctx.out, "payload": payload},
        ),
    )


def _run(artifact: Dataset) -> None:
    StepRunner().run([lower(artifact)])


def test_records_provenance_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    art = _toy("v1", "a")
    _run(art)

    record = read_record(f"{tmp_path}/datasets/toy/v1")
    assert record is not None
    assert (record.name, record.version, record.fingerprint) == ("datasets/toy", "v1", art.fingerprint())


def test_same_recipe_rerun_is_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    _run(_toy("v1", "a"))
    # Identical recipe + version: a cache hit, no error.
    _run(_toy("v1", "a"))


def test_rebuild_with_changed_recipe_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    _run(_toy("v1", "a"))

    # Same name@version, different recipe -> different fingerprint -> guarded.
    with pytest.raises(ImmutableArtifactError):
        _run(_toy("v1", "b"))


def test_dev_version_is_mutable(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    _run(_toy("dev", "a"))
    # A changed recipe under a dev version rebuilds in place instead of raising.
    _run(_toy("dev", "b"))

    saved = ArtifactIO.from_path(f"{tmp_path}/datasets/toy/dev")
    assert saved["payload"] == "b"
