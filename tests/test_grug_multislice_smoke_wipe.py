# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canary output-directory wipe in the Grug multislice smoke ferry."""

from pathlib import Path

import pytest

from experiments.ferries.grug_multislice_smoke import _override_output_path, wipe_path_if_exists


def test_wipe_path_if_exists_removes_existing_tree(tmp_path: Path):
    """An existing directory and its contents are removed recursively."""
    canary_root = tmp_path / "canary-output"
    checkpoints = canary_root / "checkpoints" / "step-5" / "params" / "token_embed"
    checkpoints.mkdir(parents=True)
    (checkpoints / "zarr.json").write_text('{"chunk_shape": [128256, 32]}')
    (canary_root / "executor_info.json").write_text("{}")
    assert canary_root.exists()

    wipe_path_if_exists(str(canary_root))

    assert not canary_root.exists()


def test_wipe_path_if_exists_is_noop_when_missing(tmp_path: Path):
    """A missing path is a no-op (first canary run, or already-clean state)."""
    missing = tmp_path / "never-existed"
    assert not missing.exists()

    wipe_path_if_exists(str(missing))

    # No exception, and the parent tmp_path is still intact.
    assert tmp_path.exists()
    assert not missing.exists()


def test_override_output_path_unset_returns_none(monkeypatch: pytest.MonkeyPatch):
    """When the env var is unset, the override resolves to None (use default path)."""
    monkeypatch.delenv("GRUG_MULTISLICE_OUTPUT_PATH", raising=False)
    assert _override_output_path() is None


def test_override_output_path_empty_value_raises(monkeypatch: pytest.MonkeyPatch):
    """An empty env value would point the wipe at the MARIN_PREFIX root; reject it."""
    monkeypatch.setenv("GRUG_MULTISLICE_OUTPUT_PATH", "")
    with pytest.raises(ValueError, match="GRUG_MULTISLICE_OUTPUT_PATH is set but empty"):
        _override_output_path()


def test_override_output_path_nonempty_value_passes_through(monkeypatch: pytest.MonkeyPatch):
    """A non-empty env value is returned verbatim."""
    monkeypatch.setenv("GRUG_MULTISLICE_OUTPUT_PATH", "gs://some-bucket/path")
    assert _override_output_path() == "gs://some-bucket/path"
