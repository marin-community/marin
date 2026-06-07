# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canary output-directory wipe in the Grug multislice smoke ferry."""

from pathlib import Path

from experiments.ferries.grug_multislice_smoke import wipe_path_if_exists


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
