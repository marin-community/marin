# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json

import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from test_utils import use_test_mesh

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.checkpoint_manifest import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointArray,
    CheckpointManifest,
    manifest_path,
    read_manifest,
    write_manifest,
)


def _manifest() -> CheckpointManifest:
    return CheckpointManifest(
        format_version=CHECKPOINT_FORMAT_VERSION,
        array_driver="zarr3",
        kvstore_driver="ocdbt",
        arrays=(
            CheckpointArray(path="model/embed", shape=(128, 16), dtype="float32", chunk_shape=(16, 16)),
            CheckpointArray(path="step", shape=(), dtype="int32", chunk_shape=()),
        ),
    )


def test_manifest_survives_a_json_roundtrip(tmp_path):
    write_manifest(str(tmp_path), _manifest())

    assert read_manifest(str(tmp_path)) == _manifest()


def test_read_manifest_is_none_for_a_checkpoint_written_before_manifests(tmp_path):
    assert read_manifest(str(tmp_path)) is None


def test_reading_a_newer_format_version_fails_loudly(tmp_path):
    payload = _manifest().to_json()
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    (tmp_path / "manifest.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Upgrade levanter"):
        read_manifest(str(tmp_path))


def test_saving_writes_a_manifest_listing_every_array(tmp_path):
    with use_test_mesh():
        A = hax.Axis("A", 8)
        state = {"model": {"w": hax.zeros(A)}, "step": jnp.array(3)}

        save_checkpoint(state, step=3, checkpoint_path=str(tmp_path))

        manifest = read_manifest(str(tmp_path))
        assert manifest is not None
        assert manifest.format_version == CHECKPOINT_FORMAT_VERSION
        assert manifest.array_paths == {"model/w", "step"}
        assert {array.path: array.shape for array in manifest.arrays} == {"model/w": (8,), "step": ()}


def test_checkpoints_without_a_manifest_still_load(tmp_path):
    """Checkpoints predating the manifest have none, so the reader has to fall back."""
    with use_test_mesh():
        A = hax.Axis("A", 8)
        state = {"model": {"w": hax.full(A, 5.0)}}

        save_checkpoint(state, step=0, checkpoint_path=str(tmp_path))
        (tmp_path / manifest_path("")).unlink()
        assert read_manifest(str(tmp_path)) is None

        restored = load_checkpoint({"model": {"w": hax.zeros(A)}}, str(tmp_path))

        assert jax.numpy.array_equal(restored["model"]["w"].array, state["model"]["w"].array)
