# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace-snapshot cache resolution in :mod:`levanter.model_loading`.

The model-building paths require a device mesh and real checkpoints, so these
tests cover the network-free decision of *whether* and *where* to mirror a
checkpoint before loading it.
"""

import pytest

from levanter import model_loading


@pytest.mark.parametrize(
    "checkpoint",
    [
        "gs://bucket/snapshot",  # object store
        "s3://bucket/snapshot",  # object store
        "hf://org/model",  # explicit fsspec HF URL
        "/local/checkpoint/dir",  # absolute local path
        "./relative/dir",  # relative local path
    ],
)
def test_hf_cache_path_loads_object_store_and_local_paths_in_place(checkpoint):
    # A path that already names a snapshot must not be mirrored; mirroring it would
    # try to treat the path as a HuggingFace repo id and fail.
    assert model_loading._hf_cache_path(checkpoint, cache_ttl_days=30) is None


def test_hf_cache_path_disabled_returns_none():
    assert model_loading._hf_cache_path("org/model", cache_ttl_days=None) is None


def test_hf_cache_path_mirrors_repo_id_with_revision_scoped_slug(monkeypatch):
    ttl_seen = []

    def fake_bucket(ttl_days, prefix):
        ttl_seen.append(ttl_days)
        return f"gs://temp/{prefix}"

    monkeypatch.setattr(model_loading, "marin_temp_bucket", fake_bucket)

    base = model_loading._hf_cache_path("Qwen/Qwen3-0.6B", cache_ttl_days=7)
    revisioned = model_loading._hf_cache_path("Qwen/Qwen3-0.6B@abc123", cache_ttl_days=7)

    assert ttl_seen == [7, 7]
    # A bare repo id is mirrored, and a pinned revision lands in a distinct cache dir
    # so two revisions of one model never share a snapshot.
    assert base is not None and revisioned is not None
    assert base != revisioned
