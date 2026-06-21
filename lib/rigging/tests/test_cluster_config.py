# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster storage profiles and user-home path resolution."""

import pytest
import rigging.cluster_config as cluster_config
from rigging.cluster_config import (
    DEFAULT_STORAGE_PROFILE,
    StorageProfile,
    load_cluster_config,
    reset_cluster_config_cache,
)
from rigging.config_discovery import find_project_root


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset config-discovery and cluster-config caches around each test."""
    find_project_root.cache_clear()
    reset_cluster_config_cache()
    yield
    reset_cluster_config_cache()


def test_default_profile_resolved_root_is_marin_prefix(monkeypatch):
    """With no region and no env, the default profile root == marin_prefix()."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(cluster_config, "marin_region", lambda: None)
    monkeypatch.setattr(cluster_config, "marin_prefix", lambda: "/tmp/marin")

    assert DEFAULT_STORAGE_PROFILE.resolved_root() == "/tmp/marin"


def test_load_cluster_config_no_cluster_returns_default(monkeypatch):
    """With no cluster arg and no MARIN_CLUSTER env, returns DEFAULT_STORAGE_PROFILE."""
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    assert load_cluster_config() is DEFAULT_STORAGE_PROFILE


def test_marin_config_matches_default_profile():
    """Drift guard: config/marin.yaml parses to a profile equal to the default.

    The committed canonical template must stay byte-for-byte consistent with the
    hard-coded ``DEFAULT_STORAGE_PROFILE`` derived from filesystem.py constants.
    """
    profile = load_cluster_config("marin")

    assert profile.region_buckets == DEFAULT_STORAGE_PROFILE.region_buckets
    assert profile.ttl_days == DEFAULT_STORAGE_PROFILE.ttl_days
    assert profile.scheme == DEFAULT_STORAGE_PROFILE.scheme
    assert profile.user_segment == DEFAULT_STORAGE_PROFILE.user_segment
    assert profile.temp_path == DEFAULT_STORAGE_PROFILE.temp_path


def test_coreweave_config_parses_root_and_scheme():
    """config/coreweave.yaml carries an explicit single-prefix root and s3 scheme."""
    profile = load_cluster_config("coreweave")

    assert profile.scheme == "s3"
    assert profile.root == "s3://marin-na/marin"
    assert profile.region_buckets == {"na": "marin-na"}
    assert profile.ttl_days == (1, 2, 3, 7)


def test_marin_cluster_env_selects_config(monkeypatch):
    """MARIN_CLUSTER env selects the cluster config when no arg is given."""
    monkeypatch.setenv("MARIN_CLUSTER", "coreweave")
    profile = load_cluster_config()
    assert profile.root == "s3://marin-na/marin"


def test_resolved_root_env_wins_over_config_root(monkeypatch):
    """MARIN_PREFIX env wins over a config-supplied root."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://override-bucket/data")
    profile = StorageProfile(region_buckets={}, root="s3://marin-na/marin")

    assert profile.resolved_root() == "gs://override-bucket/data"


def test_resolved_root_uses_config_root_when_env_unset(monkeypatch):
    """The config root is used when MARIN_PREFIX is unset."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    profile = StorageProfile(region_buckets={}, scheme="s3", root="s3://marin-na/marin")

    assert profile.resolved_root() == "s3://marin-na/marin"


def test_resolved_root_selects_region_local_bucket(monkeypatch):
    """With no env/root, a region match drives a region-local bucket selection."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(cluster_config, "marin_region", lambda: "us-east5")

    assert DEFAULT_STORAGE_PROFILE.resolved_root() == "gs://marin-us-east5"


def test_resolved_root_unknown_region_falls_back_to_marin_prefix(monkeypatch):
    """An unknown region (no bucket match) falls back to marin_prefix()."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(cluster_config, "marin_region", lambda: "antarctica-south1")
    monkeypatch.setattr(cluster_config, "marin_prefix", lambda: "gs://marin-antarctica-south1")

    assert DEFAULT_STORAGE_PROFILE.resolved_root() == "gs://marin-antarctica-south1"


def test_user_home_joins_segment_and_user(monkeypatch):
    """user_home joins resolved_root / user_segment / user."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(cluster_config, "marin_region", lambda: "us-central1")

    assert DEFAULT_STORAGE_PROFILE.user_home("alice") == "gs://marin-us-central1/users/alice"


def test_user_home_respects_explicit_base():
    """An explicit base overrides resolved_root() for user_home."""
    assert DEFAULT_STORAGE_PROFILE.user_home("bob", base="gs://b") == "gs://b/users/bob"


def test_shared_root_defaults_to_resolved_root(monkeypatch):
    """shared_root() returns resolved_root() when no base is supplied."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(cluster_config, "marin_region", lambda: "us-west4")

    assert DEFAULT_STORAGE_PROFILE.shared_root() == "gs://marin-us-west4"
    assert DEFAULT_STORAGE_PROFILE.shared_root(base="gs://explicit") == "gs://explicit"
