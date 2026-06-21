# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster DataConfig: the active-config accessor, prefix
resolution, per-user homes, and YAML cluster overrides."""

import pytest
import rigging.filesystem as fs
from rigging.config_discovery import find_project_root
from rigging.filesystem import (
    DEFAULT_DATA_CONFIG,
    DataConfig,
    data_config,
    load_cluster_config,
    marin_prefix,
    reset_data_config_cache,
    use_data_config,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset config-discovery and data-config caches around each test."""
    find_project_root.cache_clear()
    reset_data_config_cache()
    yield
    reset_data_config_cache()


def test_data_config_defaults_to_marin(monkeypatch):
    """With no override and no MARIN_CLUSTER, the active config is the marin default."""
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    assert data_config() is DEFAULT_DATA_CONFIG


def test_use_data_config_overrides_then_restores():
    """use_data_config binds a config for the block, then restores the previous one."""
    override = DataConfig(region_buckets={}, scheme="s3", root="s3://other/data")
    with use_data_config(override):
        assert data_config() is override
    assert data_config() is DEFAULT_DATA_CONFIG


def test_marin_prefix_routes_through_active_config():
    """marin_prefix() is exactly the active config's resolved_root()."""
    override = DataConfig(region_buckets={}, root="gs://pinned/root")
    with use_data_config(override):
        assert marin_prefix() == "gs://pinned/root"


def test_resolved_root_env_wins(monkeypatch):
    """MARIN_PREFIX env wins over an explicit root."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://override-bucket/data")
    config = DataConfig(region_buckets={}, root="s3://marin-na/marin")
    assert config.resolved_root() == "gs://override-bucket/data"


def test_resolved_root_uses_explicit_root(monkeypatch):
    """An explicit root is used when MARIN_PREFIX is unset."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    config = DataConfig(region_buckets={}, scheme="s3", root="s3://marin-na/marin")
    assert config.resolved_root() == "s3://marin-na/marin"


def test_resolved_root_selects_region_local_bucket(monkeypatch):
    """A detected metadata region drives a region-local bucket selection."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: "us-east5")
    assert DEFAULT_DATA_CONFIG.resolved_root() == "gs://marin-us-east5"


def test_resolved_root_constructs_default_for_unmapped_region(monkeypatch):
    """A detected-but-unmapped region constructs gs://marin-{region}."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: "antarctica-south1")
    assert DEFAULT_DATA_CONFIG.resolved_root() == "gs://marin-antarctica-south1"


def test_resolved_root_local_fallback(monkeypatch):
    """With no env and no detectable region, falls back to /tmp/marin."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: None)
    assert DEFAULT_DATA_CONFIG.resolved_root() == "/tmp/marin"


def test_user_home_joins_segment_and_user(monkeypatch):
    """user_home joins resolved_root / user_segment / user."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: "us-central1")
    assert DEFAULT_DATA_CONFIG.user_home("alice") == "gs://marin-us-central1/users/alice"


def test_user_home_respects_explicit_base():
    """An explicit base overrides resolved_root() for user_home."""
    assert DEFAULT_DATA_CONFIG.user_home("bob", base="gs://b") == "gs://b/users/bob"


def test_load_coreweave_config_parses_root_and_scheme():
    """config/coreweave.yaml carries an explicit single-prefix root and s3 scheme."""
    config = load_cluster_config("coreweave")
    assert config.scheme == "s3"
    assert config.root == "s3://marin-na/marin"
    assert config.region_buckets == {"na": "marin-na"}
    assert config.ttl_days == (1, 2, 3, 7)


def test_marin_cluster_env_selects_config(monkeypatch):
    """MARIN_CLUSTER env selects the cluster config for the active accessor."""
    monkeypatch.setenv("MARIN_CLUSTER", "coreweave")
    assert data_config().root == "s3://marin-na/marin"


def test_marin_yaml_matches_default_config():
    """Drift guard: config/marin.yaml parses to the in-code marin default."""
    config = load_cluster_config("marin")
    assert config.region_buckets == DEFAULT_DATA_CONFIG.region_buckets
    assert config.ttl_days == DEFAULT_DATA_CONFIG.ttl_days
    assert config.scheme == DEFAULT_DATA_CONFIG.scheme
    assert config.user_segment == DEFAULT_DATA_CONFIG.user_segment
    assert config.temp_path == DEFAULT_DATA_CONFIG.temp_path
