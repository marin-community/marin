# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster DataConfig: the active-config accessor, YAML loading and
field-default parsing, and prefix resolution."""

import pytest
import rigging.filesystem as fs
from rigging.config_discovery import find_project_root
from rigging.filesystem import (
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


# --- loading / active-config selection -------------------------------------


def test_default_loads_marin_config_not_fallback(monkeypatch):
    """No override and no MARIN_CLUSTER -> the marin cluster loaded from YAML.

    A non-empty region_buckets distinguishes "loaded config/marin.yaml" from the
    empty in-code fallback, which is the behavior change being guarded.
    """
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    config = data_config()
    assert config is load_cluster_config("marin")  # default resolves to the marin cluster
    assert config.region_buckets  # loaded the file, not the empty fallback


def test_committed_cluster_configs_load(monkeypatch):
    """The committed marin and coreweave configs parse and are the expected kind."""
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    assert load_cluster_config("marin").region_buckets  # region-local gs cluster
    assert load_cluster_config("coreweave").root is not None  # single-prefix s3 cluster


def test_marin_cluster_env_selects_named_config(monkeypatch):
    """MARIN_CLUSTER routes the active accessor to that cluster's config."""
    monkeypatch.setenv("MARIN_CLUSTER", "coreweave")
    assert data_config() is load_cluster_config("coreweave")


def test_missing_named_cluster_raises():
    """A missing non-default cluster raises FileNotFoundError (the load contract)."""
    with pytest.raises(FileNotFoundError):
        load_cluster_config("this-cluster-does-not-exist")


def test_cluster_yaml_sets_given_fields_and_defaults_the_rest(tmp_path, monkeypatch):
    """A data: block overrides the fields it sets; absent fields take DataConfig defaults."""
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    (cluster_dir / "synth.yaml").write_text(
        "iris: synth\n"
        "data:\n"
        "  scheme: s3\n"
        "  region_buckets: {na: synth-na}\n"
        "  root: s3://synth-na/data\n"
        "  temp: {ttl_days: [1, 5]}\n"
    )
    monkeypatch.setattr(fs, "MARIN_CLUSTER_CONFIG_DIRS", (str(cluster_dir),))
    reset_data_config_cache()

    config = load_cluster_config("synth")
    # Fields present in the YAML are parsed through:
    assert config.scheme == "s3"
    assert config.region_buckets == {"na": "synth-na"}
    assert config.root == "s3://synth-na/data"
    assert config.ttl_days == (1, 5)
    # Fields absent from the YAML fall back to the DataConfig field defaults:
    assert config.temp_path == "tmp"


def test_use_data_config_overrides_then_restores():
    """use_data_config binds a config for the block, then restores the previous one."""
    override = DataConfig(region_buckets={}, scheme="s3", root="s3://other/data")
    default = load_cluster_config("marin")
    with use_data_config(override):
        assert data_config() is override
    assert data_config() is default


# --- prefix resolution (behavior, decoupled from any committed config) ------


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
    """A detected metadata region selects its region-local bucket."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: "us-east5")
    config = DataConfig(region_buckets={"us-east5": "marin-us-east5"})
    assert config.resolved_root() == "gs://marin-us-east5"


def test_resolved_root_constructs_default_for_unmapped_region(monkeypatch):
    """A detected-but-unmapped region constructs gs://marin-{region}."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: "antarctica-south1")
    config = DataConfig(region_buckets={"us-east5": "marin-us-east5"})
    assert config.resolved_root() == "gs://marin-antarctica-south1"


def test_resolved_root_local_fallback(monkeypatch):
    """With no env and no detectable region, falls back to /tmp/marin."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(fs, "region_from_metadata", lambda: None)
    assert DataConfig(region_buckets={}).resolved_root() == "/tmp/marin"
