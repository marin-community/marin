# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from rigging.config_discovery import find_project_root, list_cluster_configs, resolve_cluster_config

# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------


def test_find_project_root_from_repo():
    """Running from within the repo finds the root (which contains .git)."""
    # Clear the lru_cache so prior test runs don't interfere.
    find_project_root.cache_clear()

    root = find_project_root()
    assert root is not None
    assert (root / ".git").exists() or (root / "pyproject.toml").exists()


def test_find_project_root_returns_none(tmp_path):
    """Returns None when no .git or pyproject.toml exists in the hierarchy."""
    find_project_root.cache_clear()

    # Use a subdirectory of tmp_path that has no markers.
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)

    result = find_project_root(start=deep)
    # tmp_path itself (and all parents up to /) should not contain .git or
    # pyproject.toml under a normal /tmp mount.  If the host happens to have
    # one, this test would need adjustment — but in practice /tmp is clean.
    # We only assert None when the temp hierarchy is truly marker-free.
    for parent in (deep, *deep.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            pytest.skip("tmp hierarchy contains a project marker — cannot test None return")

    assert result is None


# ---------------------------------------------------------------------------
# resolve_cluster_config
# ---------------------------------------------------------------------------


def test_resolve_cluster_config_explicit_path(tmp_path):
    """Passing an existing file path returns it directly."""
    config = tmp_path / "my-cluster.yaml"
    config.write_text("cluster: my-cluster\n")

    result = resolve_cluster_config(str(config))
    assert result == config


def test_resolve_cluster_config_by_name(tmp_path):
    """Resolving a name finds the file in the provided search_path."""
    config = tmp_path / "foo.yaml"
    config.write_text("cluster: foo\n")

    result = resolve_cluster_config("foo", search_paths=[tmp_path])
    assert result == config


def test_resolve_cluster_config_not_found(tmp_path):
    """Raises FileNotFoundError with a helpful message when not found."""
    with pytest.raises(FileNotFoundError, match="no-such-cluster"):
        resolve_cluster_config("no-such-cluster", search_paths=[tmp_path])


def test_resolve_adds_yaml_extension(tmp_path):
    """Resolving 'foo' finds 'foo.yaml' in the search directory."""
    config = tmp_path / "foo.yaml"
    config.write_text("cluster: foo\n")

    result = resolve_cluster_config("foo", search_paths=[tmp_path])
    assert result == config
    assert result.suffix == ".yaml"


def test_resolve_adds_yml_extension(tmp_path):
    """Resolving 'bar' finds 'bar.yml' in the search directory."""
    config = tmp_path / "bar.yml"
    config.write_text("cluster: bar\n")

    result = resolve_cluster_config("bar", search_paths=[tmp_path])
    assert result == config
    assert result.suffix == ".yml"


# ---------------------------------------------------------------------------
# list_cluster_configs
# ---------------------------------------------------------------------------


def test_list_cluster_configs(tmp_path):
    """Lists all YAML files in the provided search directory."""
    (tmp_path / "alpha.yaml").write_text("cluster: alpha\n")
    (tmp_path / "beta.yml").write_text("cluster: beta\n")
    (tmp_path / "not-a-config.txt").write_text("ignored\n")

    configs = list_cluster_configs(search_paths=[tmp_path])

    assert "alpha" in configs
    assert "beta" in configs
    assert "not-a-config" not in configs
    assert configs["alpha"] == tmp_path / "alpha.yaml"
    assert configs["beta"] == tmp_path / "beta.yml"


def test_list_cluster_configs_first_found_wins(tmp_path):
    """Higher-priority search dirs take precedence on duplicate names."""
    high_prio = tmp_path / "high"
    low_prio = tmp_path / "low"
    high_prio.mkdir()
    low_prio.mkdir()

    (high_prio / "mycluster.yaml").write_text("priority: high\n")
    (low_prio / "mycluster.yaml").write_text("priority: low\n")

    configs = list_cluster_configs(search_paths=[high_prio, low_prio])

    assert configs["mycluster"] == high_prio / "mycluster.yaml"
