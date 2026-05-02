# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.config_discovery import (
    find_configs,
    find_project_root,
    list_cluster_configs,
    resolve_cluster_config,
)

# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------

_WORKSPACE_PYPROJECT = """\
[project]
name = "sample-root"
version = "0.0.0"

[tool.uv.workspace]
members = ["lib/a"]
"""

_PLAIN_PYPROJECT = """\
[project]
name = "sample-member"
version = "0.0.0"
"""


def test_find_project_root_from_repo():
    """Running from within the marin checkout finds the workspace root."""
    find_project_root.cache_clear()

    root = find_project_root()
    assert root is not None
    # The marin root must have [tool.uv.workspace] in its pyproject.toml.
    pyproject = root / "pyproject.toml"
    assert pyproject.is_file()
    assert "[tool.uv.workspace]" in pyproject.read_text()


def test_find_project_root_skips_non_workspace_pyproject(tmp_path):
    """A bare pyproject.toml (no ``[tool.uv.workspace]``) is not treated as the marin root."""
    find_project_root.cache_clear()

    # Build a nested layout where the inner dir has a plain pyproject.toml and
    # the outer dir has a workspace-declaring pyproject.toml.
    outer = tmp_path / "outer"
    inner = outer / "lib" / "member"
    inner.mkdir(parents=True)
    (outer / "pyproject.toml").write_text(_WORKSPACE_PYPROJECT)
    (inner / "pyproject.toml").write_text(_PLAIN_PYPROJECT)

    result = find_project_root(start=inner)
    assert result == outer.resolve()


def test_find_project_root_returns_none(tmp_path):
    """Returns None when no workspace-root pyproject is found up the hierarchy."""
    find_project_root.cache_clear()

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)

    for parent in (deep, *deep.parents):
        pp = parent / "pyproject.toml"
        if pp.is_file() and "[tool.uv.workspace]" in pp.read_text(errors="ignore"):
            pytest.skip("tmp hierarchy contains a workspace pyproject — cannot test None return")

    assert find_project_root(start=deep) is None


# ---------------------------------------------------------------------------
# resolve_cluster_config
# ---------------------------------------------------------------------------


def test_resolve_cluster_config_explicit_path(tmp_path):
    """Passing an existing file path returns it directly."""
    config = tmp_path / "my-cluster.yaml"
    config.write_text("cluster: my-cluster\n")

    result = resolve_cluster_config(str(config), dirs=[])
    assert result == config


def test_resolve_cluster_config_by_name(tmp_path):
    """Resolving a name finds the file in the provided dirs."""
    config = tmp_path / "foo.yaml"
    config.write_text("cluster: foo\n")

    result = resolve_cluster_config("foo", dirs=[tmp_path])
    assert result == config


def test_resolve_cluster_config_not_found(tmp_path):
    """Raises FileNotFoundError with a helpful message when not found."""
    with pytest.raises(FileNotFoundError, match="no-such-cluster"):
        resolve_cluster_config("no-such-cluster", dirs=[tmp_path])


def test_resolve_adds_yaml_extension(tmp_path):
    """Resolving 'foo' finds 'foo.yaml' in the search directory."""
    config = tmp_path / "foo.yaml"
    config.write_text("cluster: foo\n")

    result = resolve_cluster_config("foo", dirs=[tmp_path])
    assert result == config
    assert result.suffix == ".yaml"


def test_resolve_adds_yml_extension(tmp_path):
    """Resolving 'bar' finds 'bar.yml' in the search directory."""
    config = tmp_path / "bar.yml"
    config.write_text("cluster: bar\n")

    result = resolve_cluster_config("bar", dirs=[tmp_path])
    assert result == config
    assert result.suffix == ".yml"


def test_resolve_name_with_yaml_suffix(tmp_path):
    """A name already containing ``.yaml`` still resolves via directory search."""
    config = tmp_path / "baz.yaml"
    config.write_text("cluster: baz\n")

    result = resolve_cluster_config("baz.yaml", dirs=[tmp_path])
    assert result == config


# ---------------------------------------------------------------------------
# list_cluster_configs / find_configs
# ---------------------------------------------------------------------------


def test_list_cluster_configs(tmp_path):
    """Lists all YAML files in the provided search directory."""
    (tmp_path / "alpha.yaml").write_text("cluster: alpha\n")
    (tmp_path / "beta.yml").write_text("cluster: beta\n")
    (tmp_path / "not-a-config.txt").write_text("ignored\n")

    configs = list_cluster_configs(dirs=[tmp_path])

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

    configs = list_cluster_configs(dirs=[high_prio, low_prio])

    assert configs["mycluster"] == high_prio / "mycluster.yaml"


def test_find_configs_relative_dir_resolved_against_project_root(tmp_path, monkeypatch):
    """Relative dirs resolve against the marin workspace root."""
    find_project_root.cache_clear()

    # Build a fake marin workspace root with a nested examples dir.
    root = tmp_path / "fake-root"
    examples = root / "lib" / "fake" / "examples"
    examples.mkdir(parents=True)
    (root / "pyproject.toml").write_text(_WORKSPACE_PYPROJECT)
    (examples / "cluster-a.yaml").write_text("cluster: a\n")

    monkeypatch.chdir(root)
    find_project_root.cache_clear()

    configs = find_configs(dirs=["lib/fake/examples"])
    assert "cluster-a" in configs
    assert configs["cluster-a"] == examples / "cluster-a.yaml"


def test_find_configs_empty_string_means_project_root(tmp_path, monkeypatch):
    """An empty-string dir searches the project root itself."""
    find_project_root.cache_clear()

    root = tmp_path / "fake-root"
    root.mkdir()
    (root / "pyproject.toml").write_text(_WORKSPACE_PYPROJECT)
    (root / "top.yaml").write_text("cluster: top\n")

    monkeypatch.chdir(root)
    find_project_root.cache_clear()

    configs = find_configs(dirs=[""])
    assert "top" in configs
    assert configs["top"] == root / "top.yaml"
