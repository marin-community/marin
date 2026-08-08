# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""config/external/migration.toml must stay consistent with the forks it describes.

The descriptor is read by the refresh-fork skill and the weekly coordinator, so a section that
names the wrong repository, a missing e2e, or a fork with no descriptor would silently misroute a
migration. These checks pin the descriptor to the real fork layout: the isolated uv projects on
disk, the vllm/tpu-forks.toml sections, and the pin sources each section claims to track.
"""

import tomllib
from pathlib import Path

import pytest
from rigging.config_discovery import find_project_root

_VALID_KINDS = {"patch_free", "overlay"}
_VALID_BASE_SELECT = {"upstream_main", "fork_main", "latest_release", "derived"}


def _root() -> Path:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to validate")
    return root


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _external_root(root: Path) -> Path:
    return root / "config" / "external"


def _isolated_projects(root: Path) -> set[str]:
    external = _external_root(root)
    return {child.name for child in external.iterdir() if (child / "pyproject.toml").exists()}


def _tpu_fork_sections(root: Path) -> set[str]:
    return set(_load(_external_root(root) / "vllm" / "tpu-forks.toml"))


def _descriptors(root: Path) -> dict:
    return _load(_external_root(root) / "migration.toml")


def _isolated_repository(root: Path, name: str) -> str:
    sources = _load(_external_root(root) / name / "pyproject.toml")["tool"]["uv"]["sources"]
    git_sources = [spec["git"] for spec in sources.values() if isinstance(spec, dict) and "git" in spec]
    assert len(git_sources) == 1, f"{name}: expected one git source, found {git_sources}"
    return git_sources[0]


def test_every_fork_has_exactly_one_descriptor():
    root = _root()
    expected = _isolated_projects(root) | _tpu_fork_sections(root)
    assert set(_descriptors(root)) == expected


def test_repository_matches_the_pin_source():
    root = _root()
    for name, section in _descriptors(root).items():
        pin = section["pin"]
        if pin == "isolated_project":
            expected = _isolated_repository(root, name)
        else:
            rel, _, fragment = pin.removeprefix("descriptor:").partition("#")
            expected = _load(_external_root(root) / rel)[fragment]["repository"]
        assert section["repository"] == expected, f"{name}: repository disagrees with {pin}"


def test_fields_are_well_formed():
    root = _root()
    descriptors = _descriptors(root)
    for name, section in descriptors.items():
        assert section["kind"] in _VALID_KINDS, f"{name}: bad kind"
        assert section["base_select"] in _VALID_BASE_SELECT, f"{name}: bad base_select"
        if section["base_select"] == "derived":
            fork, _, path = section["derived_from"].partition(":")
            assert fork in descriptors and path, f"{name}: derived_from must name a fork and path"
        # An overlay is rebased onto its upstream, so it must name one; patch_free may be marin-native.
        if section["kind"] == "overlay":
            assert section.get("upstream"), f"{name}: overlay needs an upstream"


def test_depends_on_is_acyclic_and_resolvable():
    root = _root()
    descriptors = _descriptors(root)
    graph = {name: section.get("depends_on", []) for name, section in descriptors.items()}
    for name, deps in graph.items():
        for dep in deps:
            assert dep in descriptors, f"{name}: depends_on unknown fork {dep!r}"

    resolved: set[str] = set()

    def visit(node: str, stack: tuple[str, ...]) -> None:
        assert node not in stack, f"dependency cycle through {node}"
        if node in resolved:
            return
        for dep in graph[node]:
            visit(dep, (*stack, node))
        resolved.add(node)

    for name in graph:
        visit(name, ())


def test_e2e_targets_exist():
    root = _root()
    for name, section in _descriptors(root).items():
        target = section["e2e"].split("::", 1)[0]
        assert (root / target).exists(), f"{name}: e2e target {target} is missing"
