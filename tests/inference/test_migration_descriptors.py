# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""config/external/migration.toml must stay consistent with the forks it describes.

The descriptor is read by the refresh-fork skill, so a section for a nonexistent fork, a fork with
no section, a pin pointing at the wrong source, a missing e2e, or a malformed field would silently
misroute a migration. These checks pin the descriptor to the real fork layout: the isolated uv
projects on disk and the vllm/tpu-forks.toml sections.
"""

import tomllib
from pathlib import Path

import pytest
from rigging.config_discovery import find_project_root

_VALID_BASE_SELECT = {"upstream_main", "latest_release", "derived", "fork_main"}


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


def test_every_fork_has_exactly_one_descriptor():
    root = _root()
    expected = _isolated_projects(root) | _tpu_fork_sections(root)
    assert set(_descriptors(root)) == expected


def test_pin_resolves_to_the_section_fork():
    root = _root()
    for name, section in _descriptors(root).items():
        pin = section["pin"]
        if pin == "isolated_project":
            assert (_external_root(root) / name / "pyproject.toml").exists(), f"{name}: no isolated project"
            continue
        rel, _, fragment = pin.removeprefix("descriptor:").partition("#")
        assert fragment == name, f"{name}: pin fragment {fragment!r} does not match the section name"
        assert fragment in _load(_external_root(root) / rel), f"{name}: {rel} has no [{fragment}] section"


def test_fields_are_well_formed():
    root = _root()
    descriptors = _descriptors(root)
    for name, section in descriptors.items():
        assert section["base_select"] in _VALID_BASE_SELECT, f"{name}: bad base_select"
        if section["base_select"] == "derived":
            fork, _, path = section["derived_from"].partition(":")
            assert fork in descriptors and path, f"{name}: derived_from must name a fork and path"
        # We rebase every fork onto its upstream; only a marin-native fork (fork_main) omits one.
        if section["base_select"] == "fork_main":
            assert not section.get("upstream"), f"{name}: fork_main is marin-native, must not name an upstream"
        else:
            assert section.get("upstream"), f"{name}: rebasing onto upstream requires an upstream"


def test_derived_forks_share_a_group():
    # A fork whose base derives from another must land in the same atomic group, so a split refresh
    # cannot pin it against a different revision of its source than the one that landed.
    root = _root()
    descriptors = _descriptors(root)
    for name, section in descriptors.items():
        derived_from = section.get("derived_from")
        if not derived_from:
            continue
        source = derived_from.partition(":")[0]
        group = section.get("group")
        assert (
            group and descriptors[source].get("group") == group
        ), f"{name} derives from {source}; both must share a group so they refresh atomically"


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
