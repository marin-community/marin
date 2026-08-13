# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""infra/cron/fork_ferry.py must stay consistent with the forks it drives.

The coordinator dispatches the refresh-fork skill per unit, so a unit naming a nonexistent fork,
or a descriptor section that no unit drives and that is not explicitly excluded, would silently
misroute or skip a migration. These checks pin the unit map to the real descriptor.
"""

import tomllib

import pytest
from rigging.config_discovery import find_project_root

from infra.cron.fork_ferry import EXCLUDED, UNITS, build_prompt


def _descriptor_sections() -> set[str]:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to validate")
    return set(tomllib.loads((root / "config" / "external" / "migration.toml").read_text()))


def test_every_unit_fork_has_a_descriptor_section():
    sections = _descriptor_sections()
    for unit, forks in UNITS.items():
        for fork in forks:
            assert fork in sections, f"{unit}: no migration.toml section for {fork}"


def test_units_and_exclusions_cover_every_fork():
    # A fork added to the descriptor must be wired into a unit or consciously excluded, so the
    # ferry cannot silently skip a new fork.
    driven = {fork for forks in UNITS.values() for fork in forks}
    assert driven.isdisjoint(EXCLUDED), "a fork is both driven and excluded"
    assert driven | EXCLUDED == _descriptor_sections()


def test_prompt_names_every_fork_in_the_unit():
    for forks in UNITS.values():
        prompt = build_prompt(forks)
        for fork in forks:
            assert fork in prompt, f"prompt drops {fork}"
