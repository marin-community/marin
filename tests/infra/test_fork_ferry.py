# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""infra/cron/fork_ferry.py must stay consistent with the forks it drives.

The coordinator derives units from the descriptor's ``group`` field, so a fork that no unit drives
and that is not explicitly excluded would silently skip a migration. These checks pin the derived
units and the exclusion set to the real descriptor.
"""

import tomllib

import pytest
from rigging.config_discovery import find_project_root

from infra.cron.fork_ferry import DESCRIPTOR_PATH, EXCLUDED, drivable_units


def _descriptor() -> dict:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to validate")
    return tomllib.loads((root / DESCRIPTOR_PATH).read_text())


def test_grouped_forks_refresh_as_one_unit():
    # tpu-inference and the TPU vllm pin share a base, so the descriptor's `group` must put them in
    # one unit; refreshing them separately could pin them against different revisions of that base.
    owner = {fork: unit for unit, forks in drivable_units(_descriptor()).items() for fork in forks}
    assert owner["tpu-inference"] == owner["vllm"]


def test_units_and_exclusions_cover_every_fork():
    # A fork added to the descriptor must fall into a driven unit or the exclusion set, so the ferry
    # cannot silently skip a new fork.
    descriptor = _descriptor()
    driven = {fork for forks in drivable_units(descriptor).values() for fork in forks}
    assert driven.isdisjoint(EXCLUDED), "a fork is both driven and excluded"
    assert driven | EXCLUDED == set(descriptor)
