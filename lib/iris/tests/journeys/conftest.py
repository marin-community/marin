# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.testing.journeys.world import JourneyWorld, journey_world


@pytest.fixture
def journey(tmp_path, monkeypatch):
    with journey_world(tmp_path, monkeypatch) as world:
        yield world


@pytest.fixture
def journey_without_capacity(tmp_path, monkeypatch):
    world = JourneyWorld(tmp_path, monkeypatch, capacity_available=False)
    try:
        yield world
    finally:
        world.close()


@pytest.fixture
def dry_run_journey(tmp_path, monkeypatch):
    world = JourneyWorld(tmp_path, monkeypatch, dry_run=True)
    try:
        yield world
    finally:
        world.close()
