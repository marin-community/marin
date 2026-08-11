# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from lib.iris.tests.journeys.world import JourneyWorld, journey_world


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


@pytest.fixture
def multi_backend_journey(tmp_path, monkeypatch):
    world = JourneyWorld(
        tmp_path,
        monkeypatch,
        backend_advertisements={
            "east": {"region": {"us-east1"}},
            "west": {"region": {"us-west1"}},
        },
    )
    try:
        yield world
    finally:
        world.close()


@pytest.fixture
def mixed_capacity_journey(tmp_path, monkeypatch):
    world = JourneyWorld(
        tmp_path,
        monkeypatch,
        backend_advertisements={
            "blocked": {"region": {"blocked"}},
            "ready": {"region": {"ready"}},
        },
        unavailable_backend_ids={"blocked"},
    )
    try:
        yield world
    finally:
        world.close()
