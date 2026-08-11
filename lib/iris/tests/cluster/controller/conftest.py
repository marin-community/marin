# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for controller tests."""

import pytest
from iris.testing.controller import (
    controller_factory,
    make_controller_service,
    make_controller_state,
    make_controller_test_harness,
    make_mock_controller,
)


@pytest.fixture
def state():
    with make_controller_state() as controller_state:
        yield controller_state


@pytest.fixture
def mock_controller():
    return make_mock_controller()


@pytest.fixture
def controller_service(state, log_client, mock_controller, tmp_path):
    return make_controller_service(state, log_client, mock_controller, tmp_path)


@pytest.fixture
def make_controller(tmp_path):
    yield from controller_factory(tmp_path)


@pytest.fixture
def harness(state):
    return make_controller_test_harness(state)
