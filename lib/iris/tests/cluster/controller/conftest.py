# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for controller tests."""

import pytest
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.testing.controller import (
    controller_factory,
    controller_process_factory,
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
    mock_controller.provider.health = state._health
    return make_controller_service(
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(EndpointRegistry(db=state._db)),
    )


@pytest.fixture
def make_controller(tmp_path):
    yield from controller_factory(tmp_path)


@pytest.fixture
def make_controller_process(tmp_path):
    yield from controller_process_factory(tmp_path)


@pytest.fixture
def harness(state):
    return make_controller_test_harness(state)
