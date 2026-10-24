# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for controller tests."""

import pytest
from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.testing.controller import (
    controller_factory,
    make_controller_service,
    make_controller_state,
    make_controller_test_harness,
    make_mock_controller,
)
from iris.testing.k8s import k8s_backend_descriptor, pod_config


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


@pytest.fixture
def coreweave_service(controller_service, mock_controller):
    backend = K8sTaskProvider(
        descriptor=k8s_backend_descriptor(),
        kubectl=InMemoryK8sService(namespace="iris"),
        pods=pod_config(require_coreweave_credentials=True),
    )
    mock_controller.backend = backend
    try:
        yield controller_service
    finally:
        backend.close()
