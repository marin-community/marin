# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for worker tests."""

import pytest
from iris.testing.worker import make_docker_runtime, make_mock_bundle_store, make_mock_runtime, make_mock_worker


@pytest.fixture
def docker_runtime(tmp_path):
    runtime = make_docker_runtime(tmp_path)
    yield runtime
    runtime.cleanup()


@pytest.fixture
def mock_bundle_store():
    return make_mock_bundle_store()


@pytest.fixture
def mock_runtime():
    return make_mock_runtime()


@pytest.fixture
def mock_worker(mock_bundle_store, mock_runtime, tmp_path):
    return make_mock_worker(mock_bundle_store, mock_runtime, tmp_path)
