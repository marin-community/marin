# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and hooks for Iris E2E tests."""

import pytest
from iris.chaos import reset_chaos
from iris.testing.e2e import (
    detect_fd_leaks,
    ensure_dashboard_built,
    local_multi_worker_test_cluster,
    local_test_cluster,
)

_LOCAL_FIXTURE_TIMEOUT = 120
_LOCAL_E2E_TIMEOUT = 30
_CLOUD_FIXTURE_TIMEOUT = 1200
_CLOUD_TEST_TIMEOUT = 120


def pytest_addoption(parser):
    parser.addoption("--iris-controller-url", default=None, help="Connect to existing controller")


def pytest_collection_modifyitems(config, items):
    """Set local and cloud timeouts for E2E tests."""
    is_cloud = config.getoption("--iris-controller-url") is not None
    first_smoke_test = True
    for item in items:
        if item.get_closest_marker("timeout"):
            continue
        uses_smoke = "smoke_cluster" in getattr(item, "fixturenames", ())
        if is_cloud:
            timeout = _CLOUD_FIXTURE_TIMEOUT if uses_smoke and first_smoke_test else _CLOUD_TEST_TIMEOUT
        else:
            timeout = _LOCAL_FIXTURE_TIMEOUT if uses_smoke and first_smoke_test else _LOCAL_E2E_TIMEOUT
        item.add_marker(pytest.mark.timeout(timeout))
        if uses_smoke and first_smoke_test:
            first_smoke_test = False


@pytest.fixture(scope="session", autouse=True)
def _ensure_dashboard_built(tmp_path_factory):
    ensure_dashboard_built(tmp_path_factory)


@pytest.fixture
def cluster():
    yield from local_test_cluster()


@pytest.fixture
def multi_worker_cluster():
    yield from local_multi_worker_test_cluster()


@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()


@pytest.fixture(autouse=True)
def _detect_fd_leaks(request):
    yield from detect_fd_leaks(request)
