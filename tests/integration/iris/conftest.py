# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Iris integration tests.

Requires --controller-url to connect to an existing controller.
"""

import logging
from pathlib import Path

import pytest
from iris.client.client import IrisClient
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.rpc.logging_connect import LogServiceClientSync

from .cluster import IrisIntegrationCluster

logger = logging.getLogger(__name__)

MARIN_ROOT = Path(__file__).resolve().parents[3]
IRIS_ROOT = MARIN_ROOT / "lib" / "iris"
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"

# Module-scoped fixtures (integration_cluster) may need extra time for worker
# registration.  Apply this timeout to the first test in every module so the
# fixture setup isn't bounded by the default per-test timeout.
_FIXTURE_SETUP_TIMEOUT = 600


def pytest_addoption(parser):
    parser.addoption("--controller-url", default=None, help="Iris controller URL (required)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give the first test in each module extra timeout for fixture setup."""
    seen_modules: set[str] = set()
    for item in items:
        module = item.module.__name__ if item.module else ""
        if module not in seen_modules:
            seen_modules.add(module)
            existing = item.get_closest_marker("timeout")
            effective = existing.args[0] if existing and existing.args else 0
            if effective < _FIXTURE_SETUP_TIMEOUT:
                item.add_marker(pytest.mark.timeout(_FIXTURE_SETUP_TIMEOUT))


@pytest.fixture(scope="module")
def integration_cluster(request):
    """Connect to an existing Iris controller."""
    url = request.config.getoption("--controller-url")
    if not url:
        pytest.skip("--controller-url not provided")
    client = IrisClient.remote(url, workspace=MARIN_ROOT)
    controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
    log_client = LogServiceClientSync(address=url, timeout_ms=30000)
    tc = IrisIntegrationCluster(
        url=url,
        client=client,
        controller_client=controller_client,
        log_client=log_client,
        job_timeout=120.0,
    )
    yield tc
    log_client.close()
    controller_client.close()
