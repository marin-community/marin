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

from .cluster import IrisIntegrationCluster

logger = logging.getLogger(__name__)

IRIS_ROOT = Path(__file__).resolve().parents[3] / "lib" / "iris"
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"


def pytest_addoption(parser):
    parser.addoption("--controller-url", default=None, help="Iris controller URL (required)")


@pytest.fixture(scope="module")
def integration_cluster(request):
    """Connect to an existing Iris controller."""
    url = request.config.getoption("--controller-url")
    if not url:
        pytest.skip("--controller-url not provided")
    client = IrisClient.remote(url, workspace=IRIS_ROOT)
    controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
    tc = IrisIntegrationCluster(
        url=url,
        client=client,
        controller_client=controller_client,
        job_timeout=120.0,
    )
    yield tc
    controller_client.close()
