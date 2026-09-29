# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a LocalClient and, when ``ray`` is installed, a session-wide RayClient.

The ``client`` fixture runs backend-agnostic tests against both backends.
Set ``FRAY_TEST_RAY_ADDRESS`` to point the Ray lane at a running cluster
(for example a docker compose head node) instead of an in-process one.
"""

import os

import pytest
from fray.local_backend import LocalClient

RAY_ADDRESS_ENV = "FRAY_TEST_RAY_ADDRESS"

# Ray reads this at import time; under `uv run` the default hook would start workers
# without ray installed. Test modules import ray only after this module is loaded.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


@pytest.fixture
def local_client():
    client = LocalClient(max_threads=4)
    yield client
    client.shutdown(wait=True)


@pytest.fixture(scope="session")
def ray_client():
    pytest.importorskip("ray")
    from fray.ray_backend import RayClient  # noqa: PLC0415

    address = os.environ.get(RAY_ADDRESS_ENV)
    if address is None:
        client = RayClient.connect(address="local", num_cpus=4, object_store_memory=200_000_000)
    else:
        client = RayClient.connect(address=address)
    yield client
    client.shutdown(wait=False)


@pytest.fixture(params=["local", "ray"])
def client(request):
    if request.param == "local":
        yield request.getfixturevalue("local_client")
        return

    ray = pytest.importorskip("ray")
    client = request.getfixturevalue("ray_client")
    # Test-module classes are not importable inside Ray workers; ship them by value.
    ray.cloudpickle.register_pickle_by_value(request.module)
    yield client
    # Actor names repeat across tests; a leftover would collide on the next create.
    client.kill_actors()
