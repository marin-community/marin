# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job callables for integration tests, serialized via cloudpickle.

All functions use logging as the primary communication channel. They are
serialized via cloudpickle (Entrypoint.from_callable).
"""
import logging
import time

from iris.client import iris_ctx
from iris.cluster.resources.endpoint import EndpointQuery


def quick():
    return 1


def sleep(duration: float):
    time.sleep(duration)
    return 1


def fail():
    raise ValueError("intentional failure")


def noop():
    return "ok"


def busy_loop(duration: float = 3.0):
    """CPU-bound busy loop for profiling tests."""

    end = time.monotonic() + duration
    while time.monotonic() < end:
        sum(range(1000))


def log_verbose(num_lines: int = 200):
    """Emit log lines at INFO/WARNING/ERROR levels with markers."""

    logger = logging.getLogger("iris.test.verbose")
    for i in range(num_lines):
        if i % 3 == 0:
            logger.info(f"step {i}: processing data batch")
        elif i % 3 == 1:
            logger.warning(f"step {i}: slow operation detected")
        else:
            logger.error(f"step {i}: validation failed for item")
    logger.info("info-marker")
    logger.warning("warning-marker")
    logger.error("error-marker")
    logger.info("DONE: all lines emitted")
    return 1


def register_endpoint(prefix):
    """Register an endpoint and verify it through the public resource inventory."""

    ctx = iris_ctx()
    if ctx.client is None:
        raise ValueError("Iris client not available")

    endpoint_name = f"{prefix}/actor1"
    endpoint_id = ctx.registry.register(endpoint_name, "localhost:5000", {"type": "actor"})
    try:
        listed = ctx.client.list_endpoints(
            EndpointQuery(name_prefix=f"{ctx.namespace}/{endpoint_name}", page_size=100)
        ).items
        matches = [endpoint for endpoint in listed if endpoint.endpoint_id == endpoint_id]
        assert len(matches) == 1
        assert matches[0].name.endswith(f"/{endpoint_name}")
    finally:
        ctx.registry.unregister(endpoint_id)
