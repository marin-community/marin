# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job callables for integration tests, serialized via cloudpickle.

All functions use logging as the primary communication channel. They are
serialized via cloudpickle (Entrypoint.from_callable).
"""


def quick():
    return 1


def sleep(duration: float):
    import time

    time.sleep(duration)
    return 1


def fail():
    raise ValueError("intentional failure")


def noop():
    return "ok"


def busy_loop(duration: float = 3.0):
    """CPU-bound busy loop for profiling tests."""
    import time

    end = time.monotonic() + duration
    while time.monotonic() < end:
        sum(range(1000))


def log_verbose(num_lines: int = 200):
    """Emit log lines at INFO/WARNING/ERROR levels with markers."""
    import logging

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
    """Register an endpoint via RPC and verify it's listed."""
    from iris.cluster.client import get_job_info
    from iris.rpc import controller_pb2
    from iris.rpc.controller_connect import ControllerServiceClientSync

    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")

    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        endpoint_name = f"{prefix}/actor1"
        request = controller_pb2.Controller.RegisterEndpointRequest(
            name=endpoint_name,
            address="localhost:5000",
            task_id=info.task_id.to_wire(),
            metadata={"type": "actor"},
        )
        response = client.register_endpoint(request)
        assert response.endpoint_id

        list_request = controller_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
        list_response = client.list_endpoints(list_request)
        assert len(list_response.endpoints) == 1
        names = [ep.name for ep in list_response.endpoints]
        assert endpoint_name in names
    finally:
        client.close()


def validate_ports():
    """Validate that requested ports are allocated via JobInfo."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if "http" not in info.ports or "grpc" not in info.ports:
        raise ValueError(f"Ports not set: {info.ports}")
    assert info.ports["http"] > 0
    assert info.ports["grpc"] > 0
