# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared job helper functions used across e2e test files.

Functions prefixed with _ are job callables that run inside workers.
They must be importable as top-level functions (no closures) so that
Entrypoint.from_callable can serialize them.
"""

import time

# ---------------------------------------------------------------------------
# Core primitives (used by both smoke and chaos tests)
# ---------------------------------------------------------------------------


def _quick():
    return 1


def _slow():
    time.sleep(120)


def _block(s):
    """Block until sentinel is signalled. Pass a SentinelFile instance."""
    s.wait()


def _failing():
    raise ValueError("intentional failure")


def _noop():
    return "ok"


# ---------------------------------------------------------------------------
# Smoke test job functions (migrated from scripts/smoke-test.py)
# ---------------------------------------------------------------------------


def _hello_job():
    """Simple job that prints and returns."""
    print("Hello from smoke test!")
    return 42


def _quick_task_job(task_id: int):
    """Quick job that sleeps briefly and returns."""
    time.sleep(2.0)
    print(f"Task {task_id} completed")
    return task_id


def _distributed_work_job():
    """Coscheduled job that validates job context via get_job_info()."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise RuntimeError("Not running in an Iris job context")
    print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
    return f"Task {info.task_index} done"


def _reservation_child():
    """Child job that runs on a reserved worker."""
    print("Child running on reserved worker")
    return 1


def _reservation_parent_job(num_children: int, device_variant: str):
    """Parent job that spawns children consuming the reservation."""
    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, ResourceSpec, tpu_device

    ctx = iris_ctx()
    children = []
    for i in range(num_children):
        child = ctx.client.submit(
            entrypoint=Entrypoint.from_callable(_reservation_child),
            name=f"reserved-child-{i}",
            resources=ResourceSpec(device=tpu_device(device_variant)),
        )
        children.append(child)

    for child in children:
        child.wait(timeout=120, raise_on_failure=True)
    print(f"All {num_children} children completed on reserved workers")


# ---------------------------------------------------------------------------
# Dashboard / log test job functions
# ---------------------------------------------------------------------------


def _verbose_task():
    """Emit 200 numbered log lines with categorized prefixes for filter testing."""
    for i in range(200):
        if i % 3 == 0:
            print(f"[INFO] step {i}: processing data batch")
        elif i % 3 == 1:
            print(f"[WARN] step {i}: slow operation detected")
        else:
            print(f"[ERROR] step {i}: validation failed for item")
    print("DONE: all 200 lines emitted")
    return 1


def _emit_multi_level_logs():
    """Callable that emits log lines at multiple levels using the unified format."""
    import logging
    import sys

    _LEVEL_PREFIX = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Fmt(logging.Formatter):
        def format(self, record):
            record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
            return super().format(record)

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Fmt(fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s", datefmt="%Y%m%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    log = logging.getLogger("test.levels")
    log.debug("debug-marker")
    log.info("info-marker")
    log.warning("warning-marker")
    log.error("error-marker")


# ---------------------------------------------------------------------------
# Endpoint / port test job functions (migrated from test_endpoints.py)
# ---------------------------------------------------------------------------


def _register_endpoint_job(prefix):
    """Runs inside the worker. Registers an endpoint and verifies it via RPC."""
    from iris.cluster.client import get_job_info
    from iris.rpc import cluster_pb2
    from iris.rpc.cluster_connect import ControllerServiceClientSync

    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")

    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        endpoint_name = f"{prefix}/actor1"
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=endpoint_name,
            address="localhost:5000",
            task_id=info.task_id.to_wire(),
            metadata={"type": "actor"},
        )
        response = client.register_endpoint(request)
        assert response.endpoint_id

        list_request = cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
        list_response = client.list_endpoints(list_request)
        assert len(list_response.endpoints) == 1
        names = [ep.name for ep in list_response.endpoints]
        assert endpoint_name in names
        time.sleep(0.5)
    finally:
        client.close()


def _port_job():
    """Runs inside the worker. Validates that requested ports are allocated."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if "http" not in info.ports or "grpc" not in info.ports:
        raise ValueError(f"Ports not set: {info.ports}")
    assert info.ports["http"] > 0
    assert info.ports["grpc"] > 0
