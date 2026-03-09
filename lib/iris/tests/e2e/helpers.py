# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job callables for e2e tests, organized as static methods on TestJobs.

All methods use logging as the primary communication channel. They are
serialized via cloudpickle (Entrypoint.from_callable) so static methods
work fine — cloudpickle pickles bytecode, not import paths.
"""


class TestJobs:
    """Namespace for job callables that run inside Iris workers.

    Each method represents a distinct job behavior. Use logging for output
    so tests can verify both substring filtering and structured log levels.
    """

    @staticmethod
    def quick():
        return 1

    @staticmethod
    def sleep(duration: float):
        import time

        time.sleep(duration)
        return 1

    @staticmethod
    def block(s):
        """Block until sentinel is signalled."""
        s.wait()

    @staticmethod
    def fail():
        raise ValueError("intentional failure")

    @staticmethod
    def noop():
        return "ok"

    @staticmethod
    def busy_loop(duration: float = 3.0):
        """CPU-bound busy loop for profiling tests."""
        import time

        end = time.monotonic() + duration
        while time.monotonic() < end:
            sum(range(1000))

    @staticmethod
    def log_verbose(num_lines: int = 200):
        """Emit log lines at INFO/WARNING/ERROR levels.

        Covers both substring filtering (grep for "processing" vs "validation failed")
        and structured level filtering (INFO vs WARNING vs ERROR). Also emits
        named markers (info-marker, warning-marker, error-marker) for precise
        level assertions.
        """
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

    @staticmethod
    def register_endpoint(prefix):
        """Register an endpoint via RPC and verify it's listed."""
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
        finally:
            client.close()

    @staticmethod
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

    @staticmethod
    def validate_job_context():
        """Validate job context via get_job_info() in a coscheduled job."""
        import logging

        from iris.cluster.client import get_job_info

        logger = logging.getLogger("iris.test.context")
        info = get_job_info()
        if info is None:
            raise RuntimeError("Not running in an Iris job context")
        logger.info(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
        return f"Task {info.task_index} done"

    @staticmethod
    def wait_for_sentinel(s):
        """Wait on a sentinel with a short timeout, used for concurrency tests."""
        from iris.time_utils import Duration

        s.wait(timeout=Duration.from_seconds(2))
        return "done"
