# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test to reproduce Python version mismatch issue with cloudpickle.

The smoke-test.py fails with:
  TypeError: bad argument type for built-in operation

Root cause analysis:
1. smoke-test.py runs as __main__ with Python 3.11 (via uv run)
2. Entrypoint.from_callable() calls cloudpickle.register_pickle_by_value(module)
   because module.__package__ is None for scripts run as __main__
3. This causes cloudpickle to pickle the function's __globals__ including __builtins__
4. The __builtins__ dict from Python 3.11 contains built-in functions pickled as
   specific Python 3.11 objects
5. When unpickled in the task container (Python 3.12 from Dockerfile.task),
   the built-in functions are incompatible, causing "TypeError: bad argument type
   for built-in operation" when calling print() or other builtins.

The fix should ensure the task container uses the same Python version as the client,
OR avoid pickling __builtins__ by not using register_pickle_by_value for __main__.
"""

import sys

import pytest

from tests.cluster.test_e2e import E2ECluster, unique_name


def _hello_tpu_job():
    """Simple job that prints and returns - same as smoke-test.py."""
    print("Hello from TPU!")
    return 42


def _job_with_print():
    """Job that uses print with formatting."""
    x = 42
    print(f"The answer is {x}")
    return x


def _job_with_builtin_operations():
    """Job that uses various builtins to verify they work."""
    # Test various built-in functions
    items = [1, 2, 3, 4, 5]
    total = sum(items)
    length = len(items)
    maximum = max(items)
    minimum = min(items)

    # Print uses sys.stdout which relies on builtins
    print(f"sum={total}, len={length}, max={maximum}, min={minimum}")

    return {"sum": total, "len": length, "max": maximum, "min": minimum}


class TestPickleVersionMismatch:
    """Test that callable jobs work correctly with Docker runtime.

    These tests verify the fix for the Python version mismatch issue where
    cloudpickle's register_pickle_by_value captures __builtins__ from Python 3.11
    and those built-in functions fail when unpickled in Python 3.12.
    """

    @pytest.mark.docker
    def test_simple_print_job_docker(self):
        """Reproduce the smoke-test failure with a simple print job in Docker.

        This is the minimal reproduction case: a function that only calls print().
        Prior to the fix, this fails with:
            TypeError: bad argument type for built-in operation
        """
        with E2ECluster(use_docker=True) as cluster:
            job_id = cluster.submit(_hello_tpu_job, name=unique_name("hello-tpu"))
            status = cluster.wait(job_id, timeout=120)
            if status["state"] != "JOB_STATE_SUCCEEDED":
                logs = cluster.get_task_logs(job_id)
                pytest.fail(f"Job failed: {status}\nLogs:\n" + "\n".join(logs[-100:]))
            assert status["state"] == "JOB_STATE_SUCCEEDED"

    @pytest.mark.docker
    def test_formatted_print_job_docker(self):
        """Test job with f-string print in Docker."""
        with E2ECluster(use_docker=True) as cluster:
            job_id = cluster.submit(_job_with_print, name=unique_name("print-format"))
            status = cluster.wait(job_id, timeout=120)
            if status["state"] != "JOB_STATE_SUCCEEDED":
                logs = cluster.get_task_logs(job_id)
                pytest.fail(f"Job failed: {status}\nLogs:\n" + "\n".join(logs[-100:]))
            assert status["state"] == "JOB_STATE_SUCCEEDED"

    @pytest.mark.docker
    def test_builtin_operations_docker(self):
        """Test that various built-in functions work correctly.

        If __builtins__ is corrupted by cross-version pickling, multiple
        built-in functions will fail, not just print().
        """
        with E2ECluster(use_docker=True) as cluster:
            job_id = cluster.submit(_job_with_builtin_operations, name=unique_name("builtins"))
            status = cluster.wait(job_id, timeout=120)
            if status["state"] != "JOB_STATE_SUCCEEDED":
                logs = cluster.get_task_logs(job_id)
                pytest.fail(f"Job failed: {status}\nLogs:\n" + "\n".join(logs[-100:]))
            assert status["state"] == "JOB_STATE_SUCCEEDED"


if __name__ == "__main__":
    # Run directly to debug
    print(f"Host Python version: {sys.version}")
    pytest.main([__file__, "-v", "-s", "-m", "docker"])
