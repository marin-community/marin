# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest wrapper for the end-to-end integration script.

Runs ``tests/integration_test.py`` against an Iris controller. If
``IRIS_CONTROLLER_URL`` is set the wrapper reuses that cluster; otherwise
it spawns a local one for the duration of the test and tears it down on
exit.
"""

import os
import re
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

MARIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONTROLLER_URL_RE = re.compile(r"Controller started at (\S+)")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skip in CI; the marin-itest workflow runs the script directly against its own cluster.",
)
def test_integration_test_run(tmp_path):
    with _iris_controller_url(tmp_path) as url:
        result = subprocess.run(
            ["uv", "run", "python", "tests/integration_test.py", "--controller-url", url],
            cwd=MARIN_ROOT,
        )
        assert result.returncode == 0, f"Integration test returned {result.returncode}"


@contextmanager
def _iris_controller_url(tmp_path) -> Iterator[str]:
    """Yield a controller URL — either from the env or from a freshly-started local cluster."""
    env_url = os.environ.get("IRIS_CONTROLLER_URL")
    if env_url:
        yield env_url
        return

    log_file = tmp_path / "iris-cluster.log"
    with log_file.open("w") as log:
        cluster = subprocess.Popen(
            ["uv", "run", "iris", "--config", "lib/iris/examples/test.yaml", "cluster", "start", "--local"],
            cwd=MARIN_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        yield _wait_for_controller_url(log_file, cluster, timeout=180)
    finally:
        _kill_process_group(cluster)


def _wait_for_controller_url(log_file, cluster: subprocess.Popen, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cluster.poll() is not None:
            raise RuntimeError(
                f"Cluster exited (code={cluster.returncode}) before printing controller URL; log at {log_file}"
            )
        if log_file.exists():
            for line in log_file.read_text().splitlines():
                match = _CONTROLLER_URL_RE.search(line)
                if match:
                    return match.group(1)
        time.sleep(1)
    raise TimeoutError(f"Cluster did not print controller URL within {timeout}s; log at {log_file}")


def _kill_process_group(cluster: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(cluster.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        cluster.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(cluster.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    cluster.wait()
