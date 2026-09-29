# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker lane: run the fray contract suite from the head of a three-node Ray cluster.

Requires a Docker daemon. The compose stack in ``ray_cluster/`` bind-mounts this
checkout, so the suite that runs inside the head container is the one on disk.
"""

import subprocess
from pathlib import Path

import pytest

COMPOSE_DIR = Path(__file__).parent / "ray_cluster"
CLUSTER_START_TIMEOUT = 1800
IN_CLUSTER_PYTEST = (
    "cd /workspace && /opt/venv/bin/pytest "
    "lib/fray/tests/test_client.py lib/fray/tests/test_actor.py lib/fray/tests/test_ray_backend.py "
    "-q -p no:cacheprovider --basetemp=/shared/pytest -o addopts=--import-mode=importlib"
)


@pytest.mark.docker
def test_fray_suite_on_three_node_docker_cluster():
    compose = ["docker", "compose", "--project-directory", str(COMPOSE_DIR)]
    subprocess.run([*compose, "up", "--build", "--wait", "--wait-timeout", str(CLUSTER_START_TIMEOUT)], check=True)
    try:
        subprocess.run([*compose, "exec", "-T", "head", "bash", "-c", IN_CLUSTER_PYTEST], check=True)
    finally:
        subprocess.run([*compose, "down"], check=False)
