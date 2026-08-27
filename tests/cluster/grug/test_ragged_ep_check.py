# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the 4-GPU ragged all-to-all correctness gate to the standing GB200 cluster.

The gate itself is `marin.testing.moe.ragged_ep_gate`, which the worker imports; this file only
submits it. All-to-all is GPU-only, so it cannot run in CPU CI. It carries the ``cluster`` marker
and is deselected by default; submit it from the repository root with::

    uv run pytest tests/cluster/grug/test_ragged_ep_check.py \
      -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s

Nothing runs it on a schedule. #8605 deleted the Marin Cluster Smoke workflow after 38 scheduled
runs in a row timed out, so no workflow runs the ``cluster`` marker at all; #8704 tracks giving
accelerator tests a runner again.

PYTEST_DONT_REWRITE: the step dispatches serialized remote functions that must not depend on
pytest.
"""

import pytest
from iris.client.client import IrisClient
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.testing.moe.ragged_ep_gate import build_benchmark

from tests.cluster.conftest import MARIN_GB200_CLUSTER

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)]


def test_the_ragged_ep_transport_matches_a_dense_reference(iris_client: IrisClient) -> None:
    """Run the gate and require the job to succeed.

    The gate raises when either half of its verdict fails, so a returning job is the assertion.
    ``iris_client`` binds the marin hub as the current Fray client, and the resources carry
    ``target_cluster``, so the hub federates the work to the GB200 peer.
    """
    StepRunner().run([lower(build_benchmark(target_cluster=MARIN_GB200_CLUSTER, version="dev"))])
