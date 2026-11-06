# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Snowball training-forward parity check as a remote Iris job.

Scores the model the training path builds at every position of the representative prompts against the stored
vLLM position reference. It is not scheduled; run it by hand from the repository root::

    uv run pytest tests/cluster/vllm/test_snowball_training_parity.py \\
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from marin.testing.inference.snowball import VLLM_POSITION_REFERENCE_SHA256
from marin.testing.inference.snowball_backend_parity_jobs import GPU_COUNT
from marin.testing.inference.snowball_training_parity import score_training_forward_against_vllm_positions

from tests.cluster.conftest import MARIN_GPU_CLUSTER

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 60 * 60.0

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)]


def _training_parity_job() -> JobRequest:
    return JobRequest(
        name=f"snowball-training-parity-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_training_forward_against_vllm_positions),
        resources=ResourceConfig.with_gpu(
            "H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g", target_cluster=MARIN_GPU_CLUSTER
        ),
        environment=create_environment(
            extras=["gpu"],
            sync_packages=["marin-levanter", "marin-core"],
            env_vars={
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                # The FP32 master copy and the BF16 compute view coexist while the view is cast.
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def test_snowball_training_forward_matches_vllm_positions(marin_gpu_client: IrisClient, run_test_job) -> None:
    if not VLLM_POSITION_REFERENCE_SHA256:
        pytest.skip("The vLLM position reference is not staged; see VLLM_POSITION_REFERENCE_SHA256")
    run_test_job(
        marin_gpu_client,
        _training_parity_job(),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
