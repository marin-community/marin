# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Snowball repeat-determinism measurement as remote Iris jobs.

Retest for https://github.com/marin-community/marin/issues/10699. Each parameter is one
fresh DP8 launch on eight H100s; compare the logged ``REPEAT_DETERMINISM_RESULT`` lines::

    uv run pytest tests/cluster/vllm/test_snowball_repeat_determinism.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.testing.inference.snowball import read_representative_goldens
from marin.testing.inference.snowball_backend_parity_jobs import GPU_COUNT
from marin.testing.inference.snowball_repeat_determinism_jobs import (
    CompileMode,
    measure_vllm_repeat_determinism,
)

# The same gd-8xh100ib-i128 nodes as cw-us-east-02a, which was fully held when this ran.
H100_CLUSTER = "cw-rno2a"

PENDING_TIMEOUT = 45 * 60.0
RUNTIME_TIMEOUT = 45 * 60.0

# Launch label -> compile mode. The two deterministic launches are compared with each other.
LAUNCHES = {
    "deterministic-a": CompileMode.DETERMINISTIC,
    "deterministic-b": CompileMode.DETERMINISTIC,
    "default": CompileMode.DEFAULT,
}


def _job(label: str, attention_backend: str) -> JobRequest:
    compile_mode = LAUNCHES[label]
    env_vars = {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    if compile_mode is CompileMode.DETERMINISTIC:
        env_vars["TORCHINDUCTOR_DETERMINISTIC"] = "1"
    return JobRequest(
        name=f"snowball-repeat-determinism-{label}-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            measure_vllm_repeat_determinism,
            args=[read_representative_goldens(), attention_backend, compile_mode],
        ),
        resources=ResourceConfig.with_gpu(
            "H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g", target_cluster=H100_CLUSTER
        ),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars=env_vars,
        ),
        # An ad-hoc measurement: normal band, and no automatic relaunch of a failed serve.
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_failure=0,
        max_task_failures=0,
    )


@pytest.mark.cluster
@pytest.mark.slow
@pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)
@pytest.mark.parametrize("label", list(LAUNCHES))
def test_snowball_repeat_determinism(
    marin_gpu_client: IrisClient,
    label: str,
    vllm_attention_backend: str,
    run_test_job,
) -> None:
    run_test_job(
        marin_gpu_client,
        _job(label, vllm_attention_backend),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
