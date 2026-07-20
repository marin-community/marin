# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final standing-cluster gate for the exported-Levanter TPU cell.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2

from tests.cluster.vllm.backend_parity import (
    assert_report_matches_contract,
    assert_report_matches_exact_goldens,
)
from tests.cluster.vllm.snowball import (
    SNOWBALL_EXPORTED_GPU,
    SNOWBALL_EXPORTED_TPU,
    SNOWBALL_GPU,
    SNOWBALL_TPU,
    read_exported_levanter_gpu_contract,
    read_exported_levanter_tpu_contract,
    read_native_tpu_goldens,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_exported_levanter import capture_exported_levanter

PENDING_TIMEOUT = 20 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


def assert_exported_levanter_gpu_matches_contract(gpu_goldens, contract) -> None:
    report = capture_exported_levanter(SNOWBALL_EXPORTED_GPU, goldens=gpu_goldens)
    assert_report_matches_contract(
        report,
        {golden.id: golden.top_logprobs for golden in gpu_goldens},
        contract,
    )


def assert_exported_levanter_tpu_matches_contract(gpu_goldens, tpu_goldens, contract) -> None:
    report = capture_exported_levanter(SNOWBALL_EXPORTED_TPU, goldens=gpu_goldens)
    assert_report_matches_contract(
        report,
        {golden.id: golden.top_logprobs for golden in gpu_goldens},
        contract,
    )
    # Exported and native Levanter use one exact TPU snapshot because discovery
    # proved their complete 64-case outputs are bitwise identical.
    assert_report_matches_exact_goldens(
        report,
        {golden.id: golden.top_logprobs for golden in tpu_goldens},
        score_source="top_logprobs",
    )


def test_snowball_export_matches_exported_levanter_gpu_contract(
    marin_gpu_client: IrisClient,
    run_test_job,
) -> None:
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-exported-levanter-gpu-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_exported_levanter_gpu_matches_contract,
                args=[read_representative_goldens(), read_exported_levanter_gpu_contract()],
            ),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="256g", disk="128g"),
            environment=create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter", "marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_GPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                },
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )


def test_snowball_export_matches_exported_levanter_tpu_contract(
    iris_client: IrisClient,
    smoke_region: str,
    run_test_job,
) -> None:
    run_test_job(
        iris_client,
        JobRequest(
            name=f"snowball-exported-levanter-tpu-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_exported_levanter_tpu_matches_contract,
                args=[
                    read_representative_goldens(),
                    read_native_tpu_goldens(),
                    read_exported_levanter_tpu_contract(),
                ],
            ),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=64,
                ram="256g",
                disk="100g",
                regions=(smoke_region,),
            ),
            environment=create_environment(
                extras=["tpu"],
                sync_packages=["marin-levanter", "marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_TPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                },
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
