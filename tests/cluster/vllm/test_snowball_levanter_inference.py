# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify Snowball checkpoint inference against representative-prompt goldens.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/cluster/vllm/test_snowball_levanter_inference.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2

from tests.cluster.vllm.backend_parity import assert_report_matches_exact_goldens
from tests.cluster.vllm.snowball import (
    SNOWBALL_GPU,
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    SNOWBALL_TPU,
    read_native_tpu_contract,
    read_native_tpu_goldens,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_levanter import assert_native_tpu_contract, capture_native_levanter

PENDING_TIMEOUT = 20 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


def assert_checkpoint_inference_matches_golden(expected_cases) -> None:
    report = capture_native_levanter(SNOWBALL_NATIVE_GPU, goldens=expected_cases)
    assert_report_matches_exact_goldens(
        report,
        {expected.id: expected.top_logprobs for expected in expected_cases},
        score_source="canonical_tokens",
    )


def assert_tpu_checkpoint_inference_matches_contract(gpu_goldens, tpu_goldens, contract) -> None:
    report = capture_native_levanter(SNOWBALL_NATIVE_TPU, goldens=gpu_goldens)
    assert_native_tpu_contract(report, gpu_goldens, contract)
    assert_report_matches_exact_goldens(
        report,
        {expected.id: expected.top_logprobs for expected in tpu_goldens},
        score_source="top_logprobs",
    )


def test_snowball_checkpoint_matches_levanter_inference_goldens(marin_gpu_client: IrisClient, run_test_job) -> None:
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-checkpoint-inference-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_inference_matches_golden,
                args=[read_representative_goldens()],
            ),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="256g", disk="64g"),
            environment=create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter", "marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_GPU.compilation_cache_dir,
                    # XLA's auxiliary caches require local paths; keep only JAX's LOTA-backed cache.
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    # Keep BF16 kernel selection reproducible across independently compiled H100 nodes.
                    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                },
            ),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )


def test_snowball_checkpoint_matches_native_tpu_contract(
    iris_client: IrisClient,
    smoke_region: str,
    run_test_job,
) -> None:
    run_test_job(
        iris_client,
        JobRequest(
            name=f"snowball-checkpoint-inference-tpu-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_tpu_checkpoint_inference_matches_contract,
                args=[read_representative_goldens(), read_native_tpu_goldens(), read_native_tpu_contract()],
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
