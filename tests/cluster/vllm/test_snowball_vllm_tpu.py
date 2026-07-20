# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final standing-cluster gate for numerical and serving parity on TPU vLLM.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import (
    assert_report_matches_contract,
    assert_report_matches_exact_goldens,
)
from tests.cluster.vllm.snowball import (
    SNOWBALL_VLLM_TPU,
    read_representative_goldens,
    read_vllm_tpu_contract,
    read_vllm_tpu_goldens,
)
from tests.cluster.vllm.snowball_vllm import capture_vllm
from tests.cluster.vllm.snowball_vllm_production import capture_production_behavior
from tests.cluster.vllm.snowball_vllm_production_oracle import (
    assert_production_behavior_matches_oracle,
    read_production_behavior_oracle,
)

PENDING_TIMEOUT = 20 * 60.0
RUNTIME_TIMEOUT = 60 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]
E2E_REPORT_ROOT = "gs://marin-us-east5/tmp/ttl=30d/snowball-parity/e2e"


def assert_vllm_tpu_matches_contracts(
    gpu_goldens,
    tpu_goldens,
    numerical_contract,
    production_oracle,
    numerical_report_uri,
    production_report_uri,
) -> None:
    numerical_report = capture_vllm(SNOWBALL_VLLM_TPU, goldens=gpu_goldens)
    StoragePath(numerical_report_uri).write_bytes(numerical_report.to_json_bytes())

    numerical_failures = []
    try:
        assert_report_matches_contract(
            numerical_report,
            {golden.id: golden.top_logprobs for golden in gpu_goldens},
            numerical_contract,
        )
    except AssertionError as error:
        numerical_failures.append(f"GPU-canonical contract:\n{error}")
    try:
        assert_report_matches_exact_goldens(
            numerical_report,
            {golden.id: golden.top_logprobs for golden in tpu_goldens},
            score_source="top_logprobs",
        )
    except AssertionError as error:
        numerical_failures.append(f"Exact TPU golden:\n{error}")
    if numerical_failures:
        raise AssertionError(
            f"Numerical report persisted before validation: {numerical_report_uri}\n\n" + "\n\n".join(numerical_failures)
        )

    production_report = capture_production_behavior(SNOWBALL_VLLM_TPU, goldens=gpu_goldens)
    StoragePath(production_report_uri).write_bytes(production_report.to_json_bytes())
    assert_production_behavior_matches_oracle(production_report, production_oracle)


def test_snowball_vllm_tpu_matches_numerical_and_production_contracts(
    iris_client: IrisClient,
    smoke_region: str,
    run_test_job,
) -> None:
    run_id = uuid.uuid4().hex
    artifact_root = StoragePath(E2E_REPORT_ROOT) / run_id
    run_test_job(
        iris_client,
        JobRequest(
            name=f"snowball-vllm-tpu-{run_id[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_vllm_tpu_matches_contracts,
                args=[
                    read_representative_goldens(),
                    read_vllm_tpu_goldens(),
                    read_vllm_tpu_contract(),
                    read_production_behavior_oracle(),
                    str(artifact_root / "numerical.json"),
                    str(artifact_root / "production.json"),
                ],
            ),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=160,
                ram="640g",
                disk="100g",
                regions=(smoke_region,),
            ),
            environment=create_environment(extras=["tpu"], sync_packages=["marin-core"]),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
