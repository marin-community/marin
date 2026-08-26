# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify schema-2 GrugMoE through the ordinary Marin serving path.

Run one case at a time and retain the complete streamed log::

    uv run pytest tests/cluster/vllm/test_grugmoe_schema2_serving.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s -k real_d1536
"""

import subprocess
import uuid
from pathlib import Path

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.testing.inference.grugmoe_schema2_serving import (
    REAL_MODEL_KEY,
    run_grugmoe_schema2_qualification,
)

from experiments.evaluation.models import models
from tests.cluster.conftest import MARIN_GPU_CLUSTER

PENDING_TIMEOUT_SECONDS = 45 * 60.0
RUNTIME_TIMEOUT_SECONDS = 75 * 60.0
ORDINARY_CATALOG_COMMAND = (
    "uv run python -m experiments.evaluation.cli launch --model rav-ladder-d1536 "
    "--evals smoke --platform gpu --accelerator H100x8 --federated_cluster cw-us-east-02a "
    "--priority interactive"
)

pytestmark = [
    pytest.mark.cluster,
    pytest.mark.slow,
    pytest.mark.timeout(PENDING_TIMEOUT_SECONDS + RUNTIME_TIMEOUT_SECONDS + 300),
]


def _clean_git_head() -> str:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise AssertionError("live qualification requires a clean committed Marin worktree")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dummy_d6144_model() -> ModelConfig:
    return ModelConfig(
        name="grugmoe-schema2-d6144-dummy-pp3",
        location="tests/cluster/vllm/resources/grugmoe_d6144_dummy",
        apply_chat_template=False,
        resource_hint=ResourceHint(
            gpu={"H100": 8},
            cpu=64,
            memory="512g",
            disk="128g",
        ),
        serve=ServeConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=3,
            data_parallel_size=8,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=64,
            vllm_batch_invariant=True,
            vllm_use_flashinfer_sampler=False,
            vllm_extra_args=(
                "--enable-expert-parallel",
                # DummyModelLoader rejects model-loader extra config. The real catalog alone uses
                # {"distributed": true}; every architecture and serving-topology setting matches.
                "--load-format",
                "dummy",
                "--skip-tokenizer-init",
                "--enforce-eager",
                "--no-enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.9",
                "--max-logprobs",
                "64",
            ),
            auto_overrides=False,
        ),
    )


def _qualification_command(case: str) -> str:
    return (
        "uv run pytest tests/cluster/vllm/test_grugmoe_schema2_serving.py "
        f"-m cluster -o addopts= --import-mode=importlib -vv -s -k {case}"
    )


@pytest.mark.parametrize("case", ["real_d1536", "dummy_d6144"])
def test_grugmoe_schema2_serving_whole_node(
    marin_gpu_client: IrisClient,
    run_test_job,
    case: str,
) -> None:
    model = models()[REAL_MODEL_KEY] if case == "real_d1536" else _dummy_d6144_model()
    request = JobRequest(
        name=f"grugmoe-schema2-{case.replace('_', '-')}-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            run_grugmoe_schema2_qualification,
            args=(case, model, _clean_git_head(), _qualification_command(case), ORDINARY_CATALOG_COMMAND),
        ),
        resources=ResourceConfig.with_cpu(
            cpu=4,
            ram="32g",
            disk="32g",
            target_cluster=MARIN_GPU_CLUSTER,
            preemptible=False,
        ),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
        ),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    run_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT_SECONDS,
        runtime_timeout=RUNTIME_TIMEOUT_SECONDS,
    )
