# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare TPU vLLM inference from the June 67B BF16 export with Levanter.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_vllm_tpu_inference.py -o addopts= -vv -s
"""

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

import pytest
import requests
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.rpc import job_pb2
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem import StoragePath

from .iris import run_remote_test_job
from .reference import (
    CHECKPOINT_NAME,
    EXPORT_TREE_SHA256,
    GCS_MODEL_COMPLETION_URI,
    GCS_MODEL_URI,
    RETURNED_LOGPROBS,
    InferenceReference,
    assert_completion_matches_reference,
    completion_request,
    read_inference_reference,
)

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 30 * 60
RUNTIME_TIMEOUT = 90 * 60
TPU_COUNT = 8
LOCAL_MODEL_DIR = Path(f"/dev/shm/june-67b-a2b-{CHECKPOINT_NAME}-hf-bf16-vllm")
JAX_COMPILATION_CACHE_DIR = f"gs://marin-us-east5/compilation-cache/vllm/june-67b-a2b-{CHECKPOINT_NAME}-v6e-8"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.tpu_ci,
    pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60),
]


def _stage_same_region_model_metadata() -> Path:
    assert StoragePath(GCS_MODEL_COMPLETION_URI).read_text().strip() == EXPORT_TREE_SHA256
    shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
    LOCAL_MODEL_DIR.mkdir(parents=True)
    root = StoragePath(GCS_MODEL_URI)
    for directory, _subdirectories, filenames in root.walk():
        for filename in filenames:
            source = directory / filename
            if source.name.endswith(".safetensors"):
                continue
            destination = LOCAL_MODEL_DIR / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as remote, destination.open("wb") as local:
                shutil.copyfileobj(remote, local, length=8 * 1024 * 1024)
    assert (LOCAL_MODEL_DIR / "config.json").is_file()
    assert (LOCAL_MODEL_DIR / "model.safetensors.index.json").is_file()
    return LOCAL_MODEL_DIR


def assert_tpu_vllm_logprobs_match_levanter(expected: InferenceReference) -> None:
    model_path = _stage_same_region_model_metadata()
    model = ModelConfig(
        name=f"june-67b-a2b-{CHECKPOINT_NAME}-bf16-tpu",
        path=str(model_path),
        engine_kwargs={"max_model_len": 128},
    )
    additional_config = {
        "grugmoe_weights_uri": GCS_MODEL_URI,
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": True,
                "attn_dp_size": TPU_COUNT,
            }
        },
    }
    extra_args = [
        "--tensor-parallel-size",
        str(TPU_COUNT),
        "--max-num-seqs",
        str(TPU_COUNT),
        "--max-num-batched-tokens",
        "128",
        "--max-logprobs",
        str(RETURNED_LOGPROBS),
        # v6e defaults auto KV caches to FP8; keep this parity test on the BF16 contract.
        "--kv-cache-dtype",
        "bfloat16",
        "--additional-config",
        json.dumps(additional_config, separators=(",", ":")),
    ]

    started = time.monotonic()
    with VllmEnvironment(model=model, timeout_seconds=RUNTIME_TIMEOUT, extra_args=extra_args) as environment:
        ready = time.monotonic()
        logger.info("TPU vLLM startup logs:\n%s", environment.logs_tail(max_lines=2_000))
        request = completion_request(expected)
        request["model"] = environment.model_id
        # One sequence per attention-DP lane exercises the full eight-device mesh in one decode.
        request["prompt"] = [expected.prompt] * TPU_COUNT
        response = requests.post(
            f"{environment.server_url}/completions",
            json=request,
            timeout=600,
        )
        response.raise_for_status()
        choices = sorted(response.json()["choices"], key=lambda choice: choice["index"])
        assert len(choices) == TPU_COUNT
        lane_metrics = [
            assert_completion_matches_reference(expected, choice, lane=lane) for lane, choice in enumerate(choices)
        ]
        logger.info(
            "TPU vLLM inference: %s",
            {
                "startup_seconds": ready - started,
                "inference_seconds": time.monotonic() - ready,
                "lane_metrics": lane_metrics,
            },
        )


def test_v6e_8_matches_levanter_logprobs(marin_tpu_client: IrisClient) -> None:
    expected = read_inference_reference()
    run_remote_test_job(
        marin_tpu_client,
        JobRequest(
            name=f"june-67b-vllm-tpu-logprobs-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_tpu_vllm_logprobs_match_levanter,
                args=[expected],
            ),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=160,
                ram="640g",
                disk="64g",
                zone="us-east5-b",
            ),
            environment=create_environment(
                extras=["tpu", "vllm"],
                sync_packages=["marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": JAX_COMPILATION_CACHE_DIR,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    "MODEL_IMPL_TYPE": "flax_nnx",
                    "NEW_MODEL_DESIGN": "1",
                    "TMPDIR": "/dev/shm",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                },
            ),
            # This manually-triggered E2E must preempt lower-priority work to acquire scarce v6e-8 capacity.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
