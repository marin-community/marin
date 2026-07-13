# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare vLLM inference from the June 67B BF16 export with Levanter.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_vllm_s3_inference.py -o addopts= -vv -s
"""

import logging
import math
import os
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

import pytest
import requests
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

from .june_67b_a2b import JUNE_67B_A2B, InferenceGolden, read_inference_golden
from .remote_job import run_remote_test_job

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 30 * 60
RUNTIME_TIMEOUT = 10 * 60
RETURNED_LOGPROBS = 50
GPU_COUNT = 8
# Clean e2es stayed below these bounds; one Triton dev node measured 0.00772
# and 0.01277, so cross-node margin needs review before merging.
MAX_PROBABILITY_ERROR = 0.008
TOP_PROBABILITY_L1_ERROR = 0.012

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


# TODO(#7135): Replace this imperative overlay with marin-core[vllm-gpu]
# once #7134 provides the managed CUDA 13 PyTorch baseline.
def _vllm_setup_script() -> str:
    source = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["vllm"]
    return (
        'VLLM_USE_PRECOMPILED=1 uv pip install --no-config --python "$IRIS_VENV/bin/python" '
        f'--torch-backend=cu130 "vllm @ git+{source["git"]}@{source["rev"]}" "runai-model-streamer[s3]==0.16.0"'
    )


def assert_vllm_logprobs_match_levanter(
    expected_inference: InferenceGolden,
    attention_backend: str,
) -> None:
    expected_logprobs = {entry.token_id: entry.logprob for entry in expected_inference.top_logprobs}
    with tempfile.TemporaryDirectory(prefix="june-67b-vllm-") as temp_dir:
        aws_config = Path(temp_dir) / "aws-config"
        # Run:ai ignores Iris's FSSPEC_S3 setting, and CoreWeave rejects its default path-style requests.
        aws_config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
        os.environ["AWS_CONFIG_FILE"] = str(aws_config)

        model = ModelConfig(
            name=JUNE_67B_A2B.vllm_model_name,
            path=JUNE_67B_A2B.export_uri,
            engine_kwargs={"max_model_len": 128},
        )
        extra_args = [
            "--data-parallel-size",
            str(GPU_COUNT),
            "--enable-expert-parallel",
            "--model-loader-extra-config",
            '{"distributed":true}',
            "--max-num-seqs",
            "1",
            "--max-logprobs",
            str(RETURNED_LOGPROBS),
            "--attention-backend",
            attention_backend,
        ]

        started = time.monotonic()
        with VllmEnvironment(model=model, timeout_seconds=RUNTIME_TIMEOUT, extra_args=extra_args) as environment:
            assert environment.model_id == JUNE_67B_A2B.export_uri
            ready = time.monotonic()
            logger.info("vLLM startup logs:\n%s", environment.logs_tail(max_lines=1_000))
            rank_metrics = []
            for rank in range(GPU_COUNT):
                request_started = time.monotonic()
                response = requests.post(
                    f"{environment.server_url}/completions",
                    # Pin every request because vLLM's DP load balancer otherwise chooses the rank.
                    headers={"X-data-parallel-rank": str(rank)},
                    json={
                        "model": environment.model_id,
                        "prompt": expected_inference.prompt,
                        "add_special_tokens": False,
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "logprobs": RETURNED_LOGPROBS,
                        "return_tokens_as_token_ids": True,
                        "return_token_ids": True,
                    },
                    timeout=300,
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                assert choice["prompt_token_ids"] == expected_inference.prompt_token_ids
                assert choice["token_ids"] == [expected_inference.top_logprobs[0].token_id]
                actual_logprobs = {
                    int(token.removeprefix("token_id:")): logprob
                    for token, logprob in choice["logprobs"]["top_logprobs"][0].items()
                }
                missing_token_ids = expected_logprobs.keys() - actual_logprobs.keys()
                assert not missing_token_ids, sorted(missing_token_ids)
                max_logprob_error = max(
                    abs(actual_logprobs[token_id] - expected_logprob)
                    for token_id, expected_logprob in expected_logprobs.items()
                )
                probability_errors = [
                    abs(math.exp(actual_logprobs[token_id]) - math.exp(expected_logprob))
                    for token_id, expected_logprob in expected_logprobs.items()
                ]
                rank_metric = {
                    "rank": rank,
                    "seconds": time.monotonic() - request_started,
                    "max_abs_logprob_error": max_logprob_error,
                    "max_abs_probability_error": max(probability_errors),
                    "top_probability_l1_error": sum(probability_errors),
                }
                rank_metrics.append(rank_metric)
                assert rank_metric["max_abs_probability_error"] <= MAX_PROBABILITY_ERROR, rank_metric
                assert rank_metric["top_probability_l1_error"] <= TOP_PROBABILITY_L1_ERROR, rank_metric

            logger.info(
                "vLLM inference: %s",
                {
                    "attention_backend": attention_backend,
                    "startup_seconds": ready - started,
                    "inference_seconds": time.monotonic() - ready,
                    "rank_metrics": rank_metrics,
                },
            )


def test_h100_node_matches_levanter_logprobs(
    marin_gpu_client: IrisClient,
    vllm_attention_backend: str,
) -> None:
    expected_inference = read_inference_golden(JUNE_67B_A2B.inference_golden_path)
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"june-67b-vllm-logprobs-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_vllm_logprobs_match_levanter,
                args=[expected_inference, vllm_attention_backend],
            ),
            resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="512g", disk="64g"),
            environment=create_environment(
                setup_scripts=[default_setup_script(packages=["marin-core"]), _vllm_setup_script()],
                env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            ),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
