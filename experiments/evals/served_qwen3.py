# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 with the brokered lm-eval suite.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-evals --cpu 1 --memory 2G \
    --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3
"""

from fray.types import ANY_REGION, ResourceConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.inference.vllm import BrokeredVllmSystemConfig, VllmProxyConfig, VllmServerConfig

from experiments.evals.brokered_eval_suite import brokered_eval_suite

_VLLM_TIMEOUT = 1800
_VLLM_WORKER_ENV_VARS = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}

QWEN3_INFERENCE = BrokeredVllmSystemConfig(
    model="Qwen/Qwen3-0.6B-Base",
    tokenizer="Qwen/Qwen3-0.6B",
    worker_resources=ResourceConfig.with_tpu(
        ["v5litepod-4", "v4-8", "v5p-8", "v6e-4"],
        ram="96g",
        regions=[ANY_REGION],
    ),
    worker_env_vars=_VLLM_WORKER_ENV_VARS,
    server=VllmServerConfig(timeout_seconds=_VLLM_TIMEOUT),
    proxy=VllmProxyConfig(
        request_timeout_seconds=_VLLM_TIMEOUT,
        readiness_timeout_seconds=_VLLM_TIMEOUT,
        ignored_request_fields=("seed",),
    ),
)

QWEN3_EVAL_RESULTS = brokered_eval_suite(
    QWEN3_INFERENCE,
    model_name="qwen3-0.6b",
    version="2026.07.17",
)

if __name__ == "__main__":
    StepRunner().run([lower(QWEN3_EVAL_RESULTS)])
