# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from fray.types import ANY_REGION, ResourceConfig
from marin.inference.vllm import BrokeredVllmSystemConfig

VLLM_WORKER_ENV_VARS = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}

_QWEN3_INFERENCE = BrokeredVllmSystemConfig(
    model="Qwen/Qwen3-0.6B-Base",
    tokenizer="Qwen/Qwen3-0.6B",
    worker_resources=ResourceConfig.with_tpu(["v5litepod-4", "v4-8", "v5p-8", "v6e-4"], ram="96g", regions=[ANY_REGION]),
    worker_env_vars=VLLM_WORKER_ENV_VARS,
)
QWEN3_INFERENCE = replace(
    _QWEN3_INFERENCE,
    proxy=replace(
        _QWEN3_INFERENCE.proxy,
        request_timeout_seconds=_QWEN3_INFERENCE.server.timeout_seconds,
        readiness_timeout_seconds=_QWEN3_INFERENCE.server.timeout_seconds,
        ignored_request_fields=("seed",),
    ),
)
EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=0.5,
    ram="6g",
    disk="16g",
    regions=[ANY_REGION],
    preemptible=False,
)
