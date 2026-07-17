# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ANY_REGION, ResourceConfig

from experiments.evals.served_lm_eval import brokered_vllm_config

QWEN3_INFERENCE = brokered_vllm_config(
    model="Qwen/Qwen3-0.6B-Base",
    tokenizer="Qwen/Qwen3-0.6B",
    worker_resources=ResourceConfig.with_tpu(["v5litepod-4", "v4-8", "v5p-8", "v6e-4"], ram="96g", regions=[ANY_REGION]),
)
EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=0.5,
    ram="6g",
    disk="16g",
    regions=[ANY_REGION],
    preemptible=False,
)
