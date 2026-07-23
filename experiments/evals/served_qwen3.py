# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 with the brokered lm-eval suite.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-evals --cpu 1 --memory 2G \
    --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3

  uv run iris --config lib/iris/config/marin.yaml job run --target-cluster cw-us-east-02a \
    --job-name qwen3-gpu-evals --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3 --accelerator gpu
"""

import argparse
from enum import StrEnum
from types import MappingProxyType

from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    IrisConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.training.run_environment import env_vars_for_dependency_groups

from experiments.evals.brokered_eval_suite import brokered_eval_suite
from experiments.evals.served_lm_eval import BrokeredEvalInference

QWEN3_EVAL_VERSION = "2026.07.17"
_VLLM_TIMEOUT = 1800
_TPU_VLLM_WORKER_ENV_VARS = (
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
    ("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1"),
    ("VLLM_TPU_SKIP_PRECOMPILE", "1"),
)


class Accelerator(StrEnum):
    TPU = "tpu"
    GPU = "gpu"


def qwen3_inference_config(
    *,
    engine: VllmEngineConfig,
    worker_resources: ResourceConfig,
    worker_extras: tuple[str, ...],
    worker_env_vars: tuple[tuple[str, str], ...],
) -> BrokeredEvalInference:
    """Compose Qwen3 model policy with an accelerator-specific serving backend."""
    return BrokeredEvalInference(
        model=ServedModelConfig(model="Qwen/Qwen3-0.6B-Base", tokenizer="Qwen/Qwen3-0.6B"),
        engine=engine,
        broker=BrokerConfig(
            proxy=InferenceProxyConfig(
                request_timeout_seconds=_VLLM_TIMEOUT,
                readiness_timeout_seconds=_VLLM_TIMEOUT,
                ignored_request_fields=("seed",),
            )
        ),
        iris=IrisConfig(
            worker_resources=worker_resources,
            worker_environment=create_environment(
                extras=worker_extras,
                env_vars=env_vars_for_dependency_groups(
                    worker_resources,
                    list(worker_extras),
                    dict(worker_env_vars),
                ),
            ),
        ),
    )


QWEN3_TPU_INFERENCE = qwen3_inference_config(
    engine=VllmEngineConfig(startup_timeout_seconds=_VLLM_TIMEOUT),
    worker_resources=ResourceConfig.with_tpu(
        ["v5litepod-4", "v4-8", "v5p-8", "v6e-4"],
        ram="96g",
        regions=[ANY_REGION],
    ),
    worker_extras=("tpu", "vllm"),
    worker_env_vars=_TPU_VLLM_WORKER_ENV_VARS,
)

QWEN3_TPU_EVAL_RESULTS = brokered_eval_suite(
    QWEN3_TPU_INFERENCE,
    model_name="qwen3-0.6b",
    version=QWEN3_EVAL_VERSION,
)

QWEN3_GPU_INFERENCE = qwen3_inference_config(
    engine=VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        startup_timeout_seconds=_VLLM_TIMEOUT,
    ),
    worker_resources=ResourceConfig.with_gpu(
        "H100",
        count=1,
        cpu=8,
        ram="64g",
        disk="100g",
        regions=[ANY_REGION],
    ),
    worker_extras=(),
    worker_env_vars=(),
)

QWEN3_GPU_EVAL_RESULTS = brokered_eval_suite(
    QWEN3_GPU_INFERENCE,
    model_name="qwen3-0.6b-gpu",
    version=QWEN3_EVAL_VERSION,
)

QWEN3_EVALS_BY_ACCELERATOR = MappingProxyType(
    {
        Accelerator.TPU: QWEN3_TPU_EVAL_RESULTS,
        Accelerator.GPU: QWEN3_GPU_EVAL_RESULTS,
    }
)


def _parse_accelerator() -> Accelerator:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accelerator",
        type=Accelerator,
        choices=tuple(Accelerator),
        default=Accelerator.TPU,
    )
    return parser.parse_args().accelerator


if __name__ == "__main__":
    StepRunner().run([lower(QWEN3_EVALS_BY_ACCELERATOR[_parse_accelerator()])])
