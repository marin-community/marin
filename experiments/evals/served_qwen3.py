# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a pinned Qwen model and evaluate published TaskSpec corpora.

Invoke inside an Iris job with --config RUN.json --dataset CORPUS --output RESULTS
and --accelerator gpu, near the corpus and result storage. The run config supplies
the Hugging Face model/tokenizer names and immutable revisions. Judge tasks also
need --judge-endpoint and the credential named by --judge-api-key-env.
"""

import argparse
import uuid
from dataclasses import replace
from enum import StrEnum

from fray.types import ANY_REGION, ResourceConfig, create_environment
from iris.client.client import iris_ctx
from iris.cluster.types import ResourceSpec
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference
from marin.rollouts.evaluation import EvaluationRunConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.evaluation.task_driver import (
    JUDGE_API_KEY,
    JUDGE_ENDPOINT,
    SOLVER_API_KEY,
    SOLVER_ENDPOINT,
    WORKER_TIMEOUT,
    submit_evaluation,
)

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
) -> RemoteInferenceConfig:
    """Compose Qwen3 model policy with an accelerator-specific serving backend."""
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights="Qwen/Qwen3-0.6B-Base", tokenizer="Qwen/Qwen3-0.6B"),
        engine=engine,
        broker=BrokerConfig(
            proxy=InferenceProxyConfig(
                request_timeout_seconds=_VLLM_TIMEOUT,
                readiness_timeout_seconds=_VLLM_TIMEOUT,
            )
        ),
        iris=IrisConfig(
            worker_resources=worker_resources,
            worker_environment=create_environment(
                extras=worker_extras,
                env_vars=dict(worker_env_vars),
            ),
        ),
        instances=1,
        capability_origin=None,
    )


QWEN3_TPU_INFERENCE = qwen3_inference_config(
    engine=VllmEngineConfig(launcher=VllmLauncherType.TPU, startup_timeout_seconds=_VLLM_TIMEOUT),
    worker_resources=ResourceConfig.with_tpu(
        ["v5litepod-4", "v4-8", "v5p-8", "v6e-4"],
        ram="96g",
        regions=[ANY_REGION],
    ),
    worker_extras=("tpu",),
    worker_env_vars=_TPU_VLLM_WORKER_ENV_VARS,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", type=Accelerator, choices=tuple(Accelerator), default=Accelerator.GPU)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-endpoint", help="Explicit judge endpoint for the configured judge deployment.")
    parser.add_argument("--judge-api-key-env", default="VERIFYIT_JUDGE_API_KEY")
    args = parser.parse_args()
    config = EvaluationRunConfig.model_validate_json(StoragePath(args.config).read_text())
    if (config.judge is not None) != (args.judge_endpoint is not None):
        parser.error("The run's judge identity and --judge-endpoint must be supplied together")
    secrets = {}
    if args.judge_endpoint:
        secrets[JUDGE_ENDPOINT] = args.judge_endpoint
        secrets[JUDGE_API_KEY] = require_secret(args.judge_api_key_env)
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Qwen evaluation must run inside an Iris job with a controller client")
    inference = QWEN3_GPU_INFERENCE if args.accelerator is Accelerator.GPU else QWEN3_TPU_INFERENCE
    inference = replace(
        inference,
        model=ServedModelConfig(
            weights=config.model.name,
            revision=config.model.revision,
            tokenizer=config.model.tokenizer,
            tokenizer_revision=config.model.tokenizer_revision,
            max_model_len=config.model.context_window,
        ),
        iris=replace(
            inference.iris, worker_resources=replace(inference.iris.worker_resources, target_cluster=args.cluster)
        ),
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        job = submit_evaluation(
            client,
            tuple(args.dataset),
            args.output,
            config,
            name=f"qwen-task-eval-{uuid.uuid4().hex[:8]}",
            resources=ResourceSpec(cpu=4, memory="32g", disk="64g"),
            timeout=WORKER_TIMEOUT,
            secrets={**secrets, SOLVER_ENDPOINT: endpoint.base_url, SOLVER_API_KEY: endpoint.api_key or "unused"},
        )
        job.wait(timeout=WORKER_TIMEOUT, stream_logs=True)


if __name__ == "__main__":
    main()
