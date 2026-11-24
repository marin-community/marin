# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned single-tray GLM-5.3-Flash recipe for independent Iris replicas."""

from fray.types import ResourceConfig, create_environment
from iris.rpc.proto_display import priority_band_value
from marin.inference.config import (
    IrisConfig,
    ObjectStoreLoadMode,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)

MODEL_REPO = "zai-org/GLM-5.3-Flash"
MODEL_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
SERVED_MODEL_NAME = "glm-5.3"
CLUSTER = "cw-us-east-08a"
WEIGHTS_PREFIX = f"s3://marin-us-east-08a/marin/users/benfeuer/models/glm-5.3-flash/{MODEL_REVISION}"
ENDPOINT_NAME = "/serve/glm-5-3-flash"
CONTROLLER_PROXY_ORIGIN = "http://iris-controller-svc:10000"
GPUS_PER_ENGINE = 4
POOL_INSTANCES = 32
MAX_MODEL_LEN = 262_144
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 32_768
GPU_MEMORY_UTILIZATION = 0.90
GLM_PARSER = "glm47"
REQUEST_TIMEOUT_SECONDS = 3600.0
MAX_RETRIES_PREEMPTION = 10
PRIORITY = priority_band_value("production")
SERVICE_TIMEOUT_HOURS = 72.0
JOB_PREFIX = "glm-synthesis-serving"


def engine() -> VllmEngineConfig:
    """Use the pinned Marin CUDA vLLM fork with compilation and graphs enabled."""
    return VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        startup_timeout_seconds=int(REQUEST_TIMEOUT_SECONDS),
        extra_args=(
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--kv-cache-dtype",
            "fp8",
            "--enable-prefix-caching",
            "--tool-call-parser",
            GLM_PARSER,
            "--reasoning-parser",
            GLM_PARSER,
            "--enable-auto-tool-choice",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
        ),
    )


def model() -> ServedModelConfig:
    return ServedModelConfig(
        weights=WEIGHTS_PREFIX,
        api_model=SERVED_MODEL_NAME,
        object_store_load_mode=ObjectStoreLoadMode.STREAM,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=GPUS_PER_ENGINE,
    )


def worker_resources() -> ResourceConfig:
    """One GB200x4 tray, submitted directly to the pool controller.

    Streaming the full 306 GiB checkpoint exceeded 256 GiB of host memory;
    512 GiB accommodates its load-time buffers.
    """
    return ResourceConfig.with_gpu("GB200", count=GPUS_PER_ENGINE, cpu=16.0, ram="512g", disk="200g", preemptible=False)


def iris_config() -> IrisConfig:
    return IrisConfig(
        worker_resources=worker_resources(),
        worker_environment=create_environment(),
    )
