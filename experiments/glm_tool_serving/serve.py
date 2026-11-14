# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a dedicated two-node GLM replica using Marin's existing inference service."""

import argparse
import json
import logging
import time

from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    ServingGeometry,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference
from rigging.log_setup import configure_logging

REVISION = "ba32f8c19c8738e2c57c71798c601d477a7bf844"
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    configure_logging()
    config = RemoteInferenceConfig(
        model=ServedModelConfig(
            weights="zai-org/GLM-5.3",
            revision=REVISION,
            api_model="glm-5.3",
            max_model_len=262144,
            tensor_parallel_size=8,
        ),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            version="0.29.0",
            startup_timeout_seconds=7200,
            max_num_batched_tokens=8192,
            max_num_seqs=32,
            extra_args=("--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.90", "--reasoning-parser", "glm47"),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                "H100",
                count=8,
                replicas=2,
                cpu=32,
                ram="512g",
                disk="1500g",
                regions=[ANY_REGION],
            ),
            worker_environment=create_environment(extras=("cpu",)),
            serving_geometry=ServingGeometry(8, 1, 2, 8),
            cache_ttl_days=0,
            endpoint_ready_timeout_seconds=10800,
            endpoint_health_timeout_seconds=7200,
            max_retries_failure=0,
            max_retries_preemption=0,
        ),
    )
    with remote_inference(config) as session:
        session.wait_until_ready()
        logger.info(
            "GLM_TOOL_READY %s",
            json.dumps(
                {
                    "endpoint_name": session.endpoint_name,
                    "revision": REVISION,
                    "gpus": 16,
                    "context": 262144,
                    "max_sequences": 32,
                }
            ),
        )
        deadline = time.monotonic() + args.hours * 3600
        while time.monotonic() < deadline:
            time.sleep(30)
            session.check_alive()


if __name__ == "__main__":
    main()
