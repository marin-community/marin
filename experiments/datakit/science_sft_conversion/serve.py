# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a configurable MiniMax conversion pool from an Iris CPU task."""

import argparse
import logging

from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    InferenceWorkerConfig,
    IrisConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import IrisServiceConfig, run_iris_service

from experiments.datakit.science_sft_conversion.conversion import MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--timeout-hours", type=float, required=True)
    parser.add_argument("--cache-ttl-days", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = IrisServiceConfig(
        model=ServedModelConfig(weights=MODEL, max_model_len=65536),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            version="0.30.0",
            max_num_batched_tokens=8192,
            max_num_seqs=args.max_sequences,
            extra_args=(
                "--gpu-memory-utilization=0.97",
                "--block-size=128",
                "--reasoning-parser=minimax_m3",
                "--kv-cache-dtype=fp8",
                "--enable-expert-parallel",
            ),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                "H100", count=8, cpu=64, ram="1024g", disk="800g", regions=[ANY_REGION]
            ),
            worker_environment=create_environment(
                env_vars={"RUNAI_STREAMER_CONCURRENCY": "2", "RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS": "30000"}
            ),
            cache_ttl_days=args.cache_ttl_days,
        ),
        endpoint_name=args.endpoint,
        instances=args.instances,
        broker=BrokerConfig(
            worker=InferenceWorkerConfig(max_in_flight=args.max_sequences, request_timeout_seconds=1440),
            proxy=InferenceProxyConfig(
                max_pending_requests=2048, request_timeout_seconds=1800, readiness_timeout_seconds=1800
            ),
            request_lease_timeout_seconds=1620,
        ),
        timeout_hours=args.timeout_hours,
        controller_proxy_timeout_seconds=1800,
        port_name=None,
    )
    run_iris_service(service)


if __name__ == "__main__":
    main()
