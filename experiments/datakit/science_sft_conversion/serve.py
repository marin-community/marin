# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a configurable MiniMax conversion pool from an Iris CPU task."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import fsspec
from fray.types import ANY_REGION, ResourceConfig, create_environment
from levanter.model_cache import (
    DEFAULT_COMPLETE_MARKER,
    CacheMetadata,
    cache_to_prefix,
    resolve_cached_model_path,
)
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
from rigging.filesystem.buckets import filesystem_for

MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
MODEL_REVISION = "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
logger = logging.getLogger(__name__)


def _copy_cached_snapshot(source: str, filesystem: fsspec.AbstractFileSystem, destination: str) -> None:
    with filesystem.open(f"{source}/{DEFAULT_COMPLETE_MARKER}") as stream:
        metadata = json.load(stream)
    if metadata["source_revision"] != MODEL_REVISION:
        raise ValueError("Cached MiniMax snapshot has the wrong model revision")
    files = [
        name
        for name, info in filesystem.find(source, detail=True).items()
        if info["type"] == "file" and not name.endswith(f"/{DEFAULT_COMPLETE_MARKER}")
    ]
    if not files:
        raise ValueError("Cached MiniMax snapshot contains no files")
    source_path = source.removeprefix("s3://").rstrip("/")

    def copy_file(name: str) -> None:
        relative = name.removeprefix(f"{source_path}/")
        target = f"{destination}/{relative}"
        # Both paths are in one CoreWeave bucket; S3 copies stay inside object storage.
        filesystem.cp_file(name, target)
        if filesystem.info(name)["size"] != filesystem.info(target)["size"]:
            raise ValueError(f"MiniMax snapshot copy size mismatch for {relative}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy_file, files))
    logger.info("Retained %d MiniMax snapshot files at %s", len(files), destination)


def retain_model(destination: str) -> str:
    """Retain the cached checkpoint outside TTL storage without downloading its weights."""
    bucket = "s3://marin-us-east-02a/"
    if not destination.startswith(bucket):
        raise ValueError("Retained MiniMax weights must stay in the RNO2A CoreWeave bucket")
    source = resolve_cached_model_path(f"{MODEL}@{MODEL_REVISION}", cache_ttl_days=14, cache_prefix="quick-serve-models")
    if not source.startswith(bucket):
        raise ValueError("MiniMax cache must be in the same CoreWeave bucket as the retained weights")
    retained = cache_to_prefix(
        destination, partial(_copy_cached_snapshot, source), metadata=CacheMetadata(source_revision=MODEL_REVISION)
    )
    filesystem, path = filesystem_for(retained)
    with filesystem.open(f"{path}/{DEFAULT_COMPLETE_MARKER}") as stream:
        metadata = json.load(stream)
    if metadata["source_revision"] != MODEL_REVISION:
        raise ValueError("Retained MiniMax snapshot has the wrong model revision")
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--timeout-hours", type=float, required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=3600)
    parser.add_argument("--cache-ttl-days", type=int, default=14)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--model-cache-path", help="Retain the cached weights at this non-TTL CoreWeave prefix")
    args = parser.parse_args()
    if args.tensor_parallel_size * args.data_parallel_size != 8:
        parser.error("tensor-parallel-size * data-parallel-size must equal the eight GPUs per worker")
    logging.basicConfig(level=logging.INFO)
    weights = retain_model(args.model_cache_path) if args.model_cache_path else MODEL
    service = IrisServiceConfig(
        model=ServedModelConfig(
            weights=weights,
            revision=MODEL_REVISION if weights == MODEL else None,
            api_model=MODEL,
            tokenizer=MODEL,
            tokenizer_revision=MODEL_REVISION,
            max_model_len=65536,
            tensor_parallel_size=args.tensor_parallel_size,
        ),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            version="0.30.0",
            startup_timeout_seconds=args.startup_timeout_seconds,
            max_num_batched_tokens=8192,
            max_num_seqs=args.max_sequences,
            extra_args=(
                "--gpu-memory-utilization=0.97",
                "--block-size=128",
                "--reasoning-parser=minimax_m3",
                "--kv-cache-dtype=fp8",
                "--enable-expert-parallel",
                f"--data-parallel-size={args.data_parallel_size}",
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
            broker_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="20g", preemptible=False),
            worker=InferenceWorkerConfig(max_in_flight=args.max_sequences, request_timeout_seconds=1440),
            proxy=InferenceProxyConfig(
                max_pending_requests=2048,
                request_timeout_seconds=1800,
                readiness_timeout_seconds=args.startup_timeout_seconds,
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
