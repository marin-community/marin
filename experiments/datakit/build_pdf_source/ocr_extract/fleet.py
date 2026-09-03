# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OCR serving fleet, at the operating point the throughput sweep measured.

Four one-GPU vLLM instances behind the Marin broker pack one GB200 node, each holding
:data:`MAX_IN_FLIGHT` requests. The fleet only reaches its throughput if the senders keep
:data:`CLIENT_CONCURRENCY` requests in flight, and the extraction step sizes its Zephyr fleet from
that number. The numbers come from ``ocr-budget-sweep.md`` on the ``mark/pdf_pipeline`` campaign
branch.
"""

import math

from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    InferenceWorkerConfig,
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)

MODEL = "infly/Infinity-Parser2-Flash"

# One GPU per instance, four instances: packs one GB200 node.
INSTANCES = 4
GPU_TYPE = "GB200"
# The smallest pod shape that packs four to a node.
GPU_WORKER_CPU = 32
GPU_WORKER_RAM_GB = 160
API_SERVER_COUNT = 2

MAX_NUM_SEQS = 1024
MAX_NUM_BATCHED_TOKENS = 131_072
# Well above an image plus its answer; a smaller declared context leaves more HBM for the KV cache.
MAX_MODEL_LEN = 24_576

# Per instance. Total in-flight across the fleet is this times INSTANCES.
MAX_IN_FLIGHT = 512
CLIENT_CONCURRENCY = MAX_IN_FLIGHT * INSTANCES

# Broker memory is in-flight payload: every leased request holds its base64 PNG in the broker
# process until the worker reports the response.
_REQUEST_MB = 2.0
_BROKER_BASE_RAM_GB = 8
_BROKER_CPU = 8

# Prebuilt FlashInfer kernel artifacts, so the engine skips FlashInfer's JIT build at startup. Both
# must match the ``flashinfer-python`` version the pinned vLLM depends on.
FLASHINFER_PACKAGES = ("flashinfer-cubin==0.6.13", "flashinfer-jit-cache==0.6.13")
FLASHINFER_INDEX = "https://flashinfer.ai/whl/cu130/"

# Must satisfy 0 < worker < lease < proxy, which BrokerConfig enforces; they bound a hung request.
_WORKER_REQUEST_TIMEOUT = 900.0
_LEASE_TIMEOUT = 1020.0
_PROXY_REQUEST_TIMEOUT = 1140.0

_STARTUP_TIMEOUT = 3600
_ENDPOINT_READY_TIMEOUT = 3600.0
_WORKER_DISK = "300g"

# How long the proxy waits for the first worker to serve /v1/models. It is a weight-download budget,
# so it must never be tighter than the engine startup budget it is waiting on.
_PROXY_READINESS_TIMEOUT = float(_STARTUP_TIMEOUT)


def _vllm_extra_args() -> tuple[str, ...]:
    """vLLM flags beyond what the Marin backend already sets.

    ``--trust-remote-code`` is absent because the backend sets it. ``--enable-prefix-caching`` is
    absent because on this hybrid architecture it forces the 'align' Mamba cache mode, and crawl
    pages share no prefix worth caching.
    """
    return (
        # The GDN prefill kernel comes from FlashInfer (the prebuilt artifacts above) or Triton.
        "--gdn-prefill-backend",
        "flashinfer",
        "--reasoning-parser",
        "qwen3",
        # Share processed image tensors across the API-server processes.
        "--mm-processor-cache-type",
        "shm",
        # Multimodal preprocessing runs in the API server, ahead of the scheduler.
        "--api-server-count",
        str(API_SERVER_COUNT),
    )


def build_inference_config(instances: int = INSTANCES) -> RemoteInferenceConfig:
    """One brokered fleet of ``instances`` one-GPU engines behind a single proxy.

    Everything per-instance is fixed at the measured operating point; the broker's payload memory
    and the proxy's pending budget scale with ``instances``.
    """
    in_flight = MAX_IN_FLIGHT * instances
    broker_ram_gb = _BROKER_BASE_RAM_GB + math.ceil(in_flight * _REQUEST_MB / 1024)
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights=MODEL, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=1),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            startup_timeout_seconds=_STARTUP_TIMEOUT,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            extra_args=_vllm_extra_args(),
            uv_with_packages=FLASHINFER_PACKAGES,
            uv_extra_index_urls=(FLASHINFER_INDEX,),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                GPU_TYPE,
                count=1,
                cpu=GPU_WORKER_CPU,
                ram=f"{GPU_WORKER_RAM_GB}g",
                disk=_WORKER_DISK,
                regions=[ANY_REGION],
            ),
            worker_environment=create_environment(),
            endpoint_ready_timeout_seconds=_ENDPOINT_READY_TIMEOUT,
            # One startup port collision under the default single retry permanently shrinks the
            # fleet; the brokered path is elastic, so generous retries cost nothing when unused.
            max_retries_failure=5,
        ),
        instances=instances,
        broker=BrokerConfig(
            worker=InferenceWorkerConfig(
                max_in_flight=MAX_IN_FLIGHT,
                request_timeout_seconds=_WORKER_REQUEST_TIMEOUT,
            ),
            request_lease_timeout_seconds=_LEASE_TIMEOUT,
            broker_resources=ResourceConfig.with_cpu(
                cpu=_BROKER_CPU,
                ram=f"{broker_ram_gb}g",
                disk="20g",
                preemptible=False,
            ),
            proxy=InferenceProxyConfig(
                request_timeout_seconds=_PROXY_REQUEST_TIMEOUT,
                readiness_timeout_seconds=_PROXY_READINESS_TIMEOUT,
                # The proxy rejects past this, so it has to sit above what the senders will offer.
                max_pending_requests=in_flight * 2,
            ),
        ),
    )
