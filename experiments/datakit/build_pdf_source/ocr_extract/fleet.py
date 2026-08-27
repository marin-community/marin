# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OCR serving fleet, at the operating point the throughput sweep measured.

Every number here comes from ``experiments/datakit/build_pdf_source/ocr-budget-sweep.md``. The short version:
four one-GPU vLLM instances behind the Marin broker serve ~71 pages/s on one GB200 node -- 15.6
GPU-hours per million pages -- at a total client concurrency of 2048, which is 512 in flight per
instance.

Three findings shape the config, and each of them contradicts an obvious guess:

* **Four one-GPU instances, not one four-GPU instance.** Per-GPU throughput is what a
  tensor-parallel serve optimises, but per-*node* throughput is what a corpus costs. The lean pod
  loses on a per-GPU basis and wins by packing four to a node.
* **More engine slots hurt.** Opening ``max_num_seqs`` to 2048 lowered throughput at matched pod and
  equal-or-higher concurrency. 1024 slots against 512 in flight is the operating point; past it,
  added in-flight buys latency and nothing else.
* **Pod shape stops mattering under the broker.** Directly load-tested, API-side CPU and server
  count set throughput and separated the pod shapes by 10-25%. Behind the broker all shapes land
  within ~3%, because the broker holds every engine at its full in-flight budget continuously where
  a client-side pool dips whenever it drains and refills. So the fleet takes the smallest pod that
  packs four to a node.

The senders are the other half of this: the fleet only reaches 71 pages/s if something keeps 2048
requests in flight. :data:`CLIENT_CONCURRENCY` is that number, and the extraction step sizes its
Zephyr fleet from it.

This module leans on serving-side work that landed with it: ``VllmEngineConfig.uv_with_packages``
/ ``uv_extra_index_urls``, which are how the prebuilt FlashInfer kernel artifacts reach the
isolated CUDA vLLM environment, and the brokered-path concurrency fixes (the inference proxy's
anyio ``to_thread`` limiter and the inference worker's httpx pool), without which a fleet quietly
serves a fraction of the throughput above.
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

# One GPU per instance, four instances: a GB200 node has four GPUs, and this packs one node.
INSTANCES = 4
GPU_TYPE = "GB200"
# The lean pod. 32 cores and two API servers is the smallest shape that packs 4/node, and behind the
# broker it came within ~1% of the 64-core pod.
GPU_WORKER_CPU = 32
GPU_WORKER_RAM_GB = 160
API_SERVER_COUNT = 2

MAX_NUM_SEQS = 1024
MAX_NUM_BATCHED_TOKENS = 131_072
# ~2.07 MP of image plus a ~700-token answer needs nothing like the model's full context, and a
# smaller declared context leaves more HBM for the KV cache.
MAX_MODEL_LEN = 24_576

# Per instance. Total in-flight across the fleet is this times INSTANCES.
MAX_IN_FLIGHT = 512
CLIENT_CONCURRENCY = MAX_IN_FLIGHT * INSTANCES

# Broker memory is in-flight payload, so it scales with the fleet: every leased request holds its
# ~1.9 MB of base64 PNG in the broker process until the worker reports the response. The margin
# covers Python object overhead and the response queue.
_REQUEST_MB = 2.0
_BROKER_BASE_RAM_GB = 8
_BROKER_CPU = 8

# Prebuilt FlashInfer kernel artifacts. CoreWeave runtime images have no nvcc, so FlashInfer's JIT
# path cannot compile; ``flashinfer-cubin`` (PyPI) ships device cubins and ``flashinfer-jit-cache``
# (FlashInfer's own index, per CUDA version) ships the prebuilt host glue. Both must match the
# ``flashinfer-python`` version the pinned vLLM depends on -- FlashInfer hard-fails on skew.
FLASHINFER_PACKAGES = ("flashinfer-cubin==0.6.13", "flashinfer-jit-cache==0.6.13")
FLASHINFER_INDEX = "https://flashinfer.ai/whl/cu130/"

# Timeouts must satisfy 0 < worker < lease < proxy, which BrokerConfig enforces. All three are far
# above the measured p50 of 21s; they exist to bound a hung request, not to shed load.
_WORKER_REQUEST_TIMEOUT = 900.0
_LEASE_TIMEOUT = 1020.0
_PROXY_REQUEST_TIMEOUT = 1140.0

_STARTUP_TIMEOUT = 3600
_ENDPOINT_READY_TIMEOUT = 3600.0
_WORKER_DISK = "300g"

# How long the proxy waits for the first worker to serve /v1/models. This is a weight-download
# budget, not a serving one, and the 300s default silently governs a fleet whose every other
# timeout is set to an hour. Flash starts in ~173s and fits under it; Infinity-Parser2-Pro takes
# ~647s and did not, so the proxy gave up mid-fetch and killed the fleet with a 504 on /v1/models
# while the workers were still healthy. Raised again to match the engine startup budget after 8 of
# 11 fleets missed a 1800s window when 176 instances cold-started at once -- under contention the
# readiness budget must never be tighter than the startup budget it is waiting on.
_PROXY_READINESS_TIMEOUT = float(_STARTUP_TIMEOUT)


def _vllm_extra_args() -> tuple[str, ...]:
    """vLLM flags beyond what the Marin backend already sets.

    ``--trust-remote-code`` is deliberately absent: the backend sets it, and passing it again logs a
    duplicate-key warning. So is ``--enable-prefix-caching``: on this hybrid architecture it forces
    the experimental 'align' Mamba cache mode, which inflates per-request state memory, and crawl
    pages share no prefix worth caching.
    """
    return (
        # Qwen3.5 is a gated-delta-net hybrid; its GDN prefill kernel comes from FlashInfer (the
        # prebuilt artifacts above) or Triton. FlashInfer measured faster and is the default here.
        "--gdn-prefill-backend",
        "flashinfer",
        "--reasoning-parser",
        "qwen3",
        # Shared-memory cache for processed image tensors, so the API-server processes doing
        # multimodal preprocessing do not each hold their own copy.
        "--mm-processor-cache-type",
        "shm",
        # A single API-server process bottlenecks ingest: multimodal preprocessing runs there,
        # ahead of the scheduler.
        "--api-server-count",
        str(API_SERVER_COUNT),
    )


def build_inference_config(instances: int = INSTANCES) -> RemoteInferenceConfig:
    """One brokered fleet of ``instances`` one-GPU engines behind a single proxy.

    Everything per-instance is fixed at the measured operating point; what scales with
    ``instances`` is the broker -- its in-flight payload memory and the proxy's pending budget are
    both linear in fleet size, so a caller sharding a large run into several fleets sizes each
    broker for exactly the fleet behind it.
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
            # The default single failure retry permanently shrank two fleets: one EADDRINUSE
            # port collision at startup and the instance never came back. The brokered path is
            # elastic -- capacity is whoever is pulling -- so generous retries cost nothing when
            # unused and keep a multi-hour batch-priority run from ratcheting down.
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
