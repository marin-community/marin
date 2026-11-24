# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serving configuration for the GLM-5.3-Flash synthesis pool — single source of truth.

Every launcher (staging, coordinator, watchdog, benchmark) builds its resources from
this module so a config change cannot drift between the jobs that share it.

Model identity: zai-org/GLM-5.3-Flash is the FP8 checkpoint behind the relay model
name "glm-5.3" that the tool-conversion drivers configure (production_data/config-1.json).
The endpoint serves the checkpoint at a pinned revision under the same "glm-5.3" name,
so driver request bodies work unmodified.

Engine shape: TP4 on one GB200x4 tray per engine — the vLLM recipe's validated FP8
configuration for this checkpoint (NVLink within the tray). Engines are independent
data-parallel replicas behind the inference broker; expert parallelism stays off.

Request shape (sampled from 120 real conversion receipts, 2026-11 ortet batches):
prompt p50 35.8k / p90 72.9k / p99 114.2k tokens, completion p50 832 / p90 2837
tokens, ~96% of tokens are prefill, strong prompt-prefix sharing. That drives the
large max_num_batched_tokens and mandatory prefix caching below.
"""

from fray.types import ResourceConfig, create_environment
from iris.rpc.proto_display import priority_band_value
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    InferenceWorkerConfig,
    IrisConfig,
    ObjectStoreLoadMode,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)
from marin.inference.iris import IrisServiceConfig, run_iris_service  # noqa: F401 (re-exported entrypoint)

MODEL_REPO = "zai-org/GLM-5.3-Flash"
# Immutable commit of the FP8 checkpoint the pool serves.
MODEL_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
# The tool-conversion drivers' model name; the endpoint serves under this id so
# driver bodies (cfg["model"]["model"]) need no override.
SERVED_MODEL_NAME = "glm-5.3"

CLUSTER = "cw-us-east-08a"
# Durable in-region staging for the pinned weights (CoreWeave 08a bucket; lifecycle
# rules only apply under tmp/ttl=Nd, so this prefix persists).
WEIGHTS_PREFIX = f"s3://marin-us-east-08a/marin/users/benfeuer/models/glm-5.3-flash/{MODEL_REVISION}"
ENDPOINT_NAME = "/serve/glm-5-3-flash"

GPUS_PER_ENGINE = 4
POOL_INSTANCES = 32
WARM_RUN_INSTANCES = 1

# Driver context contract (production_data/config-1.json: context_tokens 262144,
# output_tokens 32768; the driver enforces prompt + output <= context).
MAX_MODEL_LEN = 262_144
# Concurrency cap that keeps peak KV demand inside the cache: p99 prompts are
# ~114k tokens + 32k output, so worst-case concurrent demand at N seqs is
# ~146k x N tokens; the engine's KV pool holds ~2M (fp8 KV). Above ~16 seqs the
# scheduler starts preempting, and preemption crashes these workers with
# CUDA_ERROR_ILLEGAL_ADDRESS (17/32 workers died when 24 seqs ran on halved
# bf16-KV pools). Throughput is prefill-bound (~37k tok/s/engine at c=24
# measured the same as at c=8), so the cap moves queueing to the broker instead
# of cutting rows/h. Do not raise without re-validating crash behavior.
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 32_768
GPU_MEMORY_UTILIZATION = 0.90

# Broker budget: 32 engines x WORKER_MAX_IN_FLIGHT = 2048 concurrent requests against
# the conversion fleet's observed peak (~768) plus later synthesis consumers. Request
# timeouts stay above the driver's 1800s so the server never abandons work the client
# is still waiting on.
WORKER_MAX_IN_FLIGHT = 64
REQUEST_TIMEOUT_SECONDS = 3600.0
MAX_PENDING_REQUESTS = 4096
# A dead engine is restarted twice by Iris before the watchdog resubmits the whole
# coordinator; that bounds GPU churn from a permanently broken engine image.
WORKER_MAX_RETRIES_FAILURE = 2
MAX_RETRIES_PREEMPTION = 10

PRIORITY = priority_band_value("production")
# The coordinator self-stops on the 3-day default; the watchdog resubmits it, so the
# pool stays up until the coordinator says otherwise.
COORDINATOR_TIMEOUT_HOURS = 72.0
JOB_PREFIX = "glm-synthesis-serving"


def engine(extra_args: tuple[str, ...] = ()) -> VllmEngineConfig:
    """The pool's vLLM engine: pinned Marin fork, throughput-oriented GPU arguments.

    Compilation and CUDA graphs stay at their defaults (never enforce-eager). The
    glm47 parsers cover tool-calling consumers; text-only synthesis requests are
    unaffected by them.
    """
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
            "glm47",
            "--reasoning-parser",
            "glm47",
            "--enable-auto-tool-choice",
            # Text-only synthesis pool: skip vision-encoder profiling and keep its
            # memory budget for KV cache.
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            *extra_args,
        ),
    )


def model() -> ServedModelConfig:
    """The served model: staged in-region weights pinned to MODEL_REVISION."""
    return ServedModelConfig(
        weights=WEIGHTS_PREFIX,
        api_model=SERVED_MODEL_NAME,
        # The staged prefix is an HF-format snapshot; streaming keeps 306 GiB off the
        # worker's local disk.
        object_store_load_mode=ObjectStoreLoadMode.STREAM,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=GPUS_PER_ENGINE,
    )


def broker() -> BrokerConfig:
    return BrokerConfig(
        worker=InferenceWorkerConfig(
            max_in_flight=WORKER_MAX_IN_FLIGHT,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS * 0.9,
        ),
        proxy=InferenceProxyConfig(
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            readiness_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_pending_requests=MAX_PENDING_REQUESTS,
        ),
        request_lease_timeout_seconds=REQUEST_TIMEOUT_SECONDS * 0.95,
        max_retries_failure=WORKER_MAX_RETRIES_FAILURE,
        max_retries_preemption=MAX_RETRIES_PREEMPTION,
    )


def worker_resources() -> ResourceConfig:
    """One engine: a single GB200x4 task on the pool cluster, preemption-free.

    Host RAM must budget the full checkpoint (the distributed streamer pipelines a
    ~76 GiB shard per TP process; marin's own sizing guidance budgets the whole
    checkpoint per task), so 256 GiB OOMKills during weight loading — 512 GiB
    clears it on the 768 GiB node.
    """
    return ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_ENGINE,
        cpu=16.0,
        ram="512g",
        disk="200g",
        preemptible=False,
        target_cluster=CLUSTER,
    )


def iris_config(worker_env_vars: tuple[tuple[str, str], ...] = ()) -> IrisConfig:
    # Startup patience: "placed" includes a K8s-Pending pod, so on a saturated
    # cluster the budget must cover queueing for preempted capacity plus the
    # engine boot (~30 min with in-region streaming). PRODUCTION preemption is
    # real but not instant (observed >30 min during the 16:30Z saturation).
    ready_timeout_seconds = 4 * 3600.0
    return IrisConfig(
        worker_resources=worker_resources(),
        # The default environment bundles the launching checkout's workspace, matching
        # `marin-serve iris` GPU serving.
        worker_environment=create_environment(env_vars=dict(worker_env_vars)),
        endpoint_ready_timeout_seconds=ready_timeout_seconds,
        endpoint_health_timeout_seconds=ready_timeout_seconds,
        max_retries_preemption=MAX_RETRIES_PREEMPTION,
        priority=PRIORITY,
    )


def service(
    instances: int,
    *,
    engine_extra_args: tuple[str, ...] = (),
    worker_env_vars: tuple[tuple[str, str], ...] = (),
    timeout_hours: float = COORDINATOR_TIMEOUT_HOURS,
) -> IrisServiceConfig:
    """The brokered serving service for `instances` engines behind one endpoint.

    `engine_extra_args` extends the engine arguments (e.g. a speculative-decoding
    variant measured by the warm run); it never carries topology flags.
    `worker_env_vars` adds environment variables to the engine workers (stability
    variants).
    """
    return IrisServiceConfig(
        model=model(),
        engine=engine(engine_extra_args),
        iris=iris_config(worker_env_vars),
        endpoint_name=ENDPOINT_NAME,
        instances=instances,
        broker=broker(),
        timeout_hours=timeout_hours,
    )


def coordinator_resources() -> ResourceConfig:
    """The coordinator job's CPU placement on the pool cluster (non-preemptible).

    The coordinator is the pool's data path: the controller proxy terminates here and
    every request streams through this process's dashboard proxy into the broker
    actor before reaching an engine. Size it for the full fleet's SSE throughput —
    2 CPUs starved the lease renewals under fleet load (transient "No endpoint")
    and 16 CPUs still throttled a 32-engine pool to ~1/8 of solo throughput
    (TTFT grew to minutes at 768 concurrent).
    """
    return ResourceConfig.with_cpu(
        cpu=32.0,
        ram="96g",
        disk="40g",
        preemptible=False,
        target_cluster=CLUSTER,
    )
