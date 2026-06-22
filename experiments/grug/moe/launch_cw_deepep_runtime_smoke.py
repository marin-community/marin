# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave launcher for a one-GPU-per-process DeepEP internode runtime smoke."""

from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax._src.distributed as jax_distributed
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.runtime.jax_init import initialize_jax
from levanter.kernels.deepep.availability import DEEPEP_CUDA_ARCH_ENV, DEEPEP_KNOWN_GOOD_COMMIT, DEEPEP_SRC_ENV
from levanter.kernels.deepep.transport_ffi import (
    current_internode_process_topology,
    deepep_combine_internode,
    deepep_dispatch_internode,
    ensure_internode_runtime,
    internode_runtime_status,
    run_internode_mapped_counter_smoke,
    shutdown_internode_runtime,
)
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep
from marin.training.training import resolve_training_env
from rigging.timing import Duration, ExponentialBackoff

from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe.cw_storage import set_default_cw_grug_moe_prefix

set_default_cw_grug_moe_prefix()

DEFAULT_OUTPUT_SUBDIR = "experiments/grug-moe-cw/deepep-runtime-smoke"
DEEPEP_SOURCE_URL = "https://github.com/deepseek-ai/DeepEP.git"
DEFAULT_TASK_IMAGE = "ghcr.io/marin-community/iris-task-cuda-devel:969c0e3"
DEFAULT_JAX_COORDINATOR_ENDPOINT = "deepep_jax_coordinator"
DEFAULT_JAX_COORDINATOR_PORT = 8476


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class DeepEPRuntimeSmokeConfig:
    run_id: str
    resources: ResourceConfig
    num_nvl_bytes: int
    num_rdma_bytes: int
    processes_per_task: int


def _env_snapshot() -> dict[str, str]:
    keys = (
        "IRIS_TASK_ID",
        "IRIS_NUM_TASKS",
        "NCCL_SOCKET_IFNAME",
        "NCCL_SOCKET_FAMILY",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "XLA_FLAGS",
        "CUDA_VISIBLE_DEVICES",
        "DEEPEP_RANKS_PER_NODE",
        "DEEPEP_SRC_ROOT",
        "DEEPEP_CUDA_ARCH",
        "DEEPEP_NVSHMEM_HCA_PREFIX",
        "DEEPEP_RUNTIME_COORDINATOR_ADDRESS",
        "DEEPEP_RUNTIME_PROCESS_COUNT",
        "DEEPEP_RUNTIME_PROCESS_INDEX",
        "DEEPEP_RUNTIME_LOCAL_RANK",
        "DEEPEP_RUNTIME_PROCESSES_PER_TASK",
        "DEEPEP_RUNTIME_RUN_DISPATCH_SMOKE",
        "DEEPEP_RUNTIME_RUN_COMBINE_SMOKE",
        "DEEPEP_RUNTIME_RUN_BACKWARD_SMOKE",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _coordinator_address_for_iris_task(*, endpoint_name: str, port: int, timeout: float = 300.0) -> str:
    job_info = get_job_info()
    if job_info is None:
        return f"127.0.0.1:{port}"
    ctx = iris_ctx()
    if job_info.task_index == 0:
        address = f"{job_info.advertise_host}:{port}"
        ctx.registry.register(endpoint_name, address)
        return address

    result: list[str] = []

    def check_coordinator() -> bool:
        resolved = ctx.resolver.resolve(endpoint_name)
        if resolved.is_empty:
            return False
        result.append(resolved.first().url)
        return True

    backoff = ExponentialBackoff(initial=2.0, maximum=30.0)
    backoff.wait_until_or_raise(
        check_coordinator,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out after {timeout}s waiting for DeepEP JAX coordinator endpoint {endpoint_name!r}",
    )
    return result[0]


def _maybe_bootstrap_deepep_source() -> None:
    root = Path(os.environ.get(DEEPEP_SRC_ENV, "") or "/tmp/marin-deepep/DeepEP").expanduser()
    revision = os.environ.get("MAY_DEEPEP_REVISION") or os.environ.get("DEEPEP_REVISION", DEEPEP_KNOWN_GOOD_COMMIT)
    os.environ[DEEPEP_SRC_ENV] = str(root)
    os.environ.setdefault(DEEPEP_CUDA_ARCH_ENV, "sm_90")
    os.environ.setdefault("MARIN_DEEPEP_CACHE_DIR", "/tmp/marin-deepep-cache")

    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", DEEPEP_SOURCE_URL, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--force", revision], check=True)


def _maybe_install_rdma_headers() -> None:
    if Path("/usr/include/infiniband/mlx5dv.h").is_file():
        return
    if os.geteuid() != 0:
        raise RuntimeError("DeepEP internode smoke needs infiniband/mlx5dv.h but cannot apt-get as non-root")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends", "libibverbs-dev"],
        check=True,
        env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
    )


def _run_deepep_runtime_smoke_local(config: DeepEPRuntimeSmokeConfig) -> None:
    if config.processes_per_task > 1:
        _run_deepep_runtime_smoke_process_supervisor(config)
        return

    _maybe_install_rdma_headers()
    _maybe_bootstrap_deepep_source()
    print(f"deepep_runtime_smoke: hostname={socket.gethostname()} env={_env_snapshot()}", flush=True)
    start = time.time()
    initialize_jax()
    print(
        "deepep_runtime_smoke: jax initialized "
        f"process={jax.process_index()}/{jax.process_count()} "
        f"local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
        f"elapsed={time.time() - start:.3f}",
        flush=True,
    )
    topology = current_internode_process_topology()
    print(f"deepep_runtime_smoke: topology {topology}", flush=True)
    print(f"deepep_runtime_smoke: pre_status {internode_runtime_status()}", flush=True)

    client = jax_distributed.global_state.client
    if client is None:
        raise RuntimeError("DeepEP runtime smoke requires jax.distributed to be initialized")
    client.wait_at_barrier("deepep_process_per_gpu_runtime_smoke_pre_init", timeout_in_ms=300_000)

    ensure_internode_runtime(
        num_nvl_bytes=config.num_nvl_bytes,
        num_rdma_bytes=config.num_rdma_bytes,
        configure_nvshmem_env=True,
    )
    print(f"deepep_runtime_smoke: status {internode_runtime_status()}", flush=True)
    print(
        "deepep_runtime_smoke: counter " f"{json.dumps(run_internode_mapped_counter_smoke(), sort_keys=True)}",
        flush=True,
    )
    shutdown_internode_runtime()
    print("deepep_runtime_smoke: done", flush=True)


def _run_internode_dispatch_smoke(*, run_combine: bool) -> None:
    status = internode_runtime_status()
    if not status.initialized:
        raise RuntimeError("DeepEP dispatch smoke requires initialized internode runtime")
    if status.num_local_ranks <= 0 or status.num_global_ranks % status.num_local_ranks != 0:
        raise RuntimeError(f"DeepEP dispatch smoke got invalid runtime topology: {status}")

    process_index = jax.process_index()
    node_count = status.num_global_ranks // status.num_local_ranks
    node_rank = process_index // status.num_local_ranks
    local_rank = process_index % status.num_local_ranks
    if node_count <= 1:
        raise RuntimeError(f"DeepEP dispatch smoke requires at least two RDMA nodes, got {node_count}")

    num_tokens = env_int("DEEPEP_RUNTIME_DISPATCH_TOKENS", 8)
    hidden = env_int("DEEPEP_RUNTIME_DISPATCH_HIDDEN", 128)
    topk = env_int("DEEPEP_RUNTIME_DISPATCH_TOPK", 1)
    if num_tokens <= 0 or hidden <= 0 or topk <= 0:
        raise ValueError(f"Dispatch smoke dimensions must be positive, got {num_tokens=} {hidden=} {topk=}")
    if (hidden * jnp.dtype(jnp.bfloat16).itemsize) % 16 != 0:
        raise ValueError(f"Dispatch smoke hidden must make hidden*bfloat16 bytes divisible by int4, got {hidden}")
    hidden_int4 = hidden * jnp.dtype(jnp.bfloat16).itemsize // 16
    if run_combine and hidden_int4 % 32 != 0:
        raise ValueError(
            "DeepEP internode combine requires hidden_int4 % 32 == 0; " f"got {hidden_int4=} from {hidden=}"
        )

    num_ranks = status.num_global_ranks
    num_experts = num_ranks
    destination_node = (node_rank + 1) % node_count
    destination_ranks = [
        destination_node * status.num_local_ranks + ((local_rank + topk_index) % status.num_local_ranks)
        for topk_index in range(topk)
    ]

    x = jnp.full((num_tokens, hidden), process_index + 1, dtype=jnp.bfloat16)
    topk_idx = jnp.tile(jnp.asarray(destination_ranks, dtype=jnp.int32)[None, :], (num_tokens, 1))
    topk_weights = jnp.ones((num_tokens, topk), dtype=jnp.float32)

    tokens_per_rank = [0] * num_ranks
    for destination_rank in destination_ranks:
        tokens_per_rank[destination_rank] += num_tokens
    tokens_per_rdma_rank = [0] * node_count
    tokens_per_rdma_rank[destination_node] = num_tokens * topk
    tokens_per_expert = [0] * num_experts
    for destination_expert in destination_ranks:
        tokens_per_expert[destination_expert] += num_tokens
    token_in_rank = [[False] * num_ranks for _ in range(num_tokens)]
    for token_index in range(num_tokens):
        for destination_rank in destination_ranks:
            token_in_rank[token_index][destination_rank] = True

    max_recv_tokens = env_int("DEEPEP_RUNTIME_DISPATCH_MAX_RECV_TOKENS", num_tokens * topk * node_count)
    max_rdma_recv_tokens = env_int("DEEPEP_RUNTIME_DISPATCH_MAX_RDMA_RECV_TOKENS", max_recv_tokens)
    print(
        "deepep_runtime_smoke_worker: dispatch_smoke_begin "
        f"rank={process_index} node={node_rank}/{node_count} local_rank={local_rank} "
        f"destination_ranks={destination_ranks} {num_tokens=} {hidden=} {hidden_int4=} {topk=} "
        f"{max_recv_tokens=} {max_rdma_recv_tokens=}",
        flush=True,
    )
    dispatch = deepep_dispatch_internode(
        x,
        topk_idx,
        topk_weights,
        jnp.asarray(tokens_per_rank, dtype=jnp.int32),
        jnp.asarray(tokens_per_rdma_rank, dtype=jnp.int32),
        jnp.asarray(tokens_per_expert, dtype=jnp.int32),
        jnp.asarray(token_in_rank, dtype=jnp.bool_),
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_rdma_recv_tokens=max_rdma_recv_tokens,
        num_local_ranks=status.num_local_ranks,
    )
    jax.block_until_ready(dispatch)
    dispatch_counts = {
        "num_recv_tokens": jax.device_get(dispatch.num_recv_tokens).tolist(),
        "num_recv_rdma_tokens": jax.device_get(dispatch.num_recv_rdma_tokens).tolist(),
        "local_expert_counts": jax.device_get(dispatch.local_expert_counts).tolist(),
    }
    print(
        "deepep_runtime_smoke_worker: dispatch_smoke_done " f"{json.dumps(dispatch_counts, sort_keys=True)}",
        flush=True,
    )

    if not run_combine:
        return

    print("deepep_runtime_smoke_worker: combine_smoke_begin", flush=True)
    combined_x, combined_weights = deepep_combine_internode(
        dispatch.recv_x,
        dispatch.recv_topk_weights,
        dispatch.is_token_in_rank,
        dispatch.recv_src_meta,
        dispatch.rdma_channel_prefix_matrix,
        dispatch.recv_rdma_channel_prefix_matrix,
        dispatch.recv_rdma_rank_prefix_sum,
        dispatch.gbl_channel_prefix_matrix,
        dispatch.recv_gbl_channel_prefix_matrix,
        dispatch.recv_gbl_rank_prefix_sum,
        dispatch.send_rdma_head,
        dispatch.send_nvl_head,
        dispatch.num_recv_tokens,
        dispatch.num_recv_rdma_tokens,
    )
    jax.block_until_ready((combined_x, combined_weights))
    combine_summary = {
        "combined_x_shape": tuple(combined_x.shape),
        "combined_weights_shape": tuple(combined_weights.shape),
        "combined_x_sum": float(jax.device_get(jnp.sum(combined_x.astype(jnp.float32)))),
        "combined_weights_sum": float(jax.device_get(jnp.sum(combined_weights))),
    }
    print(
        "deepep_runtime_smoke_worker: combine_smoke_done " f"{json.dumps(combine_summary, sort_keys=True)}",
        flush=True,
    )

    if env_bool("DEEPEP_RUNTIME_RUN_BACKWARD_SMOKE"):
        print("deepep_runtime_smoke_worker: backward_smoke_begin", flush=True)

        def combine_loss(recv_x: jax.Array) -> jax.Array:
            combined_x, _ = deepep_combine_internode(
                recv_x,
                dispatch.recv_topk_weights,
                dispatch.is_token_in_rank,
                dispatch.recv_src_meta,
                dispatch.rdma_channel_prefix_matrix,
                dispatch.recv_rdma_channel_prefix_matrix,
                dispatch.recv_rdma_rank_prefix_sum,
                dispatch.gbl_channel_prefix_matrix,
                dispatch.recv_gbl_channel_prefix_matrix,
                dispatch.recv_gbl_rank_prefix_sum,
                dispatch.send_rdma_head,
                dispatch.send_nvl_head,
                dispatch.num_recv_tokens,
                dispatch.num_recv_rdma_tokens,
            )
            return jnp.sum(combined_x.astype(jnp.float32))

        backward_loss, grad_recv_x = jax.value_and_grad(combine_loss)(dispatch.recv_x)
        jax.block_until_ready((backward_loss, grad_recv_x))
        backward_summary = {
            "loss": float(jax.device_get(backward_loss)),
            "grad_recv_x_shape": tuple(grad_recv_x.shape),
            "grad_recv_x_sum": float(jax.device_get(jnp.sum(grad_recv_x.astype(jnp.float32)))),
        }
        print(
            "deepep_runtime_smoke_worker: backward_smoke_done " f"{json.dumps(backward_summary, sort_keys=True)}",
            flush=True,
        )


def _run_deepep_runtime_smoke_subprocess_from_env() -> None:
    if not Path("/usr/include/infiniband/mlx5dv.h").is_file():
        raise RuntimeError("DeepEP worker subprocess missing infiniband/mlx5dv.h after supervisor setup")
    source_root = Path(os.environ.get(DEEPEP_SRC_ENV, "") or "/tmp/marin-deepep/DeepEP").expanduser()
    if not source_root.exists():
        raise RuntimeError(f"DeepEP worker subprocess missing source root after supervisor setup: {source_root}")
    os.environ[DEEPEP_SRC_ENV] = str(source_root)
    os.environ.setdefault(DEEPEP_CUDA_ARCH_ENV, "sm_90")
    os.environ.setdefault("MARIN_DEEPEP_CACHE_DIR", "/tmp/marin-deepep-cache")
    coordinator_address = os.environ["DEEPEP_RUNTIME_COORDINATOR_ADDRESS"]
    process_count = int(os.environ["DEEPEP_RUNTIME_PROCESS_COUNT"])
    process_index = int(os.environ["DEEPEP_RUNTIME_PROCESS_INDEX"])
    local_rank = int(os.environ["DEEPEP_RUNTIME_LOCAL_RANK"])
    num_nvl_bytes = env_int("DEEPEP_RUNTIME_NVL_BYTES", 256 * 1024 * 1024)
    num_rdma_bytes = env_int("DEEPEP_RUNTIME_RDMA_BYTES", 256 * 1024 * 1024)

    print(f"deepep_runtime_smoke_worker: hostname={socket.gethostname()} env={_env_snapshot()}", flush=True)
    start = time.time()
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=process_count,
        process_id=process_index,
        local_device_ids=0,
        coordinator_bind_address=f"0.0.0.0:{DEFAULT_JAX_COORDINATOR_PORT}" if process_index == 0 else None,
    )
    print(
        "deepep_runtime_smoke_worker: jax initialized "
        f"process={jax.process_index()}/{jax.process_count()} "
        f"local_rank={local_rank} local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
        f"elapsed={time.time() - start:.3f}",
        flush=True,
    )
    topology = current_internode_process_topology()
    print(f"deepep_runtime_smoke_worker: topology {topology}", flush=True)
    print(f"deepep_runtime_smoke_worker: pre_status {internode_runtime_status()}", flush=True)

    client = jax_distributed.global_state.client
    if client is None:
        raise RuntimeError("DeepEP runtime smoke requires jax.distributed to be initialized")
    client.wait_at_barrier("deepep_process_per_gpu_runtime_smoke_pre_init", timeout_in_ms=300_000)

    ensure_internode_runtime(
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        configure_nvshmem_env=True,
    )
    print(f"deepep_runtime_smoke_worker: status {internode_runtime_status()}", flush=True)
    print(
        "deepep_runtime_smoke_worker: counter " f"{json.dumps(run_internode_mapped_counter_smoke(), sort_keys=True)}",
        flush=True,
    )
    if env_bool("DEEPEP_RUNTIME_RUN_DISPATCH_SMOKE"):
        client.wait_at_barrier("deepep_process_per_gpu_runtime_smoke_pre_dispatch", timeout_in_ms=300_000)
        _run_internode_dispatch_smoke(run_combine=env_bool("DEEPEP_RUNTIME_RUN_COMBINE_SMOKE"))
    shutdown_internode_runtime()
    print("deepep_runtime_smoke_worker: done", flush=True)


def _run_deepep_runtime_smoke_process_supervisor(config: DeepEPRuntimeSmokeConfig) -> None:
    _maybe_install_rdma_headers()
    _maybe_bootstrap_deepep_source()
    print(f"deepep_runtime_smoke_supervisor: prebuilding transport FFI status={internode_runtime_status()}", flush=True)

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("DeepEP process supervisor must run inside an Iris task")
    if config.processes_per_task <= 1:
        raise ValueError("processes_per_task must be > 1 for supervisor mode")
    if job_info.num_tasks <= 1:
        raise ValueError("process supervisor mode needs at least two Iris tasks for internode smoke")

    coordinator_address = _coordinator_address_for_iris_task(
        endpoint_name=os.environ.get(
            "DEEPEP_RUNTIME_COORDINATOR_ENDPOINT",
            f"{DEFAULT_JAX_COORDINATOR_ENDPOINT}_{config.run_id}",
        ),
        port=DEFAULT_JAX_COORDINATOR_PORT,
    )
    process_count = job_info.num_tasks * config.processes_per_task
    base_process_index = job_info.task_index * config.processes_per_task
    print(
        "deepep_runtime_smoke_supervisor: "
        f"task={job_info.task_index}/{job_info.num_tasks} hostname={socket.gethostname()} "
        f"coordinator={coordinator_address} process_count={process_count} "
        f"processes_per_task={config.processes_per_task}",
        flush=True,
    )

    procs: list[subprocess.Popen[bytes]] = []
    for local_rank in range(config.processes_per_task):
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(local_rank),
            "DEEPEP_RANKS_PER_NODE": str(config.processes_per_task),
            "DEEPEP_RUNTIME_LOCAL_WORKER": "1",
            "DEEPEP_RUNTIME_COORDINATOR_ADDRESS": coordinator_address,
            "DEEPEP_RUNTIME_PROCESS_COUNT": str(process_count),
            "DEEPEP_RUNTIME_PROCESS_INDEX": str(base_process_index + local_rank),
            "DEEPEP_RUNTIME_LOCAL_RANK": str(local_rank),
            "DEEPEP_RUNTIME_NVL_BYTES": str(config.num_nvl_bytes),
            "DEEPEP_RUNTIME_RDMA_BYTES": str(config.num_rdma_bytes),
            "JAX_COORDINATOR_ADDRESS": coordinator_address,
            "JAX_PROCESS_COUNT": str(process_count),
            "JAX_PROCESS_INDEX": str(base_process_index + local_rank),
            "JAX_LOCAL_DEVICE_IDS": "0",
        }
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "experiments.grug.moe.launch_cw_deepep_runtime_smoke"],
                env=env,
            )
        )

    failures: list[tuple[int, int]] = []
    for local_rank, proc in enumerate(procs):
        return_code = proc.wait()
        if return_code != 0:
            failures.append((local_rank, return_code))
    if failures:
        raise RuntimeError(f"DeepEP local worker subprocesses failed: {failures}")
    print("deepep_runtime_smoke_supervisor: done", flush=True)


def run_deepep_runtime_smoke(config: DeepEPRuntimeSmokeConfig) -> None:
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=config.resources)
    request = JobRequest(
        name=f"grug-train-{config.run_id}",
        entrypoint=Entrypoint.from_callable(_run_deepep_runtime_smoke_local, args=[config]),
        resources=config.resources,
        environment=create_environment(env_vars=env_vars, extras=["gpu", "deepep"]),
        max_retries_failure=0,
    )
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)


def build_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "DEEPEP-RUNTIME-SMOKE-GPU1-RPN8-N2-%Y%m%d-%H%M%S"
    )
    resources = ResourceConfig.with_gpu(
        "H100",
        count=env_int("DEEPEP_RUNTIME_GPU_COUNT", 1),
        cpu=env_int("DEEPEP_RUNTIME_WORKER_CPU", 8),
        ram=os.environ.get("DEEPEP_RUNTIME_WORKER_RAM", "64g"),
        disk=os.environ.get("DEEPEP_RUNTIME_WORKER_DISK", "64g"),
        replicas=env_int("DEEPEP_RUNTIME_REPLICAS", 16),
        image=os.environ.get("DEEPEP_RUNTIME_TASK_IMAGE", DEFAULT_TASK_IMAGE),
    )
    return ExecutorStep(
        name=f"{DEFAULT_OUTPUT_SUBDIR}/{run_id}",
        fn=run_deepep_runtime_smoke,
        config=DeepEPRuntimeSmokeConfig(
            run_id=run_id,
            resources=resources,
            num_nvl_bytes=env_int("DEEPEP_RUNTIME_NVL_BYTES", 256 * 1024 * 1024),
            num_rdma_bytes=env_int("DEEPEP_RUNTIME_RDMA_BYTES", 256 * 1024 * 1024),
            processes_per_task=env_int("DEEPEP_RUNTIME_PROCESSES_PER_TASK", 1),
        ),
    )


def main() -> None:
    if os.environ.get("DEEPEP_RUNTIME_LOCAL_WORKER") == "1":
        _run_deepep_runtime_smoke_subprocess_from_env()
        return
    if os.environ.get("DEEPEP_RUNTIME_DIRECT_CHILD") == "1":
        _run_deepep_runtime_smoke_local(build_step().config)
        return
    executor_main(steps=[build_step()])


if __name__ == "__main__":
    main()
