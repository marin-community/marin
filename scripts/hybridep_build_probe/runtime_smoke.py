# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise fused HybridEP dispatch/combine on an Iris multi-task GB200 group."""

import atexit
import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)
from iris.runtime.jax_init import _poll_for_coordinator


def _rank_info(job_info) -> tuple[int, int, int]:
    if IRIS_MULTIGPU_PROCESS_COUNT_ENV in os.environ:
        device_ids = os.environ[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV].split(",")
        if len(device_ids) != 1:
            raise ValueError(f"HybridEP expects one device per process, got {device_ids}")
        return (
            int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV]),
            int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV]),
            int(device_ids[0]),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return (
        job_info.task_index * local_world_size + local_rank,
        job_info.num_tasks * local_world_size,
        local_rank,
    )


def _initialize_process_group() -> None:
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("HybridEP runtime smoke must run inside an Iris job")
    rank, world_size, device_index = _rank_info(job_info)
    endpoint_name = f"hybridep-torch-{job_info.job_id.to_safe_token()}-attempt-{job_info.attempt_id}"
    port = job_info.ports.get("jax", 8476)
    address = f"{job_info.advertise_host}:{port}"
    if rank == 0:
        endpoint_id = iris_ctx().registry.register(endpoint_name, address)
        atexit.register(iris_ctx().registry.unregister, endpoint_id)
    else:
        address = _poll_for_coordinator(
            iris_ctx().resolver,
            endpoint_name,
            timeout=600,
            poll_interval=1,
        )
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{address}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{device_index}"),
    )
    torch.cuda.set_device(device_index)


def _elapsed(callable_, *, iterations: int = 10) -> float:
    for _ in range(3):
        callable_()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) / 1e3 / iterations
    times = [torch.zeros((), device="cuda", dtype=torch.float64) for _ in range(dist.get_world_size())]
    dist.all_gather(times, torch.asarray(elapsed, device="cuda", dtype=torch.float64))
    return max(float(value) for value in times)


def main() -> None:
    source_root = Path(os.environ["HYBRID_EP_SOURCE"]).resolve()
    sys.path.insert(0, str(source_root))
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("HybridEP runtime smoke must run inside an Iris job")
    global_rank, _, _ = _rank_info(job_info)
    run_id = os.environ.get("WANDB_RUN_ID")
    run = None
    if global_rank == 0 and run_id:
        run = wandb.init(
            entity="marin-community",
            project="rav_moe",
            id=run_id,
            name=run_id,
        )
        run.log({"hybridep/stage": 0})
    _initialize_process_group()
    if run is not None:
        run.log({"hybridep/stage": 1})

    # HybridEP has a standalone build, but deep_ep/__init__.py imports the
    # separately built DeepEP extension. Load the HybridEP module as a package
    # submodule without executing that unrelated package initializer.
    package = types.ModuleType("deep_ep")
    package.__path__ = [str(source_root / "deep_ep")]
    sys.modules["deep_ep"] = package
    HybridEPBuffer = importlib.import_module("deep_ep.hybrid_ep_buffer").HybridEPBuffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hidden_dim = int(os.environ.get("HYBRID_EP_HIDDEN", "5120"))
    tokens = int(os.environ.get("HYBRID_EP_TOKENS", "65536"))
    topk = int(os.environ.get("HYBRID_EP_TOPK", "8"))
    num_experts = int(os.environ.get("HYBRID_EP_EXPERTS", "256"))
    if num_experts % world_size != 0:
        raise ValueError(f"experts={num_experts} must be divisible by world_size={world_size}")

    buffer = HybridEPBuffer(
        group=dist.group.WORLD,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=tokens,
        num_local_experts=num_experts // world_size,
        use_fp8=False,
        num_sms_dispatch_api=32,
        num_sms_combine_api=32,
        use_shared_buffer=True,
        enable_custom_allgather=True,
    )
    if run is not None:
        run.log({"hybridep/stage": 2})
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1025 + rank)
    hidden = torch.randn(
        (tokens, hidden_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    scores = torch.rand(
        (tokens, num_experts),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    topk_indices = torch.topk(scores, topk, dim=-1, sorted=False).indices
    routing_map = torch.zeros((tokens, num_experts), device="cuda", dtype=torch.bool)
    routing_map.scatter_(1, topk_indices, True)

    dispatched, _, _, tokens_per_expert, handle = buffer.dispatch_with_permute(
        hidden=hidden,
        routing_map=routing_map,
        pad_multiple=32,
        fuse_permute_dispatch=True,
    )
    if run is not None:
        run.log({"hybridep/stage": 3})
    combined, _ = buffer.combine_with_unpermute(
        hidden=dispatched,
        handle=handle,
        pad_multiple=32,
        fuse_unpermute_combine=True,
    )
    torch.testing.assert_close(combined / topk, hidden, rtol=1e-2, atol=2e-2)
    if run is not None:
        run.log({"hybridep/stage": 4})

    num_permuted_tokens = int(tokens_per_expert.sum().item())
    dispatch_args = {
        "hidden": hidden,
        "routing_map": routing_map,
        "handle": handle,
        "num_permuted_tokens": num_permuted_tokens,
        "pad_multiple": 32,
        "fuse_permute_dispatch": True,
    }
    combine_args = {
        "hidden": dispatched,
        "handle": handle,
        "pad_multiple": 32,
        "fuse_unpermute_combine": True,
    }
    dispatch_time = _elapsed(lambda: buffer.dispatch_with_permute(**dispatch_args))
    combine_time = _elapsed(lambda: buffer.combine_with_unpermute(**combine_args))
    payload_bytes = dispatched.numel() * dispatched.element_size()
    if rank == 0:
        metrics = {
            "hybridep/ranks": world_size,
            "hybridep/tokens": tokens,
            "hybridep/hidden_dim": hidden_dim,
            "hybridep/num_experts": num_experts,
            "hybridep/permuted_tokens": num_permuted_tokens,
            "hybridep/dispatch_ms": dispatch_time * 1e3,
            "hybridep/combine_ms": combine_time * 1e3,
            "hybridep/dispatch_gbps": payload_bytes / dispatch_time / 1e9,
            "hybridep/combine_gbps": payload_bytes / combine_time / 1e9,
        }
        print(
            "HYBRID_EP_RUNTIME_PASS "
            f"ranks={world_size} tokens={tokens} hidden={hidden_dim} experts={num_experts} "
            f"permuted={num_permuted_tokens} dispatch_ms={dispatch_time * 1e3:.3f} "
            f"combine_ms={combine_time * 1e3:.3f} "
            f"dispatch_gbps={payload_bytes / dispatch_time / 1e9:.2f} "
            f"combine_gbps={payload_bytes / combine_time / 1e9:.2f}",
            flush=True,
        )
        if run is not None:
            run.config.update(metrics)
            run.log(metrics)
            run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
