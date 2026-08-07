# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep official Mixture-of-Kittens BF16 forward schedules on one GB200 tray."""

import argparse
import itertools
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from benchmark_metadata import (
    canonical_json_sha256,
    command_record,
    framed_tensor_sha256,
    nvidia_smi_snapshot,
    toolchain_snapshot,
)
from benchmarks.utils import get_num_local_experts, get_tflops, init_distributed
from gb200_mok_gmm_probe import _validate_checkout
from mok import functional
from tests.utils import generate_inputs

NUM_LOCAL_TOKENS = 2048
HIDDEN_DIMENSION = 7168
INTERMEDIATE_DIMENSION = 3072
NUM_EXPERTS = 384
TOP_K = 6
WARMUPS = 100
REPEATS = 50


def _comma_separated_integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _rank_max_samples(samples: list[float], device: torch.device) -> tuple[list[float], list[list[float]]]:
    local = torch.tensor(samples, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    per_rank = [rank_samples.cpu().tolist() for rank_samples in gathered]
    return torch.stack(gathered).max(dim=0).values.cpu().tolist(), per_rank


def _measure(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize(device)
    dist.barrier()
    rank_maxima, per_rank = _rank_max_samples([start.elapsed_time(end) for start, end in events], device)
    return {
        "rank_max_samples_ms": rank_maxima,
        "per_rank_samples_ms": per_rank,
        "median_ms": statistics.median(rank_maxima),
        "mean_ms": statistics.fmean(rank_maxima),
        "minimum_ms": min(rank_maxima),
        "maximum_ms": max(rank_maxima),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.bfloat16:
        payload = contiguous.view(torch.uint16).numpy().tobytes(order="C")
    else:
        payload = contiguous.numpy().tobytes(order="C")
    return framed_tensor_sha256(str(contiguous.dtype), tuple(contiguous.shape), payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-sms", type=_comma_separated_integers, default=(8, 16, 24, 32, 40, 48))
    parser.add_argument("--minibatches", type=_comma_separated_integers, default=(2048, 4096, 8192))
    parser.add_argument("--mok-root", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--clock-policy", default="cluster_default_unpinned")
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    source = _validate_checkout(args.mok_root.resolve())
    rank, world_size, device = init_distributed()
    local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(
        rank,
        device,
        NUM_EXPERTS,
        local_experts,
        TOP_K,
        NUM_LOCAL_TOKENS,
        HIDDEN_DIMENSION,
        INTERMEDIATE_DIMENSION,
    )
    x, topk_experts, router_weights, shared_gate, shared_up, shared_down = inputs[:6]
    routed_gate, routed_up, routed_down = inputs[6:9]
    candidates = []
    telemetry_initial = nvidia_smi_snapshot() if rank == 0 else None
    for communication_sms, minibatch_size in itertools.product(args.communication_sms, args.minibatches):
        config = functional.MoKConfig(
            fwd_num_comm_sms=communication_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=32 * minibatch_size,
        )
        workspace = functional.get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=NUM_LOCAL_TOKENS,
            hidden_size=HIDDEN_DIMENSION,
            topk=TOP_K,
        )

        def run_forward(config=config, workspace=workspace):
            schedule = functional.build_schedule(
                workspace,
                config,
                topk_experts,
                num_local_experts=local_experts,
            )
            return functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )[0]

        telemetry_before = nvidia_smi_snapshot() if rank == 0 else None
        timing = _measure(run_forward, device, warmups=args.warmups, repeats=args.repeats)
        telemetry_after = nvidia_smi_snapshot() if rank == 0 else None
        output = run_forward()
        torch.cuda.synchronize(device)
        output_hashes: list[str | None] = [None] * world_size
        dist.all_gather_object(output_hashes, _tensor_sha256(output))
        if rank == 0:
            latency = timing["median_ms"]
            throughput = get_tflops(
                latency,
                NUM_LOCAL_TOKENS,
                TOP_K,
                HIDDEN_DIMENSION,
                INTERMEDIATE_DIMENSION,
            )
            print(
                f"comm_sms={communication_sms} minibatch={minibatch_size} "
                f"latency_ms={latency:.4f} tflops={throughput:.1f}"
            )
            candidate = {
                "communication_sms": communication_sms,
                "minibatch_size": minibatch_size,
                "macrobatch_size": 32 * minibatch_size,
            }
            candidates.append(
                {
                    "candidate": {**candidate, "fingerprint_sha256": canonical_json_sha256(candidate)},
                    "timing": timing,
                    "tflops_at_median": throughput,
                    "per_rank_output_sha256": output_hashes,
                    "gpu_telemetry": {"before": telemetry_before, "after": telemetry_after},
                }
            )
    if rank == 0:
        result = {
            "schema_version": 2,
            "benchmark": "official_mok_bf16_forward_schedule_sweep",
            "status": "ok",
            "source": {"shuttle_revision": args.shuttle_revision, **source},
            "shape": {
                "world_size": world_size,
                "local_tokens": NUM_LOCAL_TOKENS,
                "global_experts": NUM_EXPERTS,
                "local_experts": local_experts,
                "top_k": TOP_K,
                "hidden_size": HIDDEN_DIMENSION,
                "intermediate_size": INTERMEDIATE_DIMENSION,
                "dtype": "bfloat16",
            },
            "protocol": {"warmups": args.warmups, "repeats": args.repeats, "selection_metric": "median rank-max ms"},
            "candidates": candidates,
            "environment": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "clock_policy": args.clock_policy,
                "command": command_record(),
                "toolchain": toolchain_snapshot("nvcc"),
                "gpu_telemetry": {"initial": telemetry_initial, "final": nvidia_smi_snapshot()},
            },
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
