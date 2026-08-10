# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the process-per-GPU Mixture-of-Kittens hero-layer oracle."""

import argparse
import dataclasses
import json
import os
import statistics

import torch
import torch.distributed as dist
from mok import functional
from tests.utils import generate_inputs

NUM_LOCAL_TOKENS = 65_536
HIDDEN_SIZE = 6_144
LOGICAL_INTERMEDIATE_SIZE = 6_272
PHYSICAL_INTERMEDIATE_SIZE = 6_400
TOP_K = 4
NUM_LOCAL_EXPERTS = 3
NUM_GLOBAL_EXPERTS = 12
WARMUP_RUNS = 2
TIMED_RUNS = 5


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleConfig:
    name: str
    forward_communication_sms: int
    backward_communication_sms: int
    minibatch_size: int
    macrobatch_size: int

    def mok_config(self) -> functional.MoKConfig:
        return functional.MoKConfig(
            fwd_num_comm_sms=self.forward_communication_sms,
            bwd_num_comm_sms=self.backward_communication_sms,
            minibatch_size=self.minibatch_size,
            macrobatch_size=self.macrobatch_size,
        )


SCHEDULE_CONFIGS = (
    ScheduleConfig("upstream_default", 40, 28, 4_096, 131_072),
    ScheduleConfig("prior_small_shape_tuning", 20, 28, 2_048, 65_536),
    ScheduleConfig("large_macro_20_28", 20, 28, 8_192, 262_144),
    ScheduleConfig("large_macro_40_28", 40, 28, 8_192, 262_144),
    ScheduleConfig("large_macro_40_40", 40, 40, 8_192, 262_144),
    ScheduleConfig("large_minibatch_40_40", 40, 40, 16_384, 262_144),
)


def _timed_iteration(
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[float, float, torch.Tensor, tuple[torch.Tensor, ...]]:
    (
        x,
        topk_experts,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        grad_output,
    ) = inputs
    start_forward = torch.cuda.Event(enable_timing=True)
    end_forward = torch.cuda.Event(enable_timing=True)
    start_backward = torch.cuda.Event(enable_timing=True)
    end_backward = torch.cuda.Event(enable_timing=True)

    start_forward.record()
    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=NUM_LOCAL_EXPERTS,
    )
    output, context = functional.forward(
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
    )
    end_forward.record()

    start_backward.record()
    gradients = functional.backward(
        config,
        workspace,
        schedule,
        context,
        grad_output,
        x,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    end_backward.record()
    torch.cuda.synchronize()
    return (
        start_forward.elapsed_time(end_forward),
        start_backward.elapsed_time(end_backward),
        output,
        gradients,
    )


def _rank_max_times(forward_time: float, backward_time: float, device: torch.device) -> tuple[float, float]:
    times = torch.tensor([forward_time, backward_time], dtype=torch.float64, device=device)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    return float(times[0].item()), float(times[1].item())


def _benchmark_config(
    schedule_config: ScheduleConfig,
    workspace: functional.MoKWorkspace,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor, tuple[torch.Tensor, ...]]:
    config = schedule_config.mok_config()
    for _ in range(WARMUP_RUNS):
        _timed_iteration(config, workspace, inputs)

    forward_samples = []
    backward_samples = []
    output = None
    gradients = None
    for _ in range(TIMED_RUNS):
        forward_time, backward_time, output, gradients = _timed_iteration(config, workspace, inputs)
        forward_time, backward_time = _rank_max_times(forward_time, backward_time, device)
        forward_samples.append(forward_time)
        backward_samples.append(backward_time)

    assert output is not None
    assert gradients is not None
    combined_samples = [forward + backward for forward, backward in zip(forward_samples, backward_samples, strict=True)]
    result = {
        **dataclasses.asdict(schedule_config),
        "forward_samples_ms": forward_samples,
        "backward_samples_ms": backward_samples,
        "combined_samples_ms": combined_samples,
        "median_forward_ms": statistics.median(forward_samples),
        "median_backward_ms": statistics.median(backward_samples),
        "median_combined_ms": statistics.median(combined_samples),
    }
    return result, output, gradients


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    reduced = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    return bool(reduced.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    inputs = generate_inputs(
        rank,
        device,
        NUM_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS,
        TOP_K,
        NUM_LOCAL_TOKENS,
        HIDDEN_SIZE,
        PHYSICAL_INTERMEDIATE_SIZE,
    )
    workspace = functional.get_workspace(
        SCHEDULE_CONFIGS[0].mok_config(),
        dist.group.WORLD,
        device=device,
        num_local_tokens=NUM_LOCAL_TOKENS,
        hidden_size=HIDDEN_SIZE,
        topk=TOP_K,
    )

    config_results = []
    finite = torch.tensor(True, dtype=torch.bool, device=device)
    first_output = None
    for schedule_config in SCHEDULE_CONFIGS:
        result, output, gradients = _benchmark_config(schedule_config, workspace, inputs, device)
        config_results.append(result)
        finite = finite & torch.isfinite(output).all()
        for gradient in gradients:
            finite = finite & torch.isfinite(gradient).all()
        if first_output is None:
            first_output = output.clone()

    assert first_output is not None
    _, _, repeat_output, _ = _timed_iteration(SCHEDULE_CONFIGS[0].mok_config(), workspace, inputs)
    repeat_deterministic = _all_ranks_true(torch.equal(first_output, repeat_output), device)
    finite_on_all_ranks = _all_ranks_true(bool(finite.item()), device)

    result = {
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "world_size": dist.get_world_size(),
            "processes_per_gpu": 1,
        },
        "shape": {
            "num_local_tokens": NUM_LOCAL_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "logical_intermediate_size": LOGICAL_INTERMEDIATE_SIZE,
            "physical_intermediate_size": PHYSICAL_INTERMEDIATE_SIZE,
            "top_k": TOP_K,
            "num_local_experts": NUM_LOCAL_EXPERTS,
            "num_global_experts": NUM_GLOBAL_EXPERTS,
            "includes_shared_expert": True,
        },
        "finite_on_all_ranks": finite_on_all_ranks,
        "repeat_deterministic_on_all_ranks": repeat_deterministic,
        "configs": config_results,
    }
    if rank == 0:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("mok_hero_benchmark", json.dumps(result, separators=(",", ":")), flush=True)

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
