# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time

import torch
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType


def _step(model: MoE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    output, aux_loss = model(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    loss = output.float().square().mean() + aux_loss.float()
    loss.backward()
    return output, loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark upstream SonicMoE forward and backward on one GPU.")
    parser.add_argument("--tokens", type=int, default=16_384)
    parser.add_argument("--hidden-dim", type=int, default=2_560)
    parser.add_argument("--intermediate-dim", type=int, default=1_280)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SonicMoE benchmark requires an NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = MoE(
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        hidden_size=args.hidden_dim,
        intermediate_size=args.intermediate_dim,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(args.tokens, args.hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    for _ in range(args.warmup):
        _step(model, x)
    torch.cuda.synchronize()

    durations = []
    output = None
    loss = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        output, loss = _step(model, x)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    assert output is not None and loss is not None
    durations.sort()
    result = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "tokens": args.tokens,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "experts": args.experts,
        "top_k": args.top_k,
        "iterations": args.iterations,
        "mean_duration": sum(durations) / len(durations),
        "median_duration": durations[len(durations) // 2],
        "min_duration": durations[0],
        "max_duration": durations[-1],
        "loss": float(loss.detach()),
        "output_abs_mean": float(output.detach().abs().float().mean()),
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
