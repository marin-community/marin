# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time

import torch

from .layers import DeltaRuleMixer, FastDeltaRuleMixer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive vs FLA GDN mixers.")
    parser.add_argument("--impl", choices=("naive", "fla"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_for_precision(args.precision)
    mixer_cls = FastDeltaRuleMixer if args.impl == "fla" else DeltaRuleMixer
    mixer = mixer_cls(args.d_model, args.n_heads).to(device=device, dtype=dtype)
    mixer.train()

    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device, dtype=dtype, requires_grad=True)

    def step() -> None:
        mixer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        y = mixer(x)
        loss = y.float().square().mean()
        if args.backward:
            loss.backward()

    for _ in range(args.warmup_iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.measure_iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = args.batch_size * args.seq_len * args.measure_iters
    metrics = {
        "impl": args.impl,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "measure_iters": args.measure_iters,
        "backward": args.backward,
        "elapsed_s": elapsed,
        "tokens_per_second": tokens / elapsed,
    }
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
