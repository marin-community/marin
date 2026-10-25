# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

from skyrl_train.distributed.megatron.stable_softmax import token_entropy, token_logprobs

VOCAB_SIZE = 151936


@torch.compile(dynamic=True)
def _mul_reduce(a, b):
    return (a * b).sum(dim=-1, keepdim=True)


class BaselineEntropy(torch.autograd.Function):
    """Entropy arithmetic at marin 2cae063c877172844fe6ff90e94354cc08a273fa."""

    @staticmethod
    def forward(ctx, logits):
        maximum = logits.max(dim=-1, keepdim=True).values
        exponentials = (logits - maximum).exp_()
        denominator = exponentials.sum(dim=-1, keepdim=True)
        probabilities = exponentials.div_(denominator)
        moment = _mul_reduce(probabilities, logits)
        ctx.save_for_backward(logits, probabilities, moment)
        return (maximum + denominator.log() - moment).squeeze(-1)

    @staticmethod
    def backward(ctx, upstream):
        logits, probabilities, moment = ctx.saved_tensors
        gradient = logits.sub(moment)
        gradient.mul_(probabilities)
        gradient.mul_(upstream.unsqueeze(-1))
        return gradient.neg_()


def _operation(logits, targets, stable, kind, grad, chunk_size):
    with torch.set_grad_enabled(grad):
        outputs = []
        if kind in ("logprob", "both", "training"):
            if stable:
                logprobs = token_logprobs(logits, targets, 0, logits.shape[-1], dist.group.WORLD, chunk_size=chunk_size)
            elif grad or logits.shape[1] <= chunk_size:
                logprobs = -parallel_cross_entropy(
                    logits, targets, dist_process_group=dist.group.WORLD, overwrite_input=False
                )
            else:
                logprobs = torch.cat(
                    [
                        -parallel_cross_entropy(x, y, dist_process_group=dist.group.WORLD, overwrite_input=False)
                        for x, y in zip(logits.split(chunk_size, dim=1), targets.split(chunk_size, dim=1), strict=True)
                    ],
                    dim=1,
                )
            outputs.append(logprobs)
        if kind in ("entropy", "both", "training"):
            with torch.set_grad_enabled(grad and kind != "training"):
                if stable:
                    entropy = token_entropy(logits, dist.group.WORLD, chunk_size=chunk_size)
                elif logits.shape[1] <= chunk_size:
                    entropy = BaselineEntropy.apply(logits)
                else:
                    entropy = torch.cat([BaselineEntropy.apply(x) for x in logits.split(chunk_size, dim=-2)], dim=-1)
            outputs.append(entropy)
        if grad:
            loss = outputs[0].sum() if kind == "training" else sum(output.sum() for output in outputs)
            return outputs, torch.autograd.grad(loss, logits)[0]
        return outputs


def _measurement(logits, targets, stable, kind, grad, chunk_size):
    logits.requires_grad_(grad)
    for _ in range(3):
        _operation(logits, targets, stable, kind, grad, chunk_size)
    torch.cuda.synchronize()
    durations = []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _operation(logits, targets, stable, kind, grad, chunk_size)
        stop.record()
        stop.synchronize()
        durations.append(start.elapsed_time(stop))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = _operation(logits, targets, stable, kind, grad, chunk_size)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del result
    return {"median_ms": statistics.median(durations), "min_ms": min(durations), "peak_extra_bytes": peak}


def _varied_measurement(stable, kind, grad, chunk_size):
    shapes = [(16, seq) for seq in (32, 40, 42, 48, 64, 80, 87, 90, 96, 128)]
    inputs = [
        (
            torch.randn((batch, seq, VOCAB_SIZE), dtype=torch.bfloat16, device="cuda", requires_grad=grad),
            torch.zeros((batch, seq), dtype=torch.long, device="cuda"),
        )
        for batch, seq in shapes
    ]

    def cycle():
        for logits, targets in inputs:
            _operation(logits, targets, stable, kind, grad, chunk_size)

    torch.cuda.synchronize()
    started = time.perf_counter()
    cycle()
    torch.cuda.synchronize()
    cold_ms = 1000 * (time.perf_counter() - started)
    binaries = {}
    for binary in Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.cubin"):
        binaries[binary.stem] = binaries.get(binary.stem, 0) + 1
    for _ in range(3):
        cycle()
    torch.cuda.synchronize()
    wall_times, device_times = [], []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        cycle()
        stop.record()
        stop.synchronize()
        wall_times.append(1000 * (time.perf_counter() - started))
        device_times.append(start.elapsed_time(stop))
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cycle()
    torch.cuda.synchronize()
    return {
        "shapes": shapes,
        "cold_cycle_wall_ms": cold_ms,
        "compiled_binaries": binaries,
        "warm_cycle_wall_ms": statistics.median(wall_times),
        "warm_cycle_cuda_ms": statistics.median(device_times),
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--varied-shapes", action="store_true")
    parser.add_argument("--case", choices=("baseline", "stable"))
    parser.add_argument("--kind", choices=("logprob", "entropy", "both", "training"))
    parser.add_argument("--grad", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(grad, kind) for grad in (False, True) for kind in ("logprob", "entropy", "both")]
    cases.append((True, "training"))
    policy_config = Path(__file__).parents[1] / "skyrl_train/config/megatron_config/policy.yaml"
    chunk_size = OmegaConf.load(policy_config).logprob_chunk_size
    if args.varied_shapes and args.case is None:
        result = {"commit": args.commit, "seed": args.seed, "chunk_size": chunk_size, "rows": []}
        for grad, kind in cases:
            row = {"grad": grad, "kind": kind}
            for case in ("baseline", "stable"):
                with tempfile.TemporaryDirectory() as directory:
                    command = [
                        sys.executable,
                        __file__,
                        "--varied-shapes",
                        "--case",
                        case,
                        "--kind",
                        kind,
                        "--commit",
                        args.commit,
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(Path(directory) / "case.json"),
                    ]
                    if grad:
                        command.append("--grad")
                    subprocess.run(
                        command,
                        check=True,
                        env={
                            **os.environ,
                            "TRITON_CACHE_DIR": str(Path(directory) / "triton"),
                            "TORCHINDUCTOR_CACHE_DIR": str(Path(directory) / "inductor"),
                        },
                    )
                    row[case] = json.loads((Path(directory) / "case.json").read_text())
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
        return
    torch.distributed.init_process_group("nccl", store=torch.distributed.HashStore(), rank=0, world_size=1)
    result = {
        "commit": args.commit,
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "chunk_size": chunk_size,
        "rows": [],
    }
    try:
        if args.varied_shapes:
            result.update(_varied_measurement(args.case == "stable", args.kind, args.grad, chunk_size))
            args.output.write_text(json.dumps(result, indent=2))
            return
        for tokens in args.tokens:
            logits = torch.randn((1, tokens, VOCAB_SIZE), dtype=torch.bfloat16, device="cuda")
            targets = torch.zeros((1, tokens), dtype=torch.long, device="cuda")
            for grad, kind in cases:
                row = {"tokens": tokens, "vocab": VOCAB_SIZE, "dtype": "bfloat16", "grad": grad, "kind": kind}
                for stable in (False, True):
                    row["stable" if stable else "baseline"] = _measurement(
                        logits, targets, stable, kind, grad, chunk_size
                    )
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            del logits, targets
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
