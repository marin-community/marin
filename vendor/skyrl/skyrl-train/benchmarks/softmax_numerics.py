# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

from skyrl_train.distributed.megatron.stable_softmax import token_entropy, token_logprobs


@torch.compile(dynamic=True)
def _mul_reduce(a, b):
    return (a * b).sum(dim=-1, keepdim=True)


class _BaselineEntropy(torch.autograd.Function):
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


def _operation(logits, targets, stable, kind, grad):
    with torch.set_grad_enabled(grad):
        outputs = []
        if kind in ("logprob", "both"):
            outputs.append(
                token_logprobs(logits, targets, 0, logits.shape[-1], dist.group.WORLD)
                if stable
                else -parallel_cross_entropy(
                    logits, targets, dist_process_group=dist.group.WORLD, overwrite_input=False
                )
            )
        if kind in ("entropy", "both"):
            outputs.append(token_entropy(logits, dist.group.WORLD) if stable else _BaselineEntropy.apply(logits))
        if grad:
            return outputs, torch.autograd.grad(sum(output.sum() for output in outputs), logits)[0]
        return outputs


def _measurement(logits, targets, stable, kind, grad):
    logits.requires_grad_(grad)
    for _ in range(3):
        _operation(logits, targets, stable, kind, grad)
    torch.cuda.synchronize()
    durations = []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _operation(logits, targets, stable, kind, grad)
        stop.record()
        stop.synchronize()
        durations.append(start.elapsed_time(stop))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = _operation(logits, targets, stable, kind, grad)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del result
    return {"median_ms": statistics.median(durations), "min_ms": min(durations), "peak_extra_bytes": peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.distributed.init_process_group("nccl", store=torch.distributed.HashStore(), rank=0, world_size=1)
    result = {"commit": args.commit, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "rows": []}
    try:
        for tokens in args.tokens:
            logits = torch.randn((1, tokens, 151936), dtype=torch.bfloat16, device="cuda")
            targets = torch.zeros((1, tokens), dtype=torch.long, device="cuda")
            for grad in (False, True):
                for kind in ("logprob", "entropy", "both"):
                    row = {"tokens": tokens, "vocab": 151936, "dtype": "bfloat16", "grad": grad, "kind": kind}
                    for stable in (False, True):
                        row["stable" if stable else "baseline"] = _measurement(logits, targets, stable, kind, grad)
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2))
                    print(json.dumps(row), flush=True)
            del logits, targets
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
