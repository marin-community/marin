# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune vLLM's Triton fused-MoE config for Hero's expert-parallel decode path (runs inside the vLLM env).

The serving path (all-gather/reduce-scatter EP) hands the Triton kernel every gathered token with
global top-k ids and an expert map that keeps only this rank's experts, so tuning must use the
same shapes: M gathered tokens, top-k over all experts, and E_local local experts. Each config is
timed with CUDA-graph replay at every M; the best config per M is written as vLLM's config-file
JSON (keyed by M; vLLM picks the nearest key).

Usage: python moe_tune_kernel.py OUT.json --experts 384 --local-experts 24 --topk 8 --hidden 3072
       --intermediate 3072 --ms 16,32,... [--gpus 4]
"""

import argparse
import functools
import itertools
import json
import time
import traceback

import torch
import torch.multiprocessing as mp
from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe import override_config

SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 64],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4, 5],
}
REPLAYS = 10
ITERS = 20


def _configs() -> list[dict[str, int]]:
    keys = list(SPACE)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(SPACE[k] for k in keys))]


def _inputs(args, m: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(m)
    x = torch.randn(m, args.hidden, device=device, dtype=torch.bfloat16, generator=generator) / 10
    scores = torch.rand(m, args.experts, device=device, generator=generator)
    topk_weights, topk_ids = torch.topk(scores, args.topk, dim=-1)
    topk_weights = topk_weights.float() / topk_weights.sum(-1, keepdim=True)
    expert_map = torch.full((args.experts,), -1, dtype=torch.int32, device=device)
    expert_map[: args.local_experts] = torch.arange(args.local_experts, dtype=torch.int32, device=device)
    return x, topk_weights, topk_ids.to(torch.int32), expert_map


def _clear_config_cache() -> None:
    """Drop any cached config lookup so override_config takes effect for every candidate.

    The 0ac9dcb wheel does not cache try_get_optimal_moe_config; newer forks wrap it in lru_cache.
    """
    function = fused_moe_module.try_get_optimal_moe_config
    while function is not None:
        if isinstance(function, functools._lru_cache_wrapper):
            function.cache_clear()
        function = getattr(function, "__wrapped__", None)


def _time(args, weights, inputs, config) -> float:
    w1, w2 = weights
    x, topk_weights, topk_ids, expert_map = inputs

    def run():
        _clear_config_cache()
        with override_config(config):
            return fused_moe_module.fused_experts_op(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                global_num_experts=args.experts,
                expert_map=expert_map,
            )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPLAYS):
            run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / REPLAYS)
    times.sort()
    return times[len(times) // 2]


def _worker(rank: int, args, configs, queue) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    w1 = torch.randn(args.local_experts, 2 * args.intermediate, args.hidden, device=device, dtype=torch.bfloat16) / 30
    w2 = torch.randn(args.local_experts, args.hidden, args.intermediate, device=device, dtype=torch.bfloat16) / 30
    inputs = {m: _inputs(args, m, device) for m in args.ms}
    results = []
    errors_shown: list[bool] = []
    for index, config in enumerate(configs):
        row = {"config": config, "ms": {}}
        for m in args.ms:
            try:
                row["ms"][m] = _time(args, (w1, w2), inputs[m], config)
            except Exception as error:  # Out-of-resource configs are skipped, not fatal.
                row["ms"][m] = None
                row["error"] = repr(error)[:2000]
                if not errors_shown:
                    traceback.print_exc()
                    errors_shown.append(True)
                break
        results.append(row)
        if index % 50 == 0:
            print(f"gpu{rank}: {index}/{len(configs)}", flush=True)
    queue.put(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--local-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    args.ms = [int(m) for m in args.ms.split(",")]
    default = {
        m: fused_moe_module.get_default_config(m, args.local_experts, args.intermediate, args.hidden, args.topk, None)
        for m in args.ms
    }
    configs = _configs()
    configs += [
        config
        for config in {json.dumps(c, sort_keys=True): c for c in default.values()}.values()
        if config not in configs
    ]
    started = time.time()
    context = mp.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(target=_worker, args=(rank, args, configs[rank :: args.gpus], queue))
        for rank in range(args.gpus)
    ]
    for worker in workers:
        worker.start()
    rows = [row for _ in workers for row in queue.get()]
    for worker in workers:
        worker.join()
    best = {}
    for m in args.ms:
        timed = [(row["ms"][m], row["config"]) for row in rows if row["ms"].get(m) is not None]
        timed.sort(key=lambda item: item[0])
        if not timed:
            raise RuntimeError(f"no config ran at M={m}")
        default_time = next((t for t, c in timed if c == default[m]), None)
        best[m] = {"config": timed[0][1], "ms": timed[0][0], "default_ms": default_time, "default": default[m]}
        print(f"M={m}: best {timed[0][0]:.4f} ms {timed[0][1]}; default {default_time}", flush=True)
    with open(args.output, "w") as handle:
        json.dump(
            {
                "vllm_config": {str(m): best[m]["config"] for m in args.ms},
                "best": {str(m): best[m] for m in args.ms},
                "rows": rows,
                "seconds": time.time() - started,
                "args": vars(args),
            },
            handle,
        )


if __name__ == "__main__":
    main()
