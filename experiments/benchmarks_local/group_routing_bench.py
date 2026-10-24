# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Group-on-one-rank versus spread routing of same-prompt sample groups on one DP x EP vLLM engine.

Each case sends ``groups`` prompt groups of ``samples`` requests that share one prompt. ``grouped`` pins a group's
requests to one data-parallel rank (group g -> rank g % dp); ``spread`` sends sample j of group g to rank
(g * samples + j) % dp, the placement that per-sample session hashing approximates. Every case uses fresh random
prompts and starts from an empty prefix cache. Output: one JSON line per case on stdout and in --out.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
import uuid

import fsspec
import vllm
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt


def stage_model(uri: str, destination: str) -> str:
    if os.path.exists(os.path.join(destination, "config.json")):
        return destination
    started = time.monotonic()
    fs, path = fsspec.core.url_to_fs(uri)
    fs.get(path.rstrip("/") + "/", destination, recursive=True)
    print(json.dumps({"event": "staged", "uri": uri, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return destination


async def run_case(
    engine, args, mode: str, prompt_len: int, decode_len: int, groups: int, group_interval: float, seed: int
) -> dict:
    await engine.reset_prefix_cache()
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000, args.vocab_limit) for _ in range(prompt_len)] for _ in range(groups)]
    sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=decode_len, ignore_eos=True)

    async def one(group: int, sample: int, rank: int, delay: float) -> dict:
        await asyncio.sleep(delay)
        start = time.monotonic()
        first = None
        final = None
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=prompts[group]),
            sampling,
            f"{mode}-{group}-{sample}-{uuid.uuid4().hex}",
            data_parallel_rank=rank,
        ):
            if first is None and output.outputs[0].token_ids:
                first = time.monotonic()
            final = output
        end = time.monotonic()
        assert final is not None
        return {
            "rank": rank,
            "prompt_tokens": len(prompts[group]),
            "cached_tokens": final.num_cached_tokens or 0,
            "output_tokens": len(final.outputs[0].token_ids),
            "ttft": (first or end) - start,
            "e2e": end - start,
        }

    tasks = []
    started = time.monotonic()
    for group in range(groups):
        delay = group * group_interval
        for sample in range(args.samples):
            rank = group % args.dp if mode == "grouped" else (group * args.samples + sample) % args.dp
            tasks.append(asyncio.create_task(one(group, sample, rank, delay)))
    rows = await asyncio.gather(*tasks)
    wall = time.monotonic() - started
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached = sum(row["cached_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    per_rank = [sum(1 for row in rows if row["rank"] == rank) for rank in range(args.dp)]
    return {
        "mode": mode,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "groups": groups,
        "samples": args.samples,
        "group_interval": group_interval,
        "requests": len(rows),
        "requests_per_rank": per_rank,
        "wall_seconds": round(wall, 2),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "prefill_tokens_computed": prompt_tokens - cached,
        "prefix_hit_rate": round(cached / prompt_tokens, 4),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / wall, 1),
        "output_tokens_per_second_per_gpu": round(output_tokens / wall / args.dp, 1),
        "ttft_median": round(statistics.median(row["ttft"] for row in rows), 3),
        "ttft_p90": round(sorted(row["ttft"] for row in rows)[int(0.9 * len(rows))], 3),
        "e2e_median": round(statistics.median(row["e2e"] for row in rows), 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--model-dir", default="/tmp/bench-model")
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    # prompt tokens x decode tokens x groups x seconds between group arrivals
    parser.add_argument("--shapes", default="4096x512x64x0.25,4096x512x64x0,1024x2048x64x0.25,16384x256x16x0.5")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--vocab-limit", type=int, default=100000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--out", default="/tmp/group_routing_bench.jsonl")
    args = parser.parse_args()

    model = stage_model(args.model_uri, args.model_dir)
    engine = vllm.AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model,
            tensor_parallel_size=1,
            data_parallel_size=args.dp,
            enable_expert_parallel=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_seqs=1024,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            moe_backend="triton",
            attention_backend="FLASH_ATTN",
        )
    )
    with open(args.out, "w") as out:
        warm = await run_case(engine, args, "spread", 512, 64, groups=16, group_interval=0.0, seed=1)
        print(json.dumps({"event": "warmup", **warm}), flush=True)
        seed = 100
        for shape in args.shapes.split(","):
            prompt_text, decode_text, groups_text, interval_text = shape.split("x")
            prompt_len, decode_len, groups = int(prompt_text), int(decode_text), int(groups_text)
            for repeat in range(args.repeats):
                order = ("grouped", "spread") if repeat % 2 == 0 else ("spread", "grouped")
                for mode in order:
                    seed += 1
                    case = await run_case(
                        engine, args, mode, prompt_len, decode_len, groups, float(interval_text), seed
                    )
                    result = {"event": "case", "repeat": repeat, **case}
                    print(json.dumps(result), flush=True)
                    out.write(json.dumps(result) + "\n")
                    out.flush()
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
