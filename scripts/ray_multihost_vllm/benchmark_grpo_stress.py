#!/usr/bin/env python3
"""GRPO mini-batch stress test for vLLM serving.

Simulates Marin's exp2039_rl_math500 inference workload:
- 10 mini-batches, each with 64 prompts × 16 completions = 1,024 requests
- temperature=1.0, max_tokens=1024 (matching Marin config)
- Round-robin across multiple replicas
- Grades responses and times reward computation

Usage:
  python benchmark_grpo_stress.py \
    --servers http://10.0.0.1:8000,http://10.0.0.2:8000,http://10.0.0.3:8000,http://10.0.0.4:8000 \
    --num-batches 10 --output /tmp/grpo_stress.json
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def load_prompts(dataset_path: str, n_prompts: int):
    """Load n_prompts from a prepared JSONL file (from prepare_hendrycks_math.py).

    Falls back to HuggingFace download if no file provided.
    """
    if dataset_path:
        prompts = []
        with open(dataset_path) as f:
            for line in f:
                ex = json.loads(line)
                prompts.append({
                    "problem": ex["problem"],
                    "answer": ex["answer"],
                    "prompt": ex["prompt"],
                })
                if len(prompts) >= n_prompts:
                    break
        return prompts

    # Fallback: download from HuggingFace (non-deterministic order!)
    from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
    print("WARNING: Loading from HuggingFace — order is NOT deterministic. Use --dataset-path for reproducibility.")
    test_ds = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    test_problems = {ex["problem"] for ex in test_ds}
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split)
                ds = ds.filter(lambda ex: ex["problem"] not in test_problems)
                pieces.append(ds)
            except Exception:
                pass
    full = concatenate_datasets(pieces)
    prompts = []
    for ex in full:
        answer = extract_boxed(ex["solution"]) if "\\boxed" in ex["solution"] else ex["solution"].strip()
        prompts.append({
            "problem": ex["problem"],
            "answer": answer,
            "prompt": ex["problem"] + QUESTION_SUFFIX,
        })
        if len(prompts) >= n_prompts:
            break
    return prompts


# ---------------------------------------------------------------------------
# Answer extraction / grading (minimal)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    i = text.find("\\boxed")
    if i == -1:
        return ""
    i += 6
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return ""
    i += 1
    start = i
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else ""


def grade_answer(model_answer: str, ground_truth: str) -> bool:
    if not model_answer or not ground_truth:
        return False
    ma = model_answer.strip().replace(" ", "")
    gt = ground_truth.strip().replace(" ", "")
    if ma == gt:
        return True
    try:
        if abs(float(ma.replace(",", "")) - float(gt.replace(",", ""))) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass
    return False


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    async with semaphore:
        t0 = time.perf_counter()
        first_tok_time = None
        chunks = []
        token_times = []
        error = ""

        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
                if resp.status != 200:
                    error = f"HTTP {resp.status}"
                    return {"error": error, "e2e": time.perf_counter() - t0, "tokens": 0}

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if first_tok_time is None:
                            first_tok_time = now
                        token_times.append(now)
                        chunks.append(content)
        except asyncio.TimeoutError:
            error = "timeout"
        except Exception as e:
            error = str(e)[:100]

        t_end = time.perf_counter()
        e2e = t_end - t0
        ttft = (first_tok_time - t0) if first_tok_time else e2e
        tpot = statistics.mean([token_times[i] - token_times[i-1] for i in range(1, len(token_times))]) if len(token_times) > 1 else 0

        return {
            "text": "".join(chunks),
            "tokens": len(chunks),
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e,
            "error": error,
        }


# ---------------------------------------------------------------------------
# Run one mini-batch
# ---------------------------------------------------------------------------

async def run_mini_batch(
    servers: list[str],
    model: str,
    prompts: list[dict],  # 64 prompts
    n_gen: int,           # 16 completions per prompt
    max_tokens: int,
    temperature: float,
    concurrency_per_replica: int,
    stream_file: str = "",  # JSONL file to stream results as they complete
) -> dict:
    """Run one mini-batch: 64 prompts × 16 completions = 1024 requests across replicas."""

    # Build request list: 64 prompts × 16 = 1024, round-robin servers
    requests = []
    for i, p in enumerate(prompts):
        for g in range(n_gen):
            server_idx = (i * n_gen + g) % len(servers)
            requests.append((servers[server_idx], p, i, g))

    total_concurrency = concurrency_per_replica * len(servers)
    semaphore = asyncio.Semaphore(total_concurrency)
    connector = aiohttp.TCPConnector(limit=total_concurrency * 2)

    # Open stream file for writing results as they complete
    stream_fh = open(stream_file, "w") if stream_file else None
    completed_count = [0]

    async def send_and_record(session, server, p, prompt_idx, gen_idx):
        r = await send_request(
            session, f"{server}/v1/chat/completions",
            model, p["prompt"], max_tokens, temperature, semaphore
        )
        # Stream result to JSONL immediately
        if stream_fh:
            model_ans = extract_boxed(r.get("text", ""))
            correct = grade_answer(model_ans, p["answer"]) if not r.get("error") else False
            record = {
                "prompt_idx": prompt_idx, "gen_idx": gen_idx,
                "problem": p["problem"], "ground_truth": p["answer"],
                "model_response": r.get("text", ""), "model_answer": model_ans,
                "correct": correct, "tokens": r.get("tokens", 0),
                "finish_reason": "error" if r.get("error") else ("length" if r.get("tokens", 0) >= max_tokens - 1 else "stop"),
                "ttft_ms": round(r.get("ttft", 0) * 1000, 1),
                "e2e_ms": round(r.get("e2e", 0) * 1000, 1),
                "error": r.get("error", ""),
            }
            stream_fh.write(json.dumps(record) + "\n")
            stream_fh.flush()
        completed_count[0] += 1
        return r

    t_batch_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_and_record(session, server, p, prompt_idx, gen_idx)
            for server, p, prompt_idx, gen_idx in requests
        ]
        results = await asyncio.gather(*tasks)

    if stream_fh:
        stream_fh.close()

    t_batch_end = time.perf_counter()
    batch_wall = t_batch_end - t_batch_start

    # Grade and time rewards
    t_reward_start = time.perf_counter()
    correct = 0
    for req, r in zip(requests, results):
        p = req[1]  # prompt dict
        if r.get("error"):
            continue
        model_ans = extract_boxed(r.get("text", ""))
        if grade_answer(model_ans, p["answer"]):
            correct += 1
    t_reward = time.perf_counter() - t_reward_start

    # Compute stats
    ok = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    total_tokens = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft"] * 1000 for r in ok if r["ttft"] > 0]
    tpots = [r["tpot"] * 1000 for r in ok if r["tpot"] > 0]
    e2es = [r["e2e"] * 1000 for r in ok]

    def pct(data, p):
        if not data:
            return 0
        s = sorted(data)
        k = (len(s) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (k - f) * (s[c] - s[f])

    return {
        "t_batch_wall": round(batch_wall, 2),
        "total_requests": len(results),
        "successful": len(ok),
        "errors": len(errors),
        "total_gen_tokens": total_tokens,
        "mean_gen_tokens": round(total_tokens / len(ok), 1) if ok else 0,
        "gen_tok_per_s": round(total_tokens / batch_wall, 1) if batch_wall > 0 else 0,
        "req_per_s": round(len(ok) / batch_wall, 2) if batch_wall > 0 else 0,
        "p50_ttft_ms": round(pct(ttfts, 50), 1),
        "p95_ttft_ms": round(pct(ttfts, 95), 1),
        "p50_tpot_ms": round(pct(tpots, 50), 1),
        "p95_tpot_ms": round(pct(tpots, 95), 1),
        "p50_e2e_ms": round(pct(e2es, 50), 1),
        "p95_e2e_ms": round(pct(e2es, 95), 1),
        "accuracy_pct": round(correct / len(ok) * 100, 2) if ok else 0,
        "t_reward_s": round(t_reward, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO mini-batch stress test")
    parser.add_argument("--servers", required=True, help="Comma-separated vLLM server URLs")
    parser.add_argument("--model", default="", help="Model name (auto-detected)")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--n-prompts", type=int, default=64, help="Prompts per mini-batch")
    parser.add_argument("--n-gen", type=int, default=16, help="Completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency-per-replica", type=int, default=64)
    parser.add_argument("--dataset-path", default="", help="Path to prepared JSONL (from prepare_hendrycks_math.py)")
    parser.add_argument("--output", default="", help="Save JSON results")
    parser.add_argument("--stream-output", default="", help="Stream per-response results to JSONL as they complete")
    args = parser.parse_args()

    servers = [s.strip() for s in args.servers.split(",")]
    total_prompts_needed = args.n_prompts * args.num_batches

    # Auto-detect model
    model = args.model
    if not model:
        import urllib.request
        try:
            req = urllib.request.Request(f"{servers[0]}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                model = json.loads(resp.read())["data"][0]["id"]
        except Exception as e:
            print(f"Could not detect model: {e}")
            sys.exit(1)

    print(f"Loading {total_prompts_needed} prompts...")
    all_prompts = load_prompts(args.dataset_path, total_prompts_needed)
    print(f"Loaded {len(all_prompts)} prompts")

    print(f"\nGRPO Stress Test Config:")
    print(f"  Servers:        {servers}")
    print(f"  Model:          {model}")
    print(f"  Batches:        {args.num_batches}")
    print(f"  Per batch:      {args.n_prompts} prompts × {args.n_gen} gen = {args.n_prompts * args.n_gen} completions")
    print(f"  Total:          {args.num_batches * args.n_prompts * args.n_gen} completions")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Concurrency:    {args.concurrency_per_replica}/replica × {len(servers)} replicas")
    if args.stream_output:
        print(f"  Stream output:  {args.stream_output}.batch<N> (JSONL, live)")
    print()

    from tqdm import tqdm

    batch_results = []
    t_total_start = time.perf_counter()
    total_completions = args.num_batches * args.n_prompts * args.n_gen
    completions_done = 0

    pbar = tqdm(total=total_completions, desc="GRPO completions", unit="req",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] batch {postfix}")
    pbar.set_postfix_str("0/?")

    for batch_id in range(args.num_batches):
        start_idx = batch_id * args.n_prompts
        batch_prompts = all_prompts[start_idx:start_idx + args.n_prompts]

        stream_file = f"{args.stream_output}.batch{batch_id+1}" if args.stream_output else ""
        result = asyncio.run(run_mini_batch(
            servers=servers,
            model=model,
            prompts=batch_prompts,
            n_gen=args.n_gen,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency_per_replica=args.concurrency_per_replica,
            stream_file=stream_file,
        ))
        result["batch_id"] = batch_id + 1
        batch_results.append(result)

        batch_completions = result["successful"] + result["errors"]
        completions_done += batch_completions
        pbar.update(batch_completions)
        pbar.set_postfix_str(
            f"{batch_id + 1}/{args.num_batches} | "
            f"{result['t_batch_wall']:.0f}s | "
            f"{result['gen_tok_per_s']:.0f} tok/s | "
            f"{result['errors']} err"
        )

    pbar.close()
    t_total = time.perf_counter() - t_total_start

    # Aggregate
    wall_times = [r["t_batch_wall"] for r in batch_results]
    tok_rates = [r["gen_tok_per_s"] for r in batch_results]
    total_tokens = sum(r["total_gen_tokens"] for r in batch_results)
    total_errors = sum(r["errors"] for r in batch_results)
    total_reward = sum(r["t_reward_s"] for r in batch_results)
    mean_batch = statistics.mean(wall_times)
    stddev_batch = statistics.stdev(wall_times) if len(wall_times) > 1 else 0

    print(f"\n{'=' * 70}")
    print(f"GRPO STRESS TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total wall time:            {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Total completions:          {sum(r['successful'] for r in batch_results)}")
    print(f"  Total tokens generated:     {total_tokens:,}")
    print(f"  Total errors:               {total_errors}")
    print(f"  Total reward time:          {total_reward:.2f}s")
    print(f"")
    print(f"  Mean batch wall time:       {mean_batch:.1f}s")
    print(f"  Stddev batch wall time:     {stddev_batch:.1f}s")
    print(f"  Mean throughput:            {statistics.mean(tok_rates):.0f} tok/s")
    print(f"  Min/Max batch time:         {min(wall_times):.1f}s / {max(wall_times):.1f}s")
    print(f"")
    print(f"  --- Extrapolation to full epoch (188 batches) ---")
    print(f"  Projected inference time:   {mean_batch * 188:.0f}s ({mean_batch * 188 / 3600:.1f} hours)")
    print(f"  Projected reward time:      {statistics.mean([r['t_reward_s'] for r in batch_results]) * 188:.0f}s")
    print(f"{'=' * 70}")

    # Per-batch table
    print(f"\n{'Batch':>5} {'Wall(s)':>8} {'Tok/s':>8} {'Req/s':>7} {'MeanTok':>8} {'Errors':>7} {'Reward(s)':>9} {'Acc%':>6}")
    for r in batch_results:
        print(f"{r['batch_id']:>5} {r['t_batch_wall']:>8.1f} {r['gen_tok_per_s']:>8.0f} {r['req_per_s']:>7.2f} {r['mean_gen_tokens']:>8.0f} {r['errors']:>7} {r['t_reward_s']:>9.3f} {r['accuracy_pct']:>6.1f}")

    if args.output:
        out = {
            "config": {
                "servers": servers, "model": model,
                "num_batches": args.num_batches, "n_prompts": args.n_prompts,
                "n_gen": args.n_gen, "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "aggregate": {
                "total_wall_time_s": round(t_total, 2),
                "total_tokens": total_tokens,
                "total_errors": total_errors,
                "total_reward_s": round(total_reward, 3),
                "mean_batch_wall_s": round(mean_batch, 2),
                "stddev_batch_wall_s": round(stddev_batch, 2),
                "mean_tok_per_s": round(statistics.mean(tok_rates), 1),
                "projected_epoch_s": round(mean_batch * 188, 0),
                "projected_epoch_hours": round(mean_batch * 188 / 3600, 2),
            },
            "batches": batch_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
