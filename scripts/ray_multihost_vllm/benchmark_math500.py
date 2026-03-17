#!/usr/bin/env python3
"""MATH-500 benchmark for vLLM serving.

Sends all 500 MATH-500 problems to a vLLM server, collects per-request
latency/throughput metrics, grades answers with SymPy, and reports timing
for both inference and reward computation.

Usage:
  python benchmark_math500.py --server http://localhost:8000 --concurrency 16
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# MATH-500 dataset loading
# ---------------------------------------------------------------------------

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."
MATH500_HF = "HuggingFaceH4/MATH-500"


def load_math500_from_hf():
    """Load MATH-500 from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(MATH500_HF, name="default", split="test")
    problems = []
    for ex in ds:
        answer = extract_boxed(ex["solution"]) if "\\boxed" in ex["solution"] else ex["solution"].strip()
        problems.append({
            "problem": ex["problem"],
            "answer": answer,
            "prompt": ex["problem"] + QUESTION_SUFFIX,
        })
    return problems


def load_math500_from_jsonl(path: str):
    """Load MATH-500 from a local JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            answer = extract_boxed(ex.get("solution", "")) if "\\boxed" in ex.get("solution", "") else ex.get("answer", "").strip()
            problems.append({
                "problem": ex["problem"],
                "answer": answer,
                "prompt": ex["problem"] + QUESTION_SUFFIX,
            })
    return problems


# ---------------------------------------------------------------------------
# Answer extraction and grading
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} with nested brace handling."""
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
    if depth == 0:
        return text[start:i - 1].strip()
    return ""


def normalize_answer(ans: str) -> str:
    """Basic normalization for answer comparison."""
    ans = ans.strip()
    ans = ans.replace("\\$", "").replace("$", "")
    ans = ans.replace("\\%", "").replace("%", "")
    ans = ans.replace("\\text{", "").replace("\\mathrm{", "")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace(" ", "")
    return ans


def grade_answer(model_answer: str, ground_truth: str) -> bool:
    """Grade answer using string comparison. Falls back to SymPy if available."""
    if not model_answer or not ground_truth:
        return False
    # Direct string match after normalization
    if normalize_answer(model_answer) == normalize_answer(ground_truth):
        return True
    # Try numeric comparison
    try:
        m = float(model_answer.replace(",", ""))
        g = float(ground_truth.replace(",", ""))
        if abs(m - g) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass
    # Try SymPy grading
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
        try:
            m_expr = parse_latex(model_answer)
            g_expr = parse_latex(ground_truth)
            if simplify(m_expr - g_expr) == 0:
                return True
        except Exception:
            pass
        try:
            m_expr = sympify(model_answer)
            g_expr = sympify(ground_truth)
            if simplify(m_expr - g_expr) == 0:
                return True
        except Exception:
            pass
    except ImportError:
        pass
    return False


# ---------------------------------------------------------------------------
# Request/response data
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    idx: int
    prompt: str
    ground_truth: str
    response_text: str = ""
    model_answer: str = ""
    correct: bool = False
    error: str = ""
    # Timing (seconds)
    ttft: float = 0.0  # time to first token
    tpot: float = 0.0  # time per output token (mean)
    e2e: float = 0.0   # end-to-end latency
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Reward timing
    reward_time: float = 0.0  # time to grade this response


# ---------------------------------------------------------------------------
# Async benchmark client
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    server: str,
    model: str,
    problem: dict,
    idx: int,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    result = RequestResult(idx=idx, prompt=problem["prompt"], ground_truth=problem["answer"])
    url = f"{server}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": problem["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    async with semaphore:
        t_start = time.perf_counter()
        first_token_time = None
        chunks = []
        token_times = []

        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    result.e2e = time.perf_counter() - t_start
                    return result

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        chunks.append(content)

                    # Capture usage from final chunk
                    usage = data.get("usage")
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens", 0)
                        result.completion_tokens = usage.get("completion_tokens", 0)

        except asyncio.TimeoutError:
            result.error = "timeout"
            result.e2e = time.perf_counter() - t_start
            return result
        except Exception as e:
            result.error = str(e)
            result.e2e = time.perf_counter() - t_start
            return result

        t_end = time.perf_counter()
        result.e2e = t_end - t_start
        result.response_text = "".join(chunks)

        if first_token_time is not None:
            result.ttft = first_token_time - t_start
        if len(token_times) > 1:
            inter_token = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            result.tpot = statistics.mean(inter_token)

        # If usage wasn't in the stream, estimate from chunks
        if result.completion_tokens == 0:
            result.completion_tokens = len(chunks)

    return result


async def run_benchmark(
    server: str,
    model: str,
    problems: list[dict],
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_request(session, server, model, p, i, max_tokens, temperature, semaphore)
            for i, p in enumerate(problems)
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# Grade all responses and time it
# ---------------------------------------------------------------------------

def grade_all(results: list[RequestResult]) -> None:
    """Grade all responses and record per-response reward timing."""
    for r in results:
        if r.error:
            continue
        t0 = time.perf_counter()
        r.model_answer = extract_boxed(r.response_text)
        r.correct = grade_answer(r.model_answer, r.ground_truth)
        r.reward_time = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Compute and print summary
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(data_sorted) else f
    d = k - f
    return data_sorted[f] + d * (data_sorted[c] - data_sorted[f])


def compute_summary(results: list[RequestResult], batch_wall_time: float) -> dict:
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    ttfts = [r.ttft * 1000 for r in ok if r.ttft > 0]  # ms
    tpots = [r.tpot * 1000 for r in ok if r.tpot > 0]  # ms
    e2es = [r.e2e * 1000 for r in ok]  # ms

    total_prompt_toks = sum(r.prompt_tokens for r in ok)
    total_gen_toks = sum(r.completion_tokens for r in ok)

    correct_count = sum(1 for r in ok if r.correct)
    format_count = sum(1 for r in ok if "\\boxed" in r.response_text or "boxed{" in r.response_text)
    empty_count = sum(1 for r in ok if not r.response_text.strip())

    reward_times = [r.reward_time * 1000 for r in ok if r.reward_time > 0]  # ms

    summary = {
        "total_requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(errors),
        "error_types": {},

        # Throughput
        "batch_wall_time_s": round(batch_wall_time, 2),
        "prompt_tok_per_s": round(total_prompt_toks / batch_wall_time, 1) if batch_wall_time > 0 else 0,
        "gen_tok_per_s": round(total_gen_toks / batch_wall_time, 1) if batch_wall_time > 0 else 0,
        "total_tok_per_s": round((total_prompt_toks + total_gen_toks) / batch_wall_time, 1) if batch_wall_time > 0 else 0,
        "req_per_s": round(len(ok) / batch_wall_time, 2) if batch_wall_time > 0 else 0,

        # Latency (ms)
        "p50_ttft_ms": round(percentile(ttfts, 50), 1),
        "p95_ttft_ms": round(percentile(ttfts, 95), 1),
        "p99_ttft_ms": round(percentile(ttfts, 99), 1),
        "p50_tpot_ms": round(percentile(tpots, 50), 1),
        "p95_tpot_ms": round(percentile(tpots, 95), 1),
        "p99_tpot_ms": round(percentile(tpots, 99), 1),
        "p50_e2e_ms": round(percentile(e2es, 50), 1),
        "p95_e2e_ms": round(percentile(e2es, 95), 1),
        "p99_e2e_ms": round(percentile(e2es, 99), 1),

        # Correctness
        "math500_accuracy_pct": round(correct_count / len(ok) * 100, 2) if ok else 0,
        "math500_correct": correct_count,
        "math500_total": len(ok),
        "math500_format_rate_pct": round(format_count / len(ok) * 100, 2) if ok else 0,
        "math500_empty_rate_pct": round(empty_count / len(ok) * 100, 2) if ok else 0,

        # Reward timing (ms)
        "t_reward_total_s": round(sum(r.reward_time for r in ok), 3),
        "t_reward_per_response_ms": round(statistics.mean(reward_times), 2) if reward_times else 0,
        "p50_t_reward_ms": round(percentile(reward_times, 50), 2),
        "p95_t_reward_ms": round(percentile(reward_times, 95), 2),
        "p99_t_reward_ms": round(percentile(reward_times, 99), 2),
        "reward_timeouts": 0,  # TODO: track if sympy times out

        # Token counts
        "total_prompt_tokens": total_prompt_toks,
        "total_gen_tokens": total_gen_toks,
        "mean_prompt_tokens": round(total_prompt_toks / len(ok), 1) if ok else 0,
        "mean_gen_tokens": round(total_gen_toks / len(ok), 1) if ok else 0,
    }

    # Categorize errors
    for r in errors:
        err_type = r.error.split(":")[0] if ":" in r.error else r.error
        summary["error_types"][err_type] = summary["error_types"].get(err_type, 0) + 1

    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 70)
    print("MATH-500 BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'THROUGHPUT':=^50}")
    print(f"  Batch wall time:     {summary['batch_wall_time_s']:>10.1f} s")
    print(f"  Requests/sec:        {summary['req_per_s']:>10.2f}")
    print(f"  Prompt tok/s:        {summary['prompt_tok_per_s']:>10.1f}")
    print(f"  Gen tok/s:           {summary['gen_tok_per_s']:>10.1f}")
    print(f"  Total tok/s:         {summary['total_tok_per_s']:>10.1f}")

    print(f"\n{'LATENCY (ms)':=^50}")
    print(f"  {'':20s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
    print(f"  TTFT:                {summary['p50_ttft_ms']:>8.1f} {summary['p95_ttft_ms']:>8.1f} {summary['p99_ttft_ms']:>8.1f}")
    print(f"  TPOT:                {summary['p50_tpot_ms']:>8.1f} {summary['p95_tpot_ms']:>8.1f} {summary['p99_tpot_ms']:>8.1f}")
    print(f"  E2E:                 {summary['p50_e2e_ms']:>8.1f} {summary['p95_e2e_ms']:>8.1f} {summary['p99_e2e_ms']:>8.1f}")

    print(f"\n{'CORRECTNESS':=^50}")
    print(f"  Accuracy:            {summary['math500_accuracy_pct']:>8.2f}% ({summary['math500_correct']}/{summary['math500_total']})")
    print(f"  Format rate:         {summary['math500_format_rate_pct']:>8.2f}%")
    print(f"  Empty rate:          {summary['math500_empty_rate_pct']:>8.2f}%")

    print(f"\n{'REWARD TIMING':=^50}")
    print(f"  Total grading time:  {summary['t_reward_total_s']:>10.3f} s")
    print(f"  Mean per response:   {summary['t_reward_per_response_ms']:>10.2f} ms")
    print(f"  p50/p95/p99:         {summary['p50_t_reward_ms']:>8.2f} / {summary['p95_t_reward_ms']:>8.2f} / {summary['p99_t_reward_ms']:>8.2f} ms")
    print(f"  Timeouts:            {summary['reward_timeouts']:>10d}")

    print(f"\n{'TOKENS':=^50}")
    print(f"  Total prompt:        {summary['total_prompt_tokens']:>10d}")
    print(f"  Total generated:     {summary['total_gen_tokens']:>10d}")
    print(f"  Mean prompt:         {summary['mean_prompt_tokens']:>10.1f}")
    print(f"  Mean generated:      {summary['mean_gen_tokens']:>10.1f}")

    print(f"\n{'ERRORS':=^50}")
    print(f"  Failed requests:     {summary['failed_requests']:>10d} / {summary['total_requests']}")
    if summary["error_types"]:
        for etype, count in summary["error_types"].items():
            print(f"    {etype}: {count}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MATH-500 benchmark for vLLM")
    parser.add_argument("--server", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="", help="Model name (auto-detected if empty)")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--dataset-path", default="", help="Local JSONL path (uses HF if empty)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of problems (0=all 500)")
    parser.add_argument("--output", default="", help="Save results JSON to this path")
    args = parser.parse_args()

    # Load dataset
    print("Loading MATH-500 dataset...")
    if args.dataset_path:
        problems = load_math500_from_jsonl(args.dataset_path)
    else:
        problems = load_math500_from_hf()
    print(f"Loaded {len(problems)} problems")

    if args.limit > 0:
        problems = problems[:args.limit]
        print(f"Limited to {len(problems)} problems")

    # Auto-detect model name
    model = args.model
    if not model:
        import urllib.request
        try:
            req = urllib.request.Request(f"{args.server}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                model = data["data"][0]["id"]
                print(f"Auto-detected model: {model}")
        except Exception as e:
            print(f"Could not auto-detect model: {e}")
            sys.exit(1)

    print(f"\nBenchmark config:")
    print(f"  Server:      {args.server}")
    print(f"  Model:       {model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Problems:    {len(problems)}")
    print()

    # Run benchmark
    print("Running benchmark...")
    t_bench_start = time.perf_counter()
    results = asyncio.run(run_benchmark(
        server=args.server,
        model=model,
        problems=problems,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ))
    batch_wall_time = time.perf_counter() - t_bench_start

    # Grade answers
    print("Grading answers...")
    t_grade_start = time.perf_counter()
    grade_all(results)
    t_grade_total = time.perf_counter() - t_grade_start
    print(f"Grading took {t_grade_total:.3f}s")

    # Compute summary
    summary = compute_summary(results, batch_wall_time)
    print_summary(summary)

    # Save to file
    if args.output:
        output_data = {
            "summary": summary,
            "config": {
                "server": args.server,
                "model": model,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "num_problems": len(problems),
            },
            "per_request": [
                {
                    "idx": r.idx,
                    "correct": r.correct,
                    "model_answer": r.model_answer,
                    "ground_truth": r.ground_truth,
                    "ttft_ms": round(r.ttft * 1000, 1),
                    "tpot_ms": round(r.tpot * 1000, 1),
                    "e2e_ms": round(r.e2e * 1000, 1),
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "reward_time_ms": round(r.reward_time * 1000, 2),
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Return exit code based on errors
    if summary["failed_requests"] > len(results) * 0.1:
        print("\nWARNING: >10% of requests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
