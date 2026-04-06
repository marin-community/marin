#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment
from marin.rl.environments.math_env import MathEnv
from marin.utils import remove_tpu_lockfile_on_exit

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_STOP = ["<|eot_id|>"]


@dataclass
class ChoiceResult:
    text: str
    finish_reason: str


@dataclass
class PromptRequestResult:
    request_index: int
    dataset_index: int
    prompt_tokens: int | None
    e2e: float
    completion_tokens: int | None
    choices: list[ChoiceResult]
    error: str | None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def build_prompt_messages(env: MathEnv, example: Any) -> list[dict[str, str]]:
    return [*env.fewshot_prefix, {"role": "user", "content": example.processed_prompt}]


def choose_indices(
    *,
    dataset_size: int,
    num_examples: int,
    seed: int,
    excluded_indices: set[int] | None = None,
) -> list[int]:
    if num_examples > dataset_size:
        raise ValueError(f"Requested {num_examples} examples, but only {dataset_size} train examples exist.")

    excluded_indices = excluded_indices or set()
    available = dataset_size - len(excluded_indices)
    if num_examples > available:
        raise ValueError(f"Requested {num_examples} examples, but only {available} non-excluded examples exist.")

    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for raw_index in rng.permutation(dataset_size):
        dataset_index = int(raw_index)
        if dataset_index in excluded_indices:
            continue
        indices.append(dataset_index)
        if len(indices) == num_examples:
            return indices

    raise RuntimeError("Failed to choose enough prompt indices.")


def choose_benchmark_examples(
    env: MathEnv,
    *,
    measured_count: int,
    warmup_count: int,
    seed: int,
) -> tuple[list[int], list[Any], list[int], list[Any]]:
    dataset_size = len(env.train_examples)
    measured_indices = choose_indices(dataset_size=dataset_size, num_examples=measured_count, seed=seed)
    warmup_indices = choose_indices(
        dataset_size=dataset_size,
        num_examples=warmup_count,
        seed=seed + 1,
        excluded_indices=set(measured_indices),
    )
    measured_examples = [env.train_examples[index] for index in measured_indices]
    warmup_examples = [env.train_examples[index] for index in warmup_indices]
    return warmup_indices, warmup_examples, measured_indices, measured_examples


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int | None:
    try:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        return None
    return len(token_ids)


async def run_prompt_request(
    *,
    client: AsyncOpenAI,
    model_id: str,
    messages: list[dict[str, str]],
    request_index: int,
    dataset_index: int,
    prompt_tokens: int | None,
    max_tokens: int,
    temperature: float,
    n_generations: int,
    top_k: int,
    stop: list[str],
    semaphore: asyncio.Semaphore | None,
) -> PromptRequestResult:
    started = time.perf_counter()

    async def _issue_request() -> PromptRequestResult:
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n_generations,
                stop=stop,
                extra_body={"top_k": top_k},
                timeout=1800,
            )
        except Exception as exc:
            return PromptRequestResult(
                request_index=request_index,
                dataset_index=dataset_index,
                prompt_tokens=prompt_tokens,
                e2e=time.perf_counter() - started,
                completion_tokens=None,
                choices=[],
                error=repr(exc),
            )

        usage = getattr(response, "usage", None)
        completion_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None
        choices = [
            ChoiceResult(
                text=choice.message.content or "",
                finish_reason=str(choice.finish_reason or "unknown"),
            )
            for choice in response.choices
        ]
        return PromptRequestResult(
            request_index=request_index,
            dataset_index=dataset_index,
            prompt_tokens=prompt_tokens,
            e2e=time.perf_counter() - started,
            completion_tokens=completion_tokens,
            choices=choices,
            error=None,
        )

    if semaphore is None:
        return await _issue_request()

    async with semaphore:
        return await _issue_request()


async def run_prompt_batch(
    *,
    client: AsyncOpenAI,
    model_id: str,
    prompt_payloads: list[tuple[int, int, list[dict[str, str]], int | None]],
    max_tokens: int,
    temperature: float,
    n_generations: int,
    top_k: int,
    stop: list[str],
    submission_mode: str,
    prompt_concurrency: int,
) -> list[PromptRequestResult]:
    if submission_mode == "sequential":
        results: list[PromptRequestResult] = []
        for request_index, dataset_index, messages, prompt_tokens in prompt_payloads:
            results.append(
                await run_prompt_request(
                    client=client,
                    model_id=model_id,
                    messages=messages,
                    request_index=request_index,
                    dataset_index=dataset_index,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n_generations=n_generations,
                    top_k=top_k,
                    stop=stop,
                    semaphore=None,
                )
            )
        return results

    if submission_mode == "bounded_concurrent":
        semaphore = asyncio.Semaphore(prompt_concurrency)
        return await asyncio.gather(
            *[
                run_prompt_request(
                    client=client,
                    model_id=model_id,
                    messages=messages,
                    request_index=request_index,
                    dataset_index=dataset_index,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n_generations=n_generations,
                    top_k=top_k,
                    stop=stop,
                    semaphore=semaphore,
                )
                for request_index, dataset_index, messages, prompt_tokens in prompt_payloads
            ]
        )

    raise ValueError(f"Unknown submission_mode: {submission_mode}")


def score_prompt_request(result: PromptRequestResult, example: Any, env: MathEnv) -> dict[str, Any]:
    finish_reasons: Counter[str] = Counter()
    correct_count = 0
    format_count = 0

    for choice in result.choices:
        finish_reasons[choice.finish_reason] += 1
        if env.check_format(choice.text):
            format_count += 1
        if env.check_answer(choice.text, example.processed_answer):
            correct_count += 1

    return {
        "finish_reasons": {key: int(value) for key, value in sorted(finish_reasons.items())},
        "correct_count": correct_count,
        "format_count": format_count,
    }


def summarize_batch(
    *,
    batch_id: int,
    results: list[PromptRequestResult],
    batch_examples: list[Any],
    env: MathEnv,
    n_prompts: int,
    t_infer: float,
) -> dict[str, Any]:
    ok = [result for result in results if result.error is None]
    errors = [result for result in results if result.error is not None]

    request_latencies_ms = [1000.0 * result.e2e for result in ok]
    prompt_token_counts = [result.prompt_tokens for result in results if result.prompt_tokens is not None]
    total_samples = sum(len(result.choices) for result in ok)
    total_output_tokens = 0
    total_correct = 0
    total_format = 0
    finish_reasons: Counter[str] = Counter()

    reward_started = time.perf_counter()
    for result, example in zip(results, batch_examples, strict=True):
        if result.error is not None:
            continue
        scored = score_prompt_request(result, example, env)
        total_output_tokens += result.completion_tokens or 0
        total_correct += scored["correct_count"]
        total_format += scored["format_count"]
        finish_reasons.update(scored["finish_reasons"])
    reward_elapsed = time.perf_counter() - reward_started

    return {
        "batch_id": batch_id,
        "prompt_requests": n_prompts,
        "completion_samples": total_samples,
        "successful_requests": len(ok),
        "failed_requests": len(errors),
        "correct_count": total_correct,
        "format_count": total_format,
        "prompt_tokens_min": min(prompt_token_counts) if prompt_token_counts else None,
        "prompt_tokens_p50": percentile(prompt_token_counts, 50.0) if prompt_token_counts else None,
        "prompt_tokens_p95": percentile(prompt_token_counts, 95.0) if prompt_token_counts else None,
        "prompt_tokens_max": max(prompt_token_counts) if prompt_token_counts else None,
        "t_infer_s": t_infer,
        "t_reward_s": reward_elapsed,
        "t_total_s": t_infer + reward_elapsed,
        "prompt_req_per_s": len(ok) / t_infer if t_infer > 0 else 0.0,
        "completion_samples_per_s": total_samples / t_infer if t_infer > 0 else 0.0,
        "output_tok_per_s": total_output_tokens / t_infer if t_infer > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "mean_output_tokens_per_sample": total_output_tokens / total_samples if total_samples else 0.0,
        "p50_request_e2e_ms": percentile(request_latencies_ms, 50.0),
        "p95_request_e2e_ms": percentile(request_latencies_ms, 95.0),
        "p99_request_e2e_ms": percentile(request_latencies_ms, 99.0),
        "correctness_pct": 100.0 * total_correct / total_samples if total_samples else 0.0,
        "format_pct": 100.0 * total_format / total_samples if total_samples else 0.0,
        "finish_reasons": {key: int(value) for key, value in sorted(finish_reasons.items())},
        "errors": [result.error for result in errors[:10]],
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    env = MathEnv(seed=args.dataset_seed)
    warmup_indices, warmup_examples, measured_indices, measured_examples = choose_benchmark_examples(
        env,
        measured_count=args.num_batches * args.n_prompts,
        warmup_count=args.warmup_prompts,
        seed=args.dataset_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    engine_kwargs: dict[str, Any] = {
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.load_format is not None:
        engine_kwargs["load_format"] = args.load_format
    if not args.disable_enforce_eager:
        engine_kwargs["enforce_eager"] = True

    model = ModelConfig(name=args.model, path=None, engine_kwargs=engine_kwargs)
    with VllmEnvironment(
        model=model,
        mode="native",
        host="127.0.0.1",
        port=8000,
        timeout_seconds=7200,
        env_overrides={"MODEL_IMPL_TYPE": args.model_impl_type},
    ) as vllm_env:
        if vllm_env.model_id is None:
            raise RuntimeError("vLLM server did not expose a model id.")

        client = AsyncOpenAI(base_url=vllm_env.server_url, api_key="benchmark")

        warmup_started = time.perf_counter()
        if warmup_examples:
            warmup_payloads = [
                (
                    -args.warmup_prompts + index,
                    dataset_index,
                    build_prompt_messages(env, example),
                    count_prompt_tokens(tokenizer, build_prompt_messages(env, example)),
                )
                for index, (dataset_index, example) in enumerate(zip(warmup_indices, warmup_examples, strict=True))
            ]
            warmup_results = await run_prompt_batch(
                client=client,
                model_id=vllm_env.model_id,
                prompt_payloads=warmup_payloads,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                n_generations=args.n_generations,
                top_k=args.top_k,
                stop=args.stop,
                submission_mode=args.submission_mode,
                prompt_concurrency=args.prompt_concurrency,
            )
        else:
            warmup_results = []
        warmup_elapsed = time.perf_counter() - warmup_started

        batches: list[dict[str, Any]] = []
        total_started = time.perf_counter()

        for batch_id in range(args.num_batches):
            start = batch_id * args.n_prompts
            end = start + args.n_prompts
            batch_examples = measured_examples[start:end]
            batch_indices = measured_indices[start:end]
            prompt_payloads = [
                (
                    start + index,
                    dataset_index,
                    build_prompt_messages(env, example),
                    count_prompt_tokens(tokenizer, build_prompt_messages(env, example)),
                )
                for index, (dataset_index, example) in enumerate(zip(batch_indices, batch_examples, strict=True))
            ]

            infer_started = time.perf_counter()
            results = await run_prompt_batch(
                client=client,
                model_id=vllm_env.model_id,
                prompt_payloads=prompt_payloads,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                n_generations=args.n_generations,
                top_k=args.top_k,
                stop=args.stop,
                submission_mode=args.submission_mode,
                prompt_concurrency=args.prompt_concurrency,
            )
            infer_elapsed = time.perf_counter() - infer_started

            batch_summary = summarize_batch(
                batch_id=batch_id + 1,
                results=results,
                batch_examples=batch_examples,
                env=env,
                n_prompts=args.n_prompts,
                t_infer=infer_elapsed,
            )
            batches.append(batch_summary)
            print(
                json.dumps(
                    {
                        "benchmark_id": args.benchmark_id,
                        "event": "batch_complete",
                        "batch_id": batch_summary["batch_id"],
                        "t_infer_s": batch_summary["t_infer_s"],
                        "t_total_s": batch_summary["t_total_s"],
                        "prompt_req_per_s": batch_summary["prompt_req_per_s"],
                        "completion_samples_per_s": batch_summary["completion_samples_per_s"],
                        "output_tok_per_s": batch_summary["output_tok_per_s"],
                        "correctness_pct": batch_summary["correctness_pct"],
                        "format_pct": batch_summary["format_pct"],
                        "failed_requests": batch_summary["failed_requests"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        total_elapsed = time.perf_counter() - total_started
        await client.close()

    infer_times = [batch["t_infer_s"] for batch in batches]
    total_times = [batch["t_total_s"] for batch in batches]
    output_tok_rates = [batch["output_tok_per_s"] for batch in batches]
    sample_rates = [batch["completion_samples_per_s"] for batch in batches]
    prompt_rates = [batch["prompt_req_per_s"] for batch in batches]
    request_p50s = [batch["p50_request_e2e_ms"] for batch in batches if batch["p50_request_e2e_ms"] is not None]
    request_p95s = [batch["p95_request_e2e_ms"] for batch in batches if batch["p95_request_e2e_ms"] is not None]
    total_output_tokens = sum(batch["total_output_tokens"] for batch in batches)
    total_samples = sum(batch["completion_samples"] for batch in batches)
    total_correct = sum(batch["correct_count"] for batch in batches)
    total_format = sum(batch["format_count"] for batch in batches)
    finish_reasons = Counter()
    for batch in batches:
        finish_reasons.update(batch["finish_reasons"])

    return {
        "benchmark_id": args.benchmark_id,
        "config": {
            "model": args.model,
            "transport": "http",
            "prompt_submission_shape": args.submission_mode,
            "renderer_path": "hf_chat_template",
            "warmup_policy": "separate_fixed_slice",
            "model_impl_type": args.model_impl_type,
            "num_batches": args.num_batches,
            "n_prompts": args.n_prompts,
            "warmup_prompts": args.warmup_prompts,
            "n_generations": args.n_generations,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "stop": args.stop,
            "submission_mode": args.submission_mode,
            "prompt_concurrency": args.prompt_concurrency,
            "dataset_seed": args.dataset_seed,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "load_format": args.load_format,
            "enforce_eager": not args.disable_enforce_eager,
        },
        "prompt_indices": {
            "warmup": warmup_indices,
            "measured": measured_indices,
        },
        "warmup": {
            "elapsed_s": warmup_elapsed,
            "request_results": [asdict(result) for result in warmup_results],
        },
        "aggregate": {
            "total_measured_wall_s": total_elapsed,
            "mean_batch_infer_s": statistics.fmean(infer_times),
            "stddev_batch_infer_s": statistics.stdev(infer_times) if len(infer_times) > 1 else 0.0,
            "mean_batch_total_s": statistics.fmean(total_times),
            "stddev_batch_total_s": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            "mean_prompt_req_per_s": statistics.fmean(prompt_rates),
            "mean_completion_samples_per_s": statistics.fmean(sample_rates),
            "mean_output_tok_per_s": statistics.fmean(output_tok_rates),
            "p50_prompt_request_e2e_ms": percentile(request_p50s, 50.0),
            "p95_prompt_request_e2e_ms": percentile(request_p95s, 95.0),
            "total_output_tokens": total_output_tokens,
            "total_completion_samples": total_samples,
            "overall_correctness_pct": 100.0 * total_correct / total_samples if total_samples else 0.0,
            "overall_format_pct": 100.0 * total_format / total_samples if total_samples else 0.0,
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "projected_20_batches_s": statistics.fmean(total_times) * 20,
            "projected_100_batches_s": statistics.fmean(total_times) * 100,
            "projected_188_batches_s": statistics.fmean(total_times) * 188,
        },
        "batches": batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic GRPO rollout benchmark for Math RL.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--warmup-prompts", type=int, default=64)
    parser.add_argument("--n-generations", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--stop", nargs="*", default=DEFAULT_STOP)
    parser.add_argument("--submission-mode", choices=["sequential", "bounded_concurrent"], default="bounded_concurrent")
    parser.add_argument("--prompt-concurrency", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--model-impl-type", default="auto")
    parser.add_argument("--disable-enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with remove_tpu_lockfile_on_exit():
        report = asyncio.run(benchmark(args))
    print(f"{args.benchmark_id}_REPORT_START")
    print(json.dumps(report, sort_keys=True))
    print(f"{args.benchmark_id}_REPORT_END")


if __name__ == "__main__":
    main()
