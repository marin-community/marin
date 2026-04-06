#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import jax
import jaxlib
import torch
from transformers import AutoConfig

from marin.rl.environments.math_env import MathEnv
from marin.rl.environments.inference_ctx.vllm import vLLMInferenceContext, vLLMInferenceContextConfig
from marin.utils import remove_tpu_lockfile_on_exit

import tpu_inference
from vllm import SamplingParams
import vllm

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_STOP = ["<|eot_id|>"]


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_model_impl(model_name: str, requested_model_impl: str) -> tuple[str, str | None]:
    if requested_model_impl != "auto":
        return requested_model_impl, None

    from tpu_inference.models.common.model_loader import _VLLM_PREFERRED_ARCHITECTURES

    config = AutoConfig.from_pretrained(model_name)
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1:
        raise ValueError(f"Expected exactly one architecture for {model_name}, got {architectures!r}")
    architecture = str(architectures[0])
    resolved = "vllm" if architecture in _VLLM_PREFERRED_ARCHITECTURES else "flax_nnx"
    return resolved, architecture


def collect_environment_census(args: argparse.Namespace) -> dict[str, Any]:
    resolved_model_impl, architecture = resolve_model_impl(args.model, args.model_impl_type)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": args.model,
        "model_impl_type_requested": args.model_impl_type,
        "model_impl_type_resolved": resolved_model_impl,
        "architecture": architecture,
        "vllm_version": package_version("vllm") or getattr(vllm, "__version__", None),
        "tpu_inference_version": package_version("tpu-inference") or getattr(tpu_inference, "__version__", None),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "libtpu_version": package_version("libtpu"),
        "torch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "triton_version": package_version("triton"),
    }


@dataclass(frozen=True)
class RequestRecord:
    batch_id: int
    request_index: int
    dataset_index: int
    prompt_tokens: int
    requested_max_tokens: int
    effective_max_tokens: int
    completion_samples: int
    completion_tokens: int
    correct_count: int
    format_count: int
    finish_reasons: dict[str, int]
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


def build_inference_context(args: argparse.Namespace) -> vLLMInferenceContext:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        n=args.n_generations,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        stop=args.stop,
        logprobs=1,
        include_stop_str_in_output=True,
    )
    return vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name=args.model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            sampling_params=sampling_params,
            load_format=args.load_format,
            enforce_eager=not args.disable_enforce_eager,
        )
    )


def determine_batch_max_tokens(
    *,
    prompt_token_counts: list[int],
    requested_max_tokens: int,
    max_model_len: int,
    mode: str,
) -> int:
    if mode == "strict-exp2039":
        return requested_max_tokens
    if mode == "fit-within-context":
        available_completion_budget = min(max_model_len - prompt_tokens - 1 for prompt_tokens in prompt_token_counts)
        return min(requested_max_tokens, available_completion_budget)
    raise ValueError(f"Unknown mode: {mode}")


def score_request(
    *,
    batch_id: int,
    request_index: int,
    dataset_index: int,
    example: Any,
    completion: Any,
    env: MathEnv,
    inference_ctx: vLLMInferenceContext,
    temperature: float,
    top_k: int,
    prompt_tokens: int,
    requested_max_tokens: int,
    effective_max_tokens: int,
) -> RequestRecord:
    correct_count = 0
    format_count = 0
    finish_reasons: Counter[str] = Counter()
    completion_tokens = 0

    usage = getattr(completion, "usage", None)
    if usage is not None and usage.completion_tokens is not None:
        completion_tokens = int(usage.completion_tokens)

    for choice in completion.choices:
        reward, format_valid, correct_answer = env._score_choice(
            example=example,
            response_text=choice.message.content or "",
            finish_reason=str(choice.finish_reason or "unknown"),
            tokenizer=inference_ctx.tokenizer,
        )
        rollout = inference_ctx.create_rollout_from_choice(
            prompt=example.processed_prompt,
            choice=choice,
            env_name="math",
            env_example_id=example.example_id,
            reward=reward,
            correctness_reward=correct_answer,
            temperature=temperature,
            top_k=top_k,
            system_prompt=None,
        )
        finish_reasons[str(choice.finish_reason or "unknown")] += 1
        correct_count += int(correct_answer)
        format_count += int(format_valid)
        if completion_tokens == 0:
            completion_tokens += int(rollout.response_tokens.size)

    return RequestRecord(
        batch_id=batch_id,
        request_index=request_index,
        dataset_index=dataset_index,
        prompt_tokens=prompt_tokens,
        requested_max_tokens=requested_max_tokens,
        effective_max_tokens=effective_max_tokens,
        completion_samples=len(completion.choices),
        completion_tokens=completion_tokens,
        correct_count=correct_count,
        format_count=format_count,
        finish_reasons=dict(sorted(finish_reasons.items())),
        error=None,
    )


def summarize_batch(
    *,
    batch_id: int,
    request_records: list[RequestRecord],
    t_infer: float,
    t_postprocess: float,
    requested_max_tokens: int,
    effective_max_tokens: int,
    prompt_token_counts: list[int],
    max_model_len: int,
    mode: str,
) -> dict[str, Any]:
    ok = [record for record in request_records if record.error is None]
    errors = [record for record in request_records if record.error is not None]

    total_samples = sum(record.completion_samples for record in ok)
    total_output_tokens = sum(record.completion_tokens for record in ok)
    total_correct = sum(record.correct_count for record in ok)
    total_format = sum(record.format_count for record in ok)
    finish_reasons: Counter[str] = Counter()
    for record in ok:
        finish_reasons.update(record.finish_reasons)

    would_overflow_count = sum(
        1 for prompt_tokens in prompt_token_counts if prompt_tokens + requested_max_tokens > max_model_len
    )
    prompt_too_long_count = sum(1 for prompt_tokens in prompt_token_counts if prompt_tokens >= max_model_len)

    return {
        "batch_id": batch_id,
        "mode": mode,
        "prompt_requests": len(request_records),
        "successful_requests": len(ok),
        "failed_requests": len(errors),
        "completion_samples": total_samples,
        "correct_count": total_correct,
        "format_count": total_format,
        "t_infer_s": t_infer,
        "t_postprocess_s": t_postprocess,
        "t_total_s": t_infer + t_postprocess,
        "requested_max_tokens": requested_max_tokens,
        "effective_max_tokens": effective_max_tokens,
        "would_overflow_count": would_overflow_count,
        "prompt_too_long_count": prompt_too_long_count,
        "prompt_tokens_min": min(prompt_token_counts),
        "prompt_tokens_p50": percentile(prompt_token_counts, 50.0),
        "prompt_tokens_p95": percentile(prompt_token_counts, 95.0),
        "prompt_tokens_max": max(prompt_token_counts),
        "prompt_req_per_s": len(ok) / t_infer if t_infer > 0 else 0.0,
        "completion_samples_per_s": total_samples / t_infer if t_infer > 0 else 0.0,
        "output_tok_per_s": total_output_tokens / t_infer if t_infer > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "mean_output_tokens_per_sample": total_output_tokens / total_samples if total_samples else 0.0,
        "correctness_pct": 100.0 * total_correct / total_samples if total_samples else 0.0,
        "format_pct": 100.0 * total_format / total_samples if total_samples else 0.0,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "errors": [record.error for record in errors[:10]],
    }


def run_batch(
    *,
    batch_id: int,
    batch_examples: list[Any],
    batch_dataset_indices: list[int],
    env: MathEnv,
    inference_ctx: vLLMInferenceContext,
    args: argparse.Namespace,
    start_request_index: int,
) -> tuple[dict[str, Any], list[RequestRecord]]:
    prompt_messages = [build_prompt_messages(env, example) for example in batch_examples]
    prompt_token_counts = [len(inference_ctx._render_messages_to_tokens(messages)) for messages in prompt_messages]
    effective_max_tokens = determine_batch_max_tokens(
        prompt_token_counts=prompt_token_counts,
        requested_max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        mode=args.mode,
    )

    t_infer_started = time.perf_counter()
    batch_exception: str | None = None
    completions: list[Any] = []
    if effective_max_tokens < 1:
        batch_exception = (
            f"no_valid_completion_budget(requested_max_tokens={args.max_tokens}, "
            f"effective_max_tokens={effective_max_tokens})"
        )
    else:
        try:
            completions = inference_ctx.batch_completions(
                prompts=prompt_messages,
                temperature=args.temperature,
                n=args.n_generations,
                max_tokens=effective_max_tokens,
                top_k=args.top_k,
                stop=args.stop,
            )
        except Exception as exc:
            batch_exception = repr(exc)
    t_infer = time.perf_counter() - t_infer_started

    t_postprocess_started = time.perf_counter()
    request_records: list[RequestRecord] = []
    if batch_exception is not None:
        for index, prompt_tokens in enumerate(prompt_token_counts):
            request_records.append(
                RequestRecord(
                    batch_id=batch_id,
                    request_index=start_request_index + index,
                    dataset_index=batch_dataset_indices[index],
                    prompt_tokens=prompt_tokens,
                    requested_max_tokens=args.max_tokens,
                    effective_max_tokens=effective_max_tokens,
                    completion_samples=0,
                    completion_tokens=0,
                    correct_count=0,
                    format_count=0,
                    finish_reasons={},
                    error=batch_exception,
                )
            )
    else:
        for index, (example, completion, prompt_tokens, dataset_index) in enumerate(
            zip(batch_examples, completions, prompt_token_counts, batch_dataset_indices, strict=True)
        ):
            request_records.append(
                score_request(
                    batch_id=batch_id,
                    request_index=start_request_index + index,
                    dataset_index=dataset_index,
                    example=example,
                    completion=completion,
                    env=env,
                    inference_ctx=inference_ctx,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    prompt_tokens=prompt_tokens,
                    requested_max_tokens=args.max_tokens,
                    effective_max_tokens=effective_max_tokens,
                )
            )
    t_postprocess = time.perf_counter() - t_postprocess_started

    return (
        summarize_batch(
            batch_id=batch_id,
            request_records=request_records,
            t_infer=t_infer,
            t_postprocess=t_postprocess,
            requested_max_tokens=args.max_tokens,
            effective_max_tokens=effective_max_tokens,
            prompt_token_counts=prompt_token_counts,
            max_model_len=args.max_model_len,
            mode=args.mode,
        ),
        request_records,
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    env = MathEnv(seed=args.dataset_seed)
    os.environ["MODEL_IMPL_TYPE"] = args.model_impl_type
    environment = collect_environment_census(args)
    print(
        json.dumps(
            {
                "benchmark_id": args.benchmark_id,
                "event": "environment_census",
                **environment,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    warmup_indices, warmup_examples, measured_indices, measured_examples = choose_benchmark_examples(
        env,
        measured_count=args.num_batches * args.n_prompts,
        warmup_count=args.warmup_prompts,
        seed=args.dataset_seed,
    )

    inference_ctx = build_inference_context(args)
    try:
        warmup_summary: dict[str, Any]
        warmup_records: list[RequestRecord]
        if warmup_examples:
            warmup_summary, warmup_records = run_batch(
                batch_id=0,
                batch_examples=warmup_examples,
                batch_dataset_indices=warmup_indices,
                env=env,
                inference_ctx=inference_ctx,
                args=args,
                start_request_index=-args.warmup_prompts,
            )
        else:
            warmup_summary = {
                "batch_id": 0,
                "mode": args.mode,
                "prompt_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "completion_samples": 0,
                "correct_count": 0,
                "format_count": 0,
                "t_infer_s": 0.0,
                "t_postprocess_s": 0.0,
                "t_total_s": 0.0,
                "requested_max_tokens": args.max_tokens,
                "effective_max_tokens": args.max_tokens,
                "would_overflow_count": 0,
                "prompt_too_long_count": 0,
                "prompt_tokens_min": None,
                "prompt_tokens_p50": None,
                "prompt_tokens_p95": None,
                "prompt_tokens_max": None,
                "prompt_req_per_s": 0.0,
                "completion_samples_per_s": 0.0,
                "output_tok_per_s": 0.0,
                "total_output_tokens": 0,
                "mean_output_tokens_per_sample": 0.0,
                "correctness_pct": 0.0,
                "format_pct": 0.0,
                "finish_reasons": {},
                "errors": [],
            }
            warmup_records = []
        print(
            json.dumps(
                {
                    "benchmark_id": args.benchmark_id,
                    "event": "warmup_complete",
                    "effective_max_tokens": warmup_summary["effective_max_tokens"],
                    "t_infer_s": warmup_summary["t_infer_s"],
                    "t_total_s": warmup_summary["t_total_s"],
                    "failed_requests": warmup_summary["failed_requests"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        batches: list[dict[str, Any]] = []
        request_records: list[RequestRecord] = []
        total_started = time.perf_counter()

        for batch_id in range(args.num_batches):
            start = batch_id * args.n_prompts
            end = start + args.n_prompts
            batch_examples = measured_examples[start:end]
            batch_dataset_indices = measured_indices[start:end]

            batch_summary, batch_records = run_batch(
                batch_id=batch_id + 1,
                batch_examples=batch_examples,
                batch_dataset_indices=batch_dataset_indices,
                env=env,
                inference_ctx=inference_ctx,
                args=args,
                start_request_index=start,
            )
            batches.append(batch_summary)
            request_records.extend(batch_records)
            print(
                json.dumps(
                    {
                        "benchmark_id": args.benchmark_id,
                        "event": "batch_complete",
                        "batch_id": batch_summary["batch_id"],
                        "effective_max_tokens": batch_summary["effective_max_tokens"],
                        "t_infer_s": batch_summary["t_infer_s"],
                        "t_total_s": batch_summary["t_total_s"],
                        "prompt_req_per_s": batch_summary["prompt_req_per_s"],
                        "completion_samples_per_s": batch_summary["completion_samples_per_s"],
                        "output_tok_per_s": batch_summary["output_tok_per_s"],
                        "correctness_pct": batch_summary["correctness_pct"],
                        "format_pct": batch_summary["format_pct"],
                        "failed_requests": batch_summary["failed_requests"],
                        "would_overflow_count": batch_summary["would_overflow_count"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        total_elapsed = time.perf_counter() - total_started
    finally:
        inference_ctx.shutdown()

    infer_times = [batch["t_infer_s"] for batch in batches]
    total_times = [batch["t_total_s"] for batch in batches]
    output_tok_rates = [batch["output_tok_per_s"] for batch in batches]
    sample_rates = [batch["completion_samples_per_s"] for batch in batches]
    prompt_rates = [batch["prompt_req_per_s"] for batch in batches]
    total_output_tokens = sum(batch["total_output_tokens"] for batch in batches)
    total_samples = sum(batch["completion_samples"] for batch in batches)
    total_correct = sum(batch["correct_count"] for batch in batches)
    total_format = sum(batch["format_count"] for batch in batches)
    total_failed_requests = sum(batch["failed_requests"] for batch in batches)
    finish_reasons = Counter()
    for batch in batches:
        finish_reasons.update(batch["finish_reasons"])

    return {
        "benchmark_id": args.benchmark_id,
        "config": {
            "model": args.model,
            "mode": args.mode,
            "transport": "inprocess",
            "prompt_submission_shape": "single_batch",
            "renderer_path": "marin_renderer",
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
            "dataset_seed": args.dataset_seed,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "load_format": args.load_format,
            "enforce_eager": not args.disable_enforce_eager,
        },
        "environment": environment,
        "prompt_indices": {
            "warmup": warmup_indices,
            "measured": measured_indices,
        },
        "warmup": {
            "summary": warmup_summary,
            "failed_requests": warmup_summary["failed_requests"],
            "request_records": [asdict(record) for record in warmup_records],
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
            "total_output_tokens": total_output_tokens,
            "total_completion_samples": total_samples,
            "total_failed_requests": total_failed_requests,
            "overall_correctness_pct": 100.0 * total_correct / total_samples if total_samples else 0.0,
            "overall_format_pct": 100.0 * total_format / total_samples if total_samples else 0.0,
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "projected_20_batches_s": statistics.fmean(total_times) * 20,
            "projected_50_batches_s": statistics.fmean(total_times) * 50,
            "projected_100_batches_s": statistics.fmean(total_times) * 100,
        },
        "batches": batches,
        "request_records": [asdict(record) for record in request_records],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-process GRPO rollout benchmark for Marin RL vLLM.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["strict-exp2039", "fit-within-context"], default="strict-exp2039")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--warmup-prompts", type=int, default=64)
    parser.add_argument("--n-generations", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--stop", nargs="*", default=DEFAULT_STOP)
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
        report = benchmark(args)
    print(f"{args.benchmark_id}_REPORT_START")
    print(json.dumps(report, sort_keys=True))
    print(f"{args.benchmark_id}_REPORT_END")


if __name__ == "__main__":
    main()
