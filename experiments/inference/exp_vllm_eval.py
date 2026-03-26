# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run lm-eval benchmarks on a model via vLLM on TPU.

Usage (via Iris):
    iris job run --tpu v6e-8 --memory 128GB --extra eval --extra tpu --extra vllm \
        -- python experiments/inference/exp_vllm_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tasks hellaswag,arc_easy \
        --num-fewshot 0
"""

import argparse
import json
import logging

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)


def run_eval(
    *,
    model_name_or_path: str,
    tasks: list[str],
    num_fewshot: int,
    max_model_len: int,
    limit: int | None,
    mode: str | None,
) -> dict:
    model = ModelConfig(
        name=model_name_or_path,
        path=None,
        engine_kwargs={"max_model_len": max_model_len},
    )

    env = VllmEnvironment(
        model=model,
        host="127.0.0.1",
        port=8000,
        timeout_seconds=3600,
        mode=mode,
    )

    with env:
        import lm_eval

        logger.info(f"Running lm-eval tasks: {tasks} ({num_fewshot}-shot)")

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=f"model={env.model_id},base_url={env.server_url}/completions,tokenizer_backend=huggingface,tokenized_requests=False",
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size="auto",
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"Model: {model_name_or_path}")
        print(f"Tasks: {tasks} ({num_fewshot}-shot)")
        if limit:
            print(f"Limit: {limit} examples per task")
        print("=" * 60)

        for task_name, task_results in results["results"].items():
            metrics = {
                k: v for k, v in task_results.items() if isinstance(v, (int, float)) and not k.endswith("_stderr")
            }
            print(f"\n{task_name}:")
            for metric, value in sorted(metrics.items()):
                stderr_key = f"{metric}_stderr"
                stderr = task_results.get(stderr_key)
                if stderr is not None:
                    print(f"  {metric}: {value:.4f} ± {stderr:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")

        print("\n" + "=" * 60)
        return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks via vLLM on TPU.")
    parser.add_argument("--model", required=True, help="HF model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval tasks (e.g. hellaswag,arc_easy)")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples (default: 0)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model sequence length (default: 4096)")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for quick testing)")
    parser.add_argument("--mode", choices=["docker", "native"], default=None, help="vLLM mode")
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",")]

    results = run_eval(
        model_name_or_path=args.model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        max_model_len=args.max_model_len,
        limit=args.limit,
        mode=args.mode,
    )

    print(
        json.dumps(
            {t: {k: v for k, v in r.items() if isinstance(v, (int, float))} for t, r in results["results"].items()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
