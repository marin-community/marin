# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline comparison: run the SAME prompts from a prior stress test through
the subprocess vLLM path (runai_streamer) to get an apples-to-apples comparison.

Reads prompts from an existing samples JSONL on GCS, sends them through the old
subprocess path, and saves results alongside the original.

Usage (via Iris):
    iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --memory 24GB --region us-central1 \
        --extra tpu --extra vllm --job-name vllm-baseline-8b \
        -- python experiments/inference/exp_vllm_baseline_comparison.py \
        --model gs://.../meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
        --samples-jsonl gs://.../stress_test/samples_20260318-040743.jsonl \
        --max-model-len 4096 --max-concurrent 4
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from iris.marin_fs import marin_prefix, open_url

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)

_IRIS_LOG_FD = os.dup(1)


def _iris_log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    line = f"I{ts} 0 vllm.baseline {message}\n"
    os.write(_IRIS_LOG_FD, line.encode())


def load_prompts_from_gcs(samples_jsonl: str) -> list[str]:
    """Read prompts from an existing samples JSONL on GCS."""
    from iris.marin_fs import open_url

    prompts = []
    with open_url(samples_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            prompts.append(sample["prompt"])
    return prompts


def send_request(server_url: str, model_id: str, prompt: str, max_tokens: int, request_id: int) -> dict:
    start = time.time()
    resp = requests.post(
        f"{server_url}/completions",
        json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    tokens_generated = data["usage"]["completion_tokens"]
    response_text = data["choices"][0]["text"]
    return {
        "elapsed": elapsed,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
        "response": response_text,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Baseline comparison using subprocess vLLM path.")
    parser.add_argument("--model", required=True, help="GCS model path")
    parser.add_argument("--samples-jsonl", required=True, help="GCS path to existing samples JSONL (for prompts)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args(argv)

    # Load the same prompts used in the in-process stress test
    _iris_log(f"Loading prompts from {args.samples_jsonl}")
    prompts = load_prompts_from_gcs(args.samples_jsonl)
    _iris_log(f"Loaded {len(prompts)} prompts")

    # Force subprocess path by setting load_format=runai_streamer
    # This makes evaluate_inprocess_eligibility return ineligible,
    # so VllmEnvironment uses the old NativeVllmServerBackend (subprocess).
    model = ModelConfig(
        name="baseline-comparison",
        path=args.model,
        engine_kwargs={
            "max_model_len": args.max_model_len,
            "load_format": "runai_streamer",
        },
    )

    env = VllmEnvironment(
        model=model,
        host="127.0.0.1",
        port=8000,
        timeout_seconds=3600,
        mode="native",
    )

    t_env_start = time.time()
    with env:
        t_server_ready = time.time() - t_env_start
        _iris_log(f"Subprocess vLLM server ready in {t_server_ready:.1f}s (runai_streamer path)")
        _iris_log(f"Baseline: prompts={len(prompts)} concurrent={args.max_concurrent} max_tokens={args.max_tokens}")

        results = []
        errors = 0
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
            futures = {
                executor.submit(send_request, env.server_url, env.model_id, p, args.max_tokens, i): i
                for i, p in enumerate(prompts)
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Request failed: {e}")

                if completed % 10 == 0 or completed == len(prompts):
                    elapsed = time.time() - total_start
                    rps = completed / elapsed if elapsed > 0 else 0
                    _iris_log(f"[{completed}/{len(prompts)}] {elapsed:.1f}s elapsed, {rps:.1f} req/s, {errors} errors")

        total_elapsed = time.time() - total_start

        if results:
            latencies = sorted(r["elapsed"] for r in results)
            tps_values = [r["tokens_per_sec"] for r in results]
            total_tokens = sum(r["tokens_generated"] for r in results)
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]

            stats = {
                "total_prompts": len(prompts),
                "successful": len(results),
                "errors": errors,
                "server_startup_sec": round(t_server_ready, 1),
                "total_time_sec": round(total_elapsed, 1),
                "requests_per_sec": round(len(results) / total_elapsed, 2),
                "total_tokens_generated": total_tokens,
                "aggregate_tokens_per_sec": round(total_tokens / total_elapsed, 1),
                "latency_p50_sec": round(p50, 3),
                "latency_p95_sec": round(p95, 3),
                "avg_tokens_per_sec_per_request": round(sum(tps_values) / len(tps_values), 1),
                "path": "subprocess (runai_streamer)",
            }
        else:
            stats = {
                "total_prompts": len(prompts),
                "successful": 0,
                "errors": errors,
                "server_startup_sec": round(t_server_ready, 1),
                "path": "subprocess (runai_streamer)",
            }

        _iris_log("=== BASELINE RESULTS ===")
        for k, v in stats.items():
            _iris_log(f"  {k}: {v}")

        # Save to GCS
        model_name = args.model.rstrip("/").split("/")[-1]
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = f"{marin_prefix()}/inference/{model_name}/stress_test"

        results_path = f"{output_dir}/baseline_results_{timestamp}.json"
        with open_url(results_path, "w") as f:
            json.dump(
                {"model": args.model, "timestamp": timestamp, "stats": stats, "source_samples": args.samples_jsonl},
                f,
                indent=2,
            )
        _iris_log(f"Results saved to {results_path}")

        samples_path = f"{output_dir}/baseline_samples_{timestamp}.jsonl"
        with open_url(samples_path, "w") as f:
            for i, result in enumerate(results):
                sample = {
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "response": result.get("response", ""),
                    "elapsed_sec": result.get("elapsed"),
                    "tokens_generated": result.get("tokens_generated"),
                    "tokens_per_sec": result.get("tokens_per_sec"),
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        _iris_log(f"Samples ({len(results)} rows) saved to {samples_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
