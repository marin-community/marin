# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test: single llm.generate(all_prompts) call — bypasses HTTP/queue entirely.

Proves that LLM.generate() with a full batch is fast (like evalchemy),
and the throughput regression is from the queue serializing to batch-size-1.

Usage (via Iris):
    iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --memory 24GB --region us-central1 \
        --extra tpu --extra vllm --job-name vllm-batch-8b \
        -- python experiments/inference/exp_vllm_batch_test.py \
        --model gs://.../meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
        --num-prompts 50 --max-tokens 128 --max-model-len 4096
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

from iris.marin_fs import marin_prefix, open_url

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_inprocess import (
    _import_inprocess_vllm_symbols,
    _llm_kwargs,
    _load_and_inject_streaming,
    _resolve_bootstrap_model_source_for_start,
    _resolve_sync_weights_callable,
    evaluate_inprocess_eligibility,
)

logger = logging.getLogger(__name__)

_IRIS_LOG_FD = os.dup(1)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _iris_log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    line = f"I{ts} 0 vllm.batch {message}\n"
    os.write(_IRIS_LOG_FD, line.encode())


PROMPTS = [
    "Explain the concept of quantum computing in simple terms.",
    "Write a short paragraph about machine learning.",
    "What are the key differences between photosynthesis and blockchain?",
    "List 3 interesting facts about neural networks.",
    "Summarize the importance of climate change in one paragraph.",
    "How does DNA replication work? Give a brief explanation.",
    "What is the relationship between black holes and supply chain management?",
    "Describe natural language processing as if explaining to a 10-year-old.",
    "What are the pros and cons of game theory?",
    "Give a creative analogy for reinforcement learning.",
    "Explain the concept of protein folding in simple terms.",
    "Write a short paragraph about renewable energy.",
    "What are the key differences between cryptography and distributed systems?",
    "List 3 interesting facts about evolutionary biology.",
    "Summarize the importance of compiler design in one paragraph.",
    "How does ocean currents work? Give a brief explanation.",
    "What is the relationship between microprocessor architecture and gravitational waves?",
    "Describe epigenetics as if explaining to a 10-year-old.",
    "What are the pros and cons of topology?",
    "Give a creative analogy for fluid dynamics.",
    "Explain the concept of signal processing in simple terms.",
    "Write a short paragraph about operating systems.",
    "What are the key differences between thermodynamics and organic chemistry?",
    "List 3 interesting facts about number theory.",
    "Summarize the importance of robotics in one paragraph.",
    "How does immunology work? Give a brief explanation.",
    "What is the relationship between cloud computing and graph theory?",
    "Describe semiconductor physics as if explaining to a 10-year-old.",
    "What are the pros and cons of information theory?",
    "Give a creative analogy for plate tectonics.",
    "Explain the concept of group theory in simple terms.",
    "Write a short paragraph about computer vision.",
    "What are the key differences between statistical mechanics and algebra?",
    "List 3 interesting facts about robotics.",
    "Summarize the importance of signal processing in one paragraph.",
    "How does cryptography work? Give a brief explanation.",
    "Describe reinforcement learning as if explaining to a 10-year-old.",
    "What are the pros and cons of cloud computing?",
    "Give a creative analogy for compiler design.",
    "Explain the concept of distributed systems in simple terms.",
    "Write a short paragraph about graph theory.",
    "What are the key differences between epigenetics and plate tectonics?",
    "List 3 interesting facts about quantum computing.",
    "Summarize the importance of machine learning in one paragraph.",
    "How does game theory work? Give a brief explanation.",
    "Describe topology as if explaining to a 10-year-old.",
    "What are the pros and cons of neural networks?",
    "Give a creative analogy for DNA replication.",
    "Explain the concept of black holes in simple terms.",
    "Write a short paragraph about thermodynamics.",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch generate test — single llm.generate() call.")
    parser.add_argument("--model", required=True, help="GCS model path")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)

    from vllm import SamplingParams

    prompts = PROMPTS[: args.num_prompts]
    _iris_log(f"Batch test: {len(prompts)} prompts, max_tokens={args.max_tokens}")

    engine_kwargs: dict = {"max_model_len": args.max_model_len}
    if args.tensor_parallel_size is not None:
        engine_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True

    model = ModelConfig(name="batch-test", path=args.model, engine_kwargs=engine_kwargs)
    model_name_or_path = args.model

    # Resolve eligibility to get mapping and bootstrap
    eligibility = evaluate_inprocess_eligibility(model=model, model_name_or_path=model_name_or_path, extra_cli_args=None)
    if not eligibility.eligible:
        _iris_log(f"FAIL: not eligible: {eligibility.reason}")
        return 1

    mapping_model_name = eligibility.mapping_model_name
    llm_cls, _ = _import_inprocess_vllm_symbols()
    events: list[str] = []

    bootstrap_model_source, _bootstrap_local_dir = _resolve_bootstrap_model_source_for_start(model)
    _iris_log(f"Bootstrap: {bootstrap_model_source}")

    # Create skeleton
    t0 = time.time()
    llm = llm_cls(**_llm_kwargs(bootstrap_model_source=bootstrap_model_source, model=model))
    t_skeleton = time.time() - t0
    _iris_log(f"LLM skeleton created in {t_skeleton:.1f}s")

    # Inject weights
    sync_weights_fn = _resolve_sync_weights_callable(llm)
    t_weights = _load_and_inject_streaming(
        model_path=args.model,
        sync_weights_fn=sync_weights_fn,
        mapping_model_name=mapping_model_name,
        bootstrap_model_source=bootstrap_model_source,
        events=events,
    )
    _iris_log(f"Weights injected in {t_weights:.1f}s")

    # Single batch generate — the evalchemy way
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    _iris_log(f"Calling llm.generate({len(prompts)} prompts) — single batch call")
    t_gen_start = time.time()
    outputs = llm.generate(prompts, params)
    t_gen = time.time() - t_gen_start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / t_gen if t_gen > 0 else 0

    _iris_log("=== BATCH RESULTS ===")
    _iris_log(f"  prompts: {len(prompts)}")
    _iris_log(f"  total_tokens: {total_tokens}")
    _iris_log(f"  generation_time_sec: {t_gen:.1f}")
    _iris_log(f"  aggregate_tok_per_sec: {tok_per_sec:.1f}")
    _iris_log(f"  per_prompt_avg_sec: {t_gen / len(prompts):.2f}")

    # Save results
    model_name = args.model.rstrip("/").split("/")[-1]
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = f"{marin_prefix()}/inference/{model_name}/stress_test"

    results_path = f"{output_dir}/batch_results_{timestamp}.json"
    with open_url(results_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "timestamp": timestamp,
                "method": "single llm.generate() call (no HTTP, no queue)",
                "stats": {
                    "prompts": len(prompts),
                    "total_tokens": total_tokens,
                    "generation_time_sec": round(t_gen, 1),
                    "aggregate_tok_per_sec": round(tok_per_sec, 1),
                    "skeleton_time_sec": round(t_skeleton, 1),
                    "weight_pipeline_sec": round(t_weights, 1),
                },
            },
            f,
            indent=2,
        )
    _iris_log(f"Results saved to {results_path}")

    # Save samples
    samples_path = f"{output_dir}/batch_samples_{timestamp}.jsonl"
    with open_url(samples_path, "w") as f:
        for i, output in enumerate(outputs):
            sample = {
                "prompt": prompts[i],
                "response": output.outputs[0].text,
                "tokens_generated": len(output.outputs[0].token_ids),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    _iris_log(f"Samples saved to {samples_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
