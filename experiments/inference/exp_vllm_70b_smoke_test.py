# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: in-process vLLM with Levanter fsspec on Llama 3.3 70B.

Validates fast weight loading + sync_weights on a 70B model, then runs
a handful of prompts to verify correctness and measure throughput.

Usage (via Iris, us-central1 v5p-8 — 448 GiB host RAM, 760 GiB HBM):
    uv run iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --region us-central1 \
        --extra tpu --extra vllm \
        --job-name vllm-70b-smoke \
        -- python experiments/inference/exp_vllm_70b_smoke_test.py
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

_REAL_STDOUT = sys.stdout

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b"

TEST_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a haiku about tensor processing units.",
    "Explain gradient descent in two sentences.",
    "Translate 'hello world' into Spanish, French, and Japanese.",
    "What is 17 * 23? Show your work briefly.",
]


def _iris_log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    _REAL_STDOUT.write(f"I{ts} 0 vllm.70b.smoke {message}\n")
    _REAL_STDOUT.flush()


def main():
    parser = argparse.ArgumentParser(description="70B in-process vLLM smoke test")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="GCS path to model")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default 4 for 70B)")
    args = parser.parse_args()

    _iris_log(f"Starting 70B smoke test with model={args.model}")

    import numpy as np
    from vllm import LLM, SamplingParams

    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_inprocess import (
        _resolve_bootstrap_model_source_for_start,
        _resolve_mapping_model_name,
        load_safetensors_from_remote,
    )
    from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
    from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu

    model = ModelConfig(
        name="70b-smoke-test",
        path=args.model,
        engine_kwargs={"max_model_len": args.max_model_len},
    )

    timings: dict[str, float] = {}

    # Stage 1: Bootstrap (config.json + tokenizer from GCS → local tmpdir)
    t0 = time.time()
    bootstrap_source, bootstrap_local_dir = _resolve_bootstrap_model_source_for_start(model)
    timings["bootstrap"] = time.time() - t0
    _iris_log(f"[1/6] Bootstrap resolved in {timings['bootstrap']:.1f}s: {bootstrap_source}")

    # Stage 2: LLM skeleton with dummy weights
    _iris_log("[2/6] Creating LLM skeleton (load_format=dummy, enforce_eager=True)...")
    t0 = time.time()
    llm = LLM(
        model=bootstrap_source,
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
    )
    timings["skeleton"] = time.time() - t0
    _iris_log(f"[2/6] LLM skeleton created in {timings['skeleton']:.1f}s")

    # Stage 3: Load weights from GCS via Levanter fsspec
    _iris_log(f"[3/6] Loading safetensors from {args.model} ...")
    t0 = time.time()
    state_dict = load_safetensors_from_remote(args.model)
    timings["load"] = time.time() - t0
    num_tensors = len(state_dict)
    total_bytes = sum(np.array(v).nbytes for v in state_dict.values())
    total_gib = total_bytes / (1024**3)
    throughput_mibs = (total_bytes / (1024**2)) / timings["load"] if timings["load"] > 0 else 0
    _iris_log(
        f"[3/6] Loaded {num_tensors} tensors ({total_gib:.1f} GiB) in {timings['load']:.1f}s "
        f"({throughput_mibs:.0f} MiB/s)"
    )

    # Stage 4: Reshape HF 2D attention → Levanter 3D
    import json as json_mod

    config_path = os.path.join(bootstrap_source, "config.json")
    with open(config_path) as f:
        model_config = json_mod.load(f)
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    head_dim = model_config["hidden_size"] // num_heads

    t0 = time.time()
    reshaped = 0
    for key in list(state_dict.keys()):
        if "q_proj" in key and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(num_heads, head_dim, -1)
            reshaped += 1
        elif ("k_proj" in key or "v_proj" in key) and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(num_kv_heads, head_dim, -1)
            reshaped += 1
        elif "o_proj" in key and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(-1, num_heads, head_dim)
            reshaped += 1
    timings["reshape"] = time.time() - t0
    _iris_log(f"[4/6] Reshaped {reshaped} attention projections in {timings['reshape']:.1f}s")

    # Stage 5: Convert to NNX + inject via sync_weights
    mapping_name = _resolve_mapping_model_name(model, args.model)
    _iris_log(f"[5/6] Converting + injecting weights (mapping={mapping_name})...")

    t0 = time.time()
    nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)
    timings["convert"] = time.time() - t0
    _iris_log(f"[5/6] NNX conversion: {timings['convert']:.1f}s")

    t0 = time.time()
    llm.llm_engine.model_executor.driver_worker.sync_weights(
        nnx_state,
        mappings=MODEL_MAPPINGS[mapping_name],
        transpose_keys=MODEL_TRANSPOSE_KEYS[mapping_name],
        reshard_fn=None,
    )
    timings["inject"] = time.time() - t0
    _iris_log(f"[5/6] Weight injection: {timings['inject']:.1f}s")

    weight_pipeline = timings["load"] + timings["reshape"] + timings["convert"] + timings["inject"]
    _iris_log(
        f"[5/6] WEIGHT PIPELINE TOTAL: {weight_pipeline:.1f}s "
        f"(load={timings['load']:.1f} reshape={timings['reshape']:.1f} "
        f"convert={timings['convert']:.1f} inject={timings['inject']:.1f})"
    )

    # Stage 6: Generate from test prompts
    _iris_log(f"[6/6] Running {len(TEST_PROMPTS)} test prompts (max_tokens={args.max_tokens})...")
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.7)

    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, params)
    timings["generate"] = time.time() - t0

    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    gen_tps = total_gen_tokens / timings["generate"] if timings["generate"] > 0 else 0
    _iris_log(
        f"[6/6] Generated {total_gen_tokens} tokens across {len(TEST_PROMPTS)} prompts "
        f"in {timings['generate']:.1f}s ({gen_tps:.1f} tok/s)"
    )

    # Print results
    print(f"\n{'='*70}")
    print("70B IN-PROCESS vLLM SMOKE TEST RESULTS")
    print(f"{'='*70}")
    print(f"Model:         {args.model}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Tensors:       {num_tensors} ({total_gib:.1f} GiB)")
    print(f"{'='*70}")
    print("TIMING BREAKDOWN:")
    print(f"  Bootstrap:      {timings['bootstrap']:>8.1f}s")
    print(f"  LLM skeleton:   {timings['skeleton']:>8.1f}s")
    print(f"  Weight load:    {timings['load']:>8.1f}s  ({throughput_mibs:.0f} MiB/s)")
    print(f"  Reshape:        {timings['reshape']:>8.1f}s")
    print(f"  NNX convert:    {timings['convert']:>8.1f}s")
    print(f"  Weight inject:  {timings['inject']:>8.1f}s")
    print(f"  Generation:     {timings['generate']:>8.1f}s  ({gen_tps:.1f} tok/s)")
    print("  ---")
    print(f"  Weight pipeline:{weight_pipeline:>8.1f}s  (load+reshape+convert+inject)")
    total = sum(timings.values())
    print(f"  Total:          {total:>8.1f}s")
    print(f"{'='*70}")

    for i, (prompt, output) in enumerate(zip(TEST_PROMPTS, outputs, strict=True)):
        text = output.outputs[0].text.strip()
        tokens = len(output.outputs[0].token_ids)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"  [{tokens} tokens] {text[:200]}")

    print(f"\n{'='*70}")
    _iris_log("SUCCESS")

    # Cleanup
    if bootstrap_local_dir:
        import shutil

        shutil.rmtree(bootstrap_local_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
