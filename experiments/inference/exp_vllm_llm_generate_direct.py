# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct LLM.generate() benchmark — no HTTP server, no fsspec bypass.

Uses vLLM's standard LLM() constructor with the default weight loading path
(runai_streamer on TPU). This tests the same code path as `vllm serve` but
without subprocess/HTTP overhead, matching the original experiment's surface.

Usage (8B on v6e-4, same-region model):
    uv run iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v6e-4 --memory 64GB \
        --extra tpu --extra vllm \
        --job-name vllm-sr-llm-direct \
        --no-wait \
        -e MODEL_IMPL_TYPE auto \
        -- python experiments/inference/exp_vllm_llm_generate_direct.py \
        --model gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f
"""

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model path (gs://...) or model name for auto-regional resolution",
    )
    parser.add_argument("--prompt", default="Write a short haiku about TPUs.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--load-format", default="runai_streamer")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    # Auto-resolve model path to same-region bucket if not a full gs:// path
    model_path = args.model
    if not model_path.startswith("gs://") and not model_path.startswith("/"):
        from experiments.inference.regional_model_path import resolve_regional_model_path

        model_path = resolve_regional_model_path(model_path)

    print(f"MODEL_IMPL_TYPE={os.environ.get('MODEL_IMPL_TYPE', '<not set>')}")
    print(f"model={model_path}")
    print(f"load_format={args.load_format}")
    print(f"tp={args.tp}")
    sys.stdout.flush()

    from vllm import LLM, SamplingParams

    t_total_start = time.time()

    # Create LLM — this does model loading through tpu-inference's get_model()
    t0 = time.time()
    llm = LLM(
        model=model_path,
        load_format=args.load_format,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
    )
    t_load = time.time() - t0
    print(f"[phase] LLM() constructor: {t_load:.1f}s")
    sys.stdout.flush()

    # Generate
    t0 = time.time()
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.7),
    )
    t_gen = time.time() - t0

    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    tps = tokens / t_gen if t_gen > 0 else 0

    t_total = time.time() - t_total_start

    print(f"[phase] generate: {t_gen:.1f}s ({tokens} tokens, {tps:.1f} tok/s)")
    print(f"[phase] TOTAL: {t_total:.1f}s (load={t_load:.1f} gen={t_gen:.1f})")
    print(f"Output: {text}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
