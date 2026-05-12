# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time one IID vLLM chunk on a dev TPU.

Run on a TPU worker with the Marin vllm extra available:

    uv run --package marin --extra vllm python \
      experiments/downstream_scaling/evals/time_iid_vllm_chunk.py --model-key 2e19
"""

from __future__ import annotations

import argparse
import os
import time

from marin.evaluation.utils import discover_hf_checkpoints

from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS

MARIN_EAST5_PREFIX = "gs://marin-us-east5"
VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


def _load_vllm_for_timing(model_path: str, seed: int, max_model_len: int):
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    from vllm import LLM, SamplingParams

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=seed,
        max_model_len=max_model_len,
    )
    return llm, SamplingParams


def _prompt(index: int) -> str:
    a = 1000 + index
    b = 2000 + index
    return f"Question: What is {a} + {b}? Show your work, then give the final answer.\nAnswer:"


def _num_generated_tokens(outputs) -> int:
    total = 0
    for output in outputs:
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(DELPHI_CHECKPOINTS), default="2e19")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    args = parser.parse_args()

    model_path = os.path.join(MARIN_EAST5_PREFIX, DELPHI_CHECKPOINTS[args.model_key])
    print(f"model_key={args.model_key}")
    print(f"model_path={model_path}")
    print(f"chunk_size={args.chunk_size}")
    print(f"max_tokens={args.max_tokens}")
    print(f"max_model_len={args.max_model_len}")

    load_start = time.perf_counter()
    llm, SamplingParams = _load_vllm_for_timing(
        model_path,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )
    load_seconds = time.perf_counter() - load_start

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    prompts = [_prompt(i) for i in range(args.chunk_size)]
    warmup_prompts = prompts[: args.warmup_prompts]
    warmup_start = time.perf_counter()
    llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
    warmup_seconds = time.perf_counter() - warmup_start

    generate_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generate_seconds = time.perf_counter() - generate_start
    generated_tokens = _num_generated_tokens(outputs)

    print(f"load_seconds={load_seconds:.3f}")
    print(f"warmup_seconds={warmup_seconds:.3f}")
    print(f"generate_seconds={generate_seconds:.3f}")
    print(f"requests_per_second={args.chunk_size / generate_seconds:.3f}")
    if generated_tokens:
        print(f"generated_tokens={generated_tokens}")
        print(f"generated_tokens_per_second={generated_tokens / generate_seconds:.3f}")


if __name__ == "__main__":
    main()
