# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Boot-smoke the vLLM TPU stack: load one model and greedy-generate.

Deliberately self-contained — no experiments/algorithms imports — so it
tests exactly the vllm + tpu-inference engine stack and nothing built on
top of it. Run on a TPU worker with the Marin vllm extra available:

    uv run python experiments/downstream_scaling/evals/smoke_vllm_tpu_boot.py \\
      --model-path "gs://marin-us-east5/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf/step-37334"

The model path must point directly at an HF checkpoint directory
(config.json at the top level).
"""

from __future__ import annotations

import argparse
import json
import os

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


def _patch_rpa_default_block_sizes() -> None:
    # Halve the RPA block sizes; required for delphi-shaped models (vmem
    # limits), harmless for a smoke on other models.
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    original = rpa_kernel.get_default_block_sizes
    if getattr(original, "_marin_boot_smoke_patched", False):
        return

    def patched_get_default_block_sizes(*args, **kwargs):
        sizes = dict(original(*args, **kwargs))
        case = kwargs.get("case")
        if case is not rpa_kernel.RpaCase.DECODE:
            page_size = args[5]
            sizes["bq_sz"] = max(1, sizes["bq_sz"] // 2)
            sizes["bq_csz"] = max(1, sizes["bq_csz"] // 2)
            sizes["bkv_sz"] = max(page_size, sizes["bkv_sz"] // 2)
            sizes["bkv_csz"] = max(page_size, sizes["bkv_csz"] // 2)
        return sizes

    patched_get_default_block_sizes._marin_boot_smoke_patched = True  # type: ignore[attr-defined]
    rpa_kernel.get_default_block_sizes = patched_get_default_block_sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Question: What is 2 + 2?\nAnswer:")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)
    _patch_rpa_default_block_sizes()

    import fsspec
    from vllm import LLM, SamplingParams

    with fsspec.open(f"{args.model_path}/config.json", "r") as f:
        n_heads = json.load(f)["num_attention_heads"]
    tensor_parallel_size = 2 if n_heads % 2 == 0 else 1
    print(f"model_path={args.model_path} n_heads={n_heads} tp={tensor_parallel_size}")

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=args.seed,
        tensor_parallel_size=tensor_parallel_size,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=1000, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0].text
    print(f"completion={completion!r}")
    assert completion, "empty completion"
    print("vllm TPU boot smoke passed")


if __name__ == "__main__":
    main()
