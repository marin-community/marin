# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test IID vLLM setup on a dev TPU.

Run on a TPU worker with the Marin vllm extra available, for example:

    uv run --package marin --extra vllm python \
      experiments/downstream_scaling/evals/smoke_iid_vllm_tpu.py --model-key 1e22
"""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp

from experiments.downstream_scaling.evals.algorithms.iid import _load_vllm
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS

MARIN_EAST5_PREFIX = "gs://marin-us-east5"


def _patch_rpa_default_block_sizes_for_smoke() -> None:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    original = rpa_kernel.get_default_block_sizes
    if getattr(original, "_marin_iid_smoke_patched", False):
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

    patched_get_default_block_sizes._marin_iid_smoke_patched = True  # type: ignore[attr-defined]
    rpa_kernel.get_default_block_sizes = patched_get_default_block_sizes


def _rpa_mixed_block_sizes() -> dict[str, int]:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    return rpa_kernel.get_default_block_sizes(
        jnp.dtype("bfloat16"),
        jnp.dtype("bfloat16"),
        32,
        8,
        128,
        256,
        2048,
        1,
        4096 // 256,
        case=rpa_kernel.RpaCase.MIXED,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(DELPHI_CHECKPOINTS), default="1e22")
    parser.add_argument("--prompt", default="Question: What is 2 + 2?\nAnswer:")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    model_path = os.path.join(MARIN_EAST5_PREFIX, DELPHI_CHECKPOINTS[args.model_key])
    print(f"model_key={args.model_key}")
    print(f"model_path={model_path}")
    print(f"rpa_block_sizes_before={_rpa_mixed_block_sizes()}")
    _patch_rpa_default_block_sizes_for_smoke()
    print(f"rpa_block_sizes_after={_rpa_mixed_block_sizes()}")

    llm, SamplingParams = _load_vllm(model_path, seed=0)
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=1000,
            max_tokens=args.max_tokens,
        ),
        use_tqdm=False,
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
