# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the unified joint-decode TPU backend on a dev TPU.

Runs the modern protocol (paired decode with side-local tokens,
sliding-window admission, token-budget ramp) on two local chips via the
joint-decode package and the tpu-inference token-decision callback overlay.
Run on a TPU worker with the Marin vllm extra available, for example:

    uv sync --all-packages --extra=vllm
    uv run --no-sync python \
      experiments/downstream_scaling/evals/smoke_joint_decode_unified.py \
      --model-a <hf-checkpoint-path> --model-b <hf-checkpoint-path>

With the default budget of max_model_len + max_microbatch_size, the
coordinator admits one request per decision round, so the run exercises the
admission ramp as well as steady-state paired decode.
"""

from __future__ import annotations

import argparse
import logging
import tempfile

from joint_decode.config import JointDecodeSamplingConfig
from joint_decode.selection import select_top_rank
from joint_decode.tpu.config import JointDecodeConfig, JointDecodeModelConfig
from joint_decode.tpu.decoder import run_joint_decode

logger = logging.getLogger(__name__)

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The first person to walk on the moon was",
    "Two plus two equals",
    "The chemical symbol for gold is",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--chip-a", type=int, default=0)
    parser.add_argument("--chip-b", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-microbatch-size", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--barrier-timeout-s", type=float, default=600.0)
    parser.add_argument("--apply-rpa-block-size-patch", action="store_true")
    args = parser.parse_args()

    max_num_batched_tokens = (
        args.max_num_batched_tokens
        if args.max_num_batched_tokens is not None
        else args.max_model_len + args.max_microbatch_size
    )

    def model_config(model_path: str, chip: int) -> JointDecodeModelConfig:
        return JointDecodeModelConfig(
            model_path=model_path,
            chip_index=chip,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=None,
            enable_prefix_caching=False,
            apply_rpa_block_size_patch=args.apply_rpa_block_size_patch,
        )

    with tempfile.TemporaryDirectory(prefix="joint_decode_smoke_") as cache_dir:
        config = JointDecodeConfig(
            model_a=model_config(args.model_a, args.chip_a),
            model_b=model_config(args.model_b, args.chip_b),
            sampling=JointDecodeSamplingConfig(
                max_tokens_a=args.max_tokens,
                max_tokens_b=args.max_tokens,
                top_k_a=args.top_k,
                top_k_b=args.top_k,
                barrier_timeout_s=args.barrier_timeout_s,
                seed=0,
                stop=(),
                max_microbatch_size=args.max_microbatch_size,
                max_num_batched_tokens=max_num_batched_tokens,
            ),
            cache_dir=cache_dir,
        )
        outputs = run_joint_decode(config, PROMPTS, PROMPTS, select_token=select_top_rank)

    for prompt, output in zip(PROMPTS, outputs, strict=True):
        logger.info("prompt=%r finish=%s completion=%r", prompt, output.finish_reason, output.text)
    assert len(outputs) == len(PROMPTS)
    assert all(output.text for output in outputs)
    logger.info("joint-decode unified TPU smoke passed: %d completions", len(outputs))


if __name__ == "__main__":
    main()
