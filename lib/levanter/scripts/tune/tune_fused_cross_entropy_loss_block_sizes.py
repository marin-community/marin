# Copyright 2026 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import os
import time

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)

_SKIP_LABEL_LOGITS_ENV = "LEVANTER_PALLAS_TPU_SKIP_LABEL_LOGITS_BENCH"
_FWD_XW_BF16_ENV = "LEVANTER_PALLAS_TPU_FWD_XW_BF16_BENCH"
_FWD_DOT_ACCUM_BF16_ENV = "LEVANTER_PALLAS_TPU_FWD_DOT_ACCUM_BF16_BENCH"
_FWD_FULL_H_DOT_ENV = "LEVANTER_PALLAS_TPU_FWD_FULL_H_DOT_BENCH"
_FWD_SPLIT_LABEL_DOT_ENV = "LEVANTER_PALLAS_TPU_FWD_SPLIT_LABEL_DOT_BENCH"
_FWD_SPLIT_LABEL_BF16_MUL_ENV = "LEVANTER_PALLAS_TPU_FWD_SPLIT_LABEL_BF16_MUL_BENCH"
_FWD_SPLIT_LABEL_PALLAS_ENV = "LEVANTER_PALLAS_TPU_FWD_SPLIT_LABEL_PALLAS_BENCH"
_FWD_INLINE_LABEL_SCALAR_ENV = "LEVANTER_PALLAS_TPU_FWD_INLINE_LABEL_SCALAR_BENCH"
_FWD_INLINE_LABEL_TAKE_ENV = "LEVANTER_PALLAS_TPU_FWD_INLINE_LABEL_TAKE_BENCH"
_FWD_LSE_FORI_LOOP_ENV = "LEVANTER_PALLAS_TPU_FWD_LSE_FORI_LOOP_BENCH"
_FWD_LSE_FORI_V_MULT_ENV = "LEVANTER_PALLAS_TPU_FWD_LSE_FORI_V_MULT_BENCH"
_FWD_LSE_STORE_PATH_ENV = "LEVANTER_PALLAS_TPU_FWD_LSE_STORE_PATH_BENCH"
_MAX_ERROR_CHARS = 600


@dataclass(frozen=True, slots=True)
class BenchVariant:
    name: str
    implementation: str
    skip_label_logits: bool = False
    fwd_xw_bf16: bool = False
    fwd_dot_accum_bf16: bool = False
    fwd_full_h_dot: bool = False
    fwd_split_label_dot: bool = False
    fwd_split_label_bf16_mul: bool = False
    fwd_split_label_pallas: bool = False
    fwd_inline_label_scalar: bool = False
    fwd_inline_label_take: bool = False
    fwd_lse_fori_loop: bool = False
    fwd_lse_fori_v_mult: int = 4
    fwd_lse_store_path: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune block sizes for fused cross-entropy kernel.")
    parser.add_argument("--batch", type=int, default=128, help="Global batch size.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--data-shards", type=int, default=4, help="Data-parallel shards to divide batch*seq.")
    parser.add_argument("--embed", type=int, default=512, help="Hidden dimension (H).")
    parser.add_argument("--vocab", type=int, default=128256, help="Vocabulary size (V).")
    parser.add_argument(
        "--implementation",
        type=str,
        default="pallas_tpu",
        choices=("pallas_tpu", "xla", "reference"),
        help="Kernel backend implementation to benchmark.",
    )
    parser.add_argument("--input-dtype", type=str, default="bfloat16", help="Input dtype for x and w.")
    parser.add_argument("--accum-dtype", type=str, default="float32", help="Accumulation dtype for loss math.")
    parser.add_argument("--steps", type=int, default=3, help="Steady-state timing iterations.")
    parser.add_argument(
        "--b-block-sizes",
        type=str,
        default=None,
        help="Comma-separated B block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--h-block-sizes",
        type=str,
        default=None,
        help="Comma-separated H block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--v-block-sizes",
        type=str,
        default=None,
        help="Comma-separated V block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--include-infer",
        action="store_true",
        help="Also benchmark infer/default behavior (block_sizes=None) for pallas_tpu.",
    )
    parser.add_argument(
        "--skip-label-logits", action="store_true", help="Skip label-logit extraction (benchmark-only)."
    )
    parser.add_argument("--fwd-xw-bf16", action="store_true", help="Use bf16 xw_tiled in forward (benchmark-only).")
    parser.add_argument(
        "--fwd-dot-accum-bf16",
        action="store_true",
        help="Use default/bf16 dot accumulation in forward (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-full-h-dot",
        action="store_true",
        help="Use full-H dot schedule in forward (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-split-label-dot",
        action="store_true",
        help="Compute label logits outside the pallas forward kernel (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-split-label-bf16-mul",
        action="store_true",
        help="Use bf16 multiplies in split-label dot path (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-split-label-pallas",
        action="store_true",
        help="Use pallas kernel for split label-dot path (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-inline-label-scalar",
        action="store_true",
        help="Use scalar per-row accumulators for inline label logits (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-inline-label-take",
        action="store_true",
        help="Use take-based in-kernel label extraction for inline label logits (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-lse-fori-loop",
        action="store_true",
        help="Use fori-loop V sub-tiling in LSE path (benchmark-only).",
    )
    parser.add_argument(
        "--fwd-lse-fori-v-mult",
        type=int,
        default=4,
        help="Outer V block multiplier for --fwd-lse-fori-loop.",
    )
    parser.add_argument(
        "--fwd-lse-store-path",
        action="store_true",
        help="Force one-pass LSE store path with external label-dot for CE forward (benchmark-only).",
    )
    parser.add_argument("--forward-only", action="store_true", help="Benchmark only forward loss, skip gradients.")
    parser.add_argument(
        "--variant-sweep",
        action="store_true",
        help="Run pallas variant sweep.",
    )
    parser.add_argument(
        "--compare-xla",
        action="store_true",
        help="When --variant-sweep is set for pallas, also run an xla baseline.",
    )
    parser.add_argument(
        "--compare-no-label-logits",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark skip-label-logits mode.",
    )
    parser.add_argument(
        "--compare-fwd-xw-bf16",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark bf16 xw_tiled mode.",
    )
    parser.add_argument(
        "--compare-fwd-dot-accum-bf16",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark bf16 dot-accum mode.",
    )
    parser.add_argument(
        "--compare-fwd-full-h-dot",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark full-H dot schedule.",
    )
    parser.add_argument(
        "--compare-fwd-split-label-dot",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark split label-dot mode.",
    )
    parser.add_argument(
        "--compare-fwd-split-label-bf16-mul",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark split-label bf16-mul mode.",
    )
    parser.add_argument(
        "--compare-fwd-split-label-pallas",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark split-label pallas mode.",
    )
    parser.add_argument(
        "--compare-fwd-inline-label-scalar",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark inline-label scalar mode.",
    )
    parser.add_argument(
        "--compare-fwd-inline-label-take",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark inline-label take mode.",
    )
    parser.add_argument(
        "--compare-fwd-lse-fori-loop",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark split-label + lse-fori mode.",
    )
    parser.add_argument(
        "--compare-fwd-lse-store-path",
        action="store_true",
        help="When --variant-sweep is set for pallas, also benchmark one-pass LSE store path mode.",
    )
    return parser.parse_args()


def _parse_csv_ints(value: str | None) -> list[int]:
    if value is None:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _summarize_error(exc: Exception) -> tuple[str, str]:
    full = str(exc)
    first_line = full.splitlines()[0] if full else type(exc).__name__
    if len(full) <= _MAX_ERROR_CHARS:
        return first_line, full
    return first_line, f"{full[:_MAX_ERROR_CHARS]}...<truncated>"


def _build_variants(args: argparse.Namespace) -> list[BenchVariant]:
    if not args.variant_sweep:
        return [
            BenchVariant(
                name="single",
                implementation=args.implementation,
                skip_label_logits=args.skip_label_logits if args.implementation == "pallas_tpu" else False,
                fwd_xw_bf16=args.fwd_xw_bf16 if args.implementation == "pallas_tpu" else False,
                fwd_dot_accum_bf16=args.fwd_dot_accum_bf16 if args.implementation == "pallas_tpu" else False,
                fwd_full_h_dot=args.fwd_full_h_dot if args.implementation == "pallas_tpu" else False,
                fwd_split_label_dot=args.fwd_split_label_dot if args.implementation == "pallas_tpu" else False,
                fwd_split_label_bf16_mul=(
                    args.fwd_split_label_bf16_mul if args.implementation == "pallas_tpu" else False
                ),
                fwd_split_label_pallas=args.fwd_split_label_pallas if args.implementation == "pallas_tpu" else False,
                fwd_inline_label_scalar=args.fwd_inline_label_scalar if args.implementation == "pallas_tpu" else False,
                fwd_inline_label_take=args.fwd_inline_label_take if args.implementation == "pallas_tpu" else False,
                fwd_lse_fori_loop=args.fwd_lse_fori_loop if args.implementation == "pallas_tpu" else False,
                fwd_lse_fori_v_mult=args.fwd_lse_fori_v_mult if args.implementation == "pallas_tpu" else 4,
                fwd_lse_store_path=args.fwd_lse_store_path if args.implementation == "pallas_tpu" else False,
            )
        ]

    variants: list[BenchVariant] = []
    if args.implementation == "pallas_tpu":
        variants.append(BenchVariant(name="pallas_baseline", implementation="pallas_tpu"))
        if args.compare_no_label_logits:
            variants.append(
                BenchVariant(name="pallas_no_label_logits", implementation="pallas_tpu", skip_label_logits=True)
            )
        if args.compare_fwd_xw_bf16:
            variants.append(BenchVariant(name="pallas_fwd_xw_bf16", implementation="pallas_tpu", fwd_xw_bf16=True))
        if args.compare_fwd_dot_accum_bf16:
            variants.append(
                BenchVariant(name="pallas_fwd_dot_accum_bf16", implementation="pallas_tpu", fwd_dot_accum_bf16=True)
            )
        if args.compare_fwd_full_h_dot:
            variants.append(
                BenchVariant(name="pallas_fwd_full_h_dot", implementation="pallas_tpu", fwd_full_h_dot=True)
            )
        if args.compare_fwd_split_label_dot:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_split_label_dot",
                    implementation="pallas_tpu",
                    fwd_split_label_dot=True,
                )
            )
        if args.compare_fwd_split_label_bf16_mul:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_split_label_dot_bf16_mul",
                    implementation="pallas_tpu",
                    fwd_split_label_dot=True,
                    fwd_split_label_bf16_mul=True,
                )
            )
        if args.compare_fwd_split_label_pallas:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_split_label_dot_pallas",
                    implementation="pallas_tpu",
                    fwd_split_label_dot=True,
                    fwd_split_label_pallas=True,
                )
            )
        if args.compare_fwd_inline_label_scalar:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_inline_label_scalar",
                    implementation="pallas_tpu",
                    fwd_inline_label_scalar=True,
                )
            )
        if args.compare_fwd_inline_label_take:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_inline_label_take",
                    implementation="pallas_tpu",
                    fwd_inline_label_take=True,
                )
            )
        if args.compare_fwd_lse_store_path:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_lse_store_path",
                    implementation="pallas_tpu",
                    fwd_lse_store_path=True,
                )
            )
        if args.compare_no_label_logits and args.compare_fwd_xw_bf16:
            variants.append(
                BenchVariant(
                    name="pallas_no_label_logits_fwd_xw_bf16",
                    implementation="pallas_tpu",
                    skip_label_logits=True,
                    fwd_xw_bf16=True,
                )
            )
        if args.compare_no_label_logits and args.compare_fwd_dot_accum_bf16:
            variants.append(
                BenchVariant(
                    name="pallas_no_label_logits_fwd_dot_accum_bf16",
                    implementation="pallas_tpu",
                    skip_label_logits=True,
                    fwd_dot_accum_bf16=True,
                )
            )
        if args.compare_no_label_logits and args.compare_fwd_full_h_dot:
            variants.append(
                BenchVariant(
                    name="pallas_no_label_logits_fwd_full_h_dot",
                    implementation="pallas_tpu",
                    skip_label_logits=True,
                    fwd_full_h_dot=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_split_label_dot:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_split_label_dot",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_split_label_dot=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_split_label_dot and args.compare_fwd_split_label_bf16_mul:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_split_label_dot_bf16_mul",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_split_label_dot=True,
                    fwd_split_label_bf16_mul=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_split_label_pallas:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_split_label_dot_pallas",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_split_label_dot=True,
                    fwd_split_label_pallas=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_inline_label_scalar:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_inline_label_scalar",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_inline_label_scalar=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_inline_label_take:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_inline_label_take",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_inline_label_take=True,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_split_label_dot and args.compare_fwd_lse_fori_loop:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_split_label_dot_lse_fori_loop",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_split_label_dot=True,
                    fwd_lse_fori_loop=True,
                    fwd_lse_fori_v_mult=args.fwd_lse_fori_v_mult,
                )
            )
        if args.compare_fwd_full_h_dot and args.compare_fwd_lse_store_path:
            variants.append(
                BenchVariant(
                    name="pallas_fwd_full_h_dot_lse_store_path",
                    implementation="pallas_tpu",
                    fwd_full_h_dot=True,
                    fwd_lse_store_path=True,
                )
            )
    else:
        variants.append(BenchVariant(name=args.implementation, implementation=args.implementation))

    if args.compare_xla:
        variants.append(BenchVariant(name="xla_baseline", implementation="xla"))
    return variants


@contextmanager
def _variant_env(variant: BenchVariant):
    previous = {
        _SKIP_LABEL_LOGITS_ENV: os.environ.get(_SKIP_LABEL_LOGITS_ENV),
        _FWD_XW_BF16_ENV: os.environ.get(_FWD_XW_BF16_ENV),
        _FWD_DOT_ACCUM_BF16_ENV: os.environ.get(_FWD_DOT_ACCUM_BF16_ENV),
        _FWD_FULL_H_DOT_ENV: os.environ.get(_FWD_FULL_H_DOT_ENV),
        _FWD_SPLIT_LABEL_DOT_ENV: os.environ.get(_FWD_SPLIT_LABEL_DOT_ENV),
        _FWD_SPLIT_LABEL_BF16_MUL_ENV: os.environ.get(_FWD_SPLIT_LABEL_BF16_MUL_ENV),
        _FWD_SPLIT_LABEL_PALLAS_ENV: os.environ.get(_FWD_SPLIT_LABEL_PALLAS_ENV),
        _FWD_INLINE_LABEL_SCALAR_ENV: os.environ.get(_FWD_INLINE_LABEL_SCALAR_ENV),
        _FWD_INLINE_LABEL_TAKE_ENV: os.environ.get(_FWD_INLINE_LABEL_TAKE_ENV),
        _FWD_LSE_FORI_LOOP_ENV: os.environ.get(_FWD_LSE_FORI_LOOP_ENV),
        _FWD_LSE_FORI_V_MULT_ENV: os.environ.get(_FWD_LSE_FORI_V_MULT_ENV),
        _FWD_LSE_STORE_PATH_ENV: os.environ.get(_FWD_LSE_STORE_PATH_ENV),
    }
    updates = {
        _SKIP_LABEL_LOGITS_ENV: "1" if variant.skip_label_logits else None,
        _FWD_XW_BF16_ENV: "1" if variant.fwd_xw_bf16 else None,
        _FWD_DOT_ACCUM_BF16_ENV: "1" if variant.fwd_dot_accum_bf16 else None,
        _FWD_FULL_H_DOT_ENV: "1" if variant.fwd_full_h_dot else None,
        _FWD_SPLIT_LABEL_DOT_ENV: "1" if variant.fwd_split_label_dot else None,
        _FWD_SPLIT_LABEL_BF16_MUL_ENV: "1" if variant.fwd_split_label_bf16_mul else None,
        _FWD_SPLIT_LABEL_PALLAS_ENV: "1" if variant.fwd_split_label_pallas else None,
        _FWD_INLINE_LABEL_SCALAR_ENV: "1" if variant.fwd_inline_label_scalar else None,
        _FWD_INLINE_LABEL_TAKE_ENV: "1" if variant.fwd_inline_label_take else None,
        _FWD_LSE_FORI_LOOP_ENV: "1" if variant.fwd_lse_fori_loop else None,
        _FWD_LSE_FORI_V_MULT_ENV: str(variant.fwd_lse_fori_v_mult) if variant.fwd_lse_fori_loop else None,
        _FWD_LSE_STORE_PATH_ENV: "1" if variant.fwd_lse_store_path else None,
    }
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def main() -> None:
    args = _parse_args()
    if args.skip_label_logits and not args.forward_only:
        raise ValueError("--skip-label-logits is benchmark-only and requires --forward-only.")
    if args.compare_no_label_logits and not args.forward_only:
        raise ValueError("--compare-no-label-logits requires --forward-only.")
    if args.fwd_xw_bf16 and not args.forward_only:
        raise ValueError("--fwd-xw-bf16 is benchmark-only and requires --forward-only.")
    if args.compare_fwd_xw_bf16 and not args.forward_only:
        raise ValueError("--compare-fwd-xw-bf16 requires --forward-only.")
    if args.fwd_dot_accum_bf16 and not args.forward_only:
        raise ValueError("--fwd-dot-accum-bf16 is benchmark-only and requires --forward-only.")
    if args.compare_fwd_dot_accum_bf16 and not args.forward_only:
        raise ValueError("--compare-fwd-dot-accum-bf16 requires --forward-only.")
    if args.fwd_split_label_bf16_mul and not args.forward_only:
        raise ValueError("--fwd-split-label-bf16-mul is benchmark-only and requires --forward-only.")
    if args.compare_fwd_split_label_bf16_mul and not args.forward_only:
        raise ValueError("--compare-fwd-split-label-bf16-mul requires --forward-only.")
    if args.fwd_split_label_bf16_mul and not args.fwd_split_label_dot:
        raise ValueError("--fwd-split-label-bf16-mul requires --fwd-split-label-dot.")
    if args.compare_fwd_split_label_bf16_mul and not args.compare_fwd_split_label_dot:
        raise ValueError("--compare-fwd-split-label-bf16-mul requires --compare-fwd-split-label-dot.")
    if args.fwd_split_label_pallas and not args.forward_only:
        raise ValueError("--fwd-split-label-pallas is benchmark-only and requires --forward-only.")
    if args.compare_fwd_split_label_pallas and not args.forward_only:
        raise ValueError("--compare-fwd-split-label-pallas requires --forward-only.")
    if args.fwd_split_label_pallas and not args.fwd_split_label_dot:
        raise ValueError("--fwd-split-label-pallas requires --fwd-split-label-dot.")
    if args.compare_fwd_split_label_pallas and not args.compare_fwd_split_label_dot:
        raise ValueError("--compare-fwd-split-label-pallas requires --compare-fwd-split-label-dot.")
    if args.fwd_inline_label_scalar and not args.forward_only:
        raise ValueError("--fwd-inline-label-scalar is benchmark-only and requires --forward-only.")
    if args.compare_fwd_inline_label_scalar and not args.forward_only:
        raise ValueError("--compare-fwd-inline-label-scalar requires --forward-only.")
    if args.fwd_inline_label_take and not args.forward_only:
        raise ValueError("--fwd-inline-label-take is benchmark-only and requires --forward-only.")
    if args.compare_fwd_inline_label_take and not args.forward_only:
        raise ValueError("--compare-fwd-inline-label-take requires --forward-only.")
    if args.compare_fwd_lse_fori_loop and not args.compare_fwd_full_h_dot:
        raise ValueError("--compare-fwd-lse-fori-loop requires --compare-fwd-full-h-dot.")
    if args.compare_fwd_lse_fori_loop and not args.compare_fwd_split_label_dot:
        raise ValueError("--compare-fwd-lse-fori-loop requires --compare-fwd-split-label-dot.")
    if args.fwd_inline_label_scalar and args.fwd_split_label_dot:
        raise ValueError("--fwd-inline-label-scalar is incompatible with --fwd-split-label-dot.")
    if args.fwd_inline_label_take and args.fwd_split_label_dot:
        raise ValueError("--fwd-inline-label-take is incompatible with --fwd-split-label-dot.")
    if args.fwd_lse_fori_v_mult < 1:
        raise ValueError("--fwd-lse-fori-v-mult must be >= 1.")
    if args.fwd_lse_fori_loop and not args.fwd_full_h_dot:
        raise ValueError("--fwd-lse-fori-loop currently requires --fwd-full-h-dot.")
    if args.fwd_lse_fori_loop and not args.fwd_split_label_dot:
        raise ValueError("--fwd-lse-fori-loop currently requires --fwd-split-label-dot.")

    print("devices:", jax.devices())

    tokens = args.batch * args.seq_len
    if tokens % args.data_shards != 0:
        raise ValueError(f"batch*seq ({tokens}) must be divisible by data_shards ({args.data_shards}).")
    batch = tokens // args.data_shards
    embed = args.embed
    vocab = args.vocab
    print(
        "shape",
        {
            "global_batch": args.batch,
            "seq_len": args.seq_len,
            "data_shards": args.data_shards,
            "kernel_batch": batch,
            "embed": embed,
            "vocab": vocab,
        },
    )

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    input_dtype = jnp.dtype(args.input_dtype)
    accum_dtype = jnp.dtype(args.accum_dtype)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=input_dtype)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=input_dtype)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    pallas_configs: list[tuple[str, BlockSizes | None]]
    b_block_sizes = _parse_csv_ints(args.b_block_sizes)
    h_block_sizes = _parse_csv_ints(args.h_block_sizes)
    v_block_sizes = _parse_csv_ints(args.v_block_sizes)

    if b_block_sizes or h_block_sizes or v_block_sizes:
        b_values = b_block_sizes or [BlockSizes.get_default().b_block_size]
        h_values = h_block_sizes or [BlockSizes.get_default().h_block_size]
        v_values = v_block_sizes or [BlockSizes.get_default().v_block_size]
        pallas_configs = [
            (f"b{b}_h{h}_v{v}", BlockSizes(b_block_size=b, h_block_size=h, v_block_size=v))
            for b, h, v in itertools.product(b_values, h_values, v_values)
        ]
    else:
        pallas_configs = [
            ("b1024_h128_v1024", BlockSizes(b_block_size=1024, h_block_size=128, v_block_size=1024)),
            ("b1024_h256_v1024", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024)),
            ("b1024_h512_v1024", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)),
            ("b1024_h256_v2048", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048)),
            ("b1024_h512_v2048", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048)),
            ("b1024_h256_v4096", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=4096)),
            ("b1024_h512_v4096", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=4096)),
            ("b2048_h256_v2048", BlockSizes(b_block_size=2048, h_block_size=256, v_block_size=2048)),
            ("b2048_h512_v2048", BlockSizes(b_block_size=2048, h_block_size=512, v_block_size=2048)),
        ]
    if args.include_infer:
        pallas_configs.append(("infer", None))

    variants = _build_variants(args)

    def make_loss_fn(block_sizes: BlockSizes | None, implementation: str):
        def loss_fn(x_in, w_in, y_in):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                x_in,
                y_in,
                w_in,
                reduction="mean",
                logsumexp_weight=0.0,
                block_sizes=block_sizes,
                dtype=accum_dtype,
                logit_soft_cap=None,
                implementation=implementation,
            )

        return loss_fn

    for variant in variants:
        if len(variants) > 1:
            # Env vars affect compile-time kernel behavior, so force retrace.
            jax.clear_caches()

        variant_configs = pallas_configs if variant.implementation == "pallas_tpu" else [("none", None)]
        with _variant_env(variant):
            for label, cfg in variant_configs:
                print("variant", variant.name, variant.implementation, label, cfg)
                loss_fn = make_loss_fn(cfg, implementation=variant.implementation)
                if args.forward_only:
                    loss_jit = jax.jit(loss_fn)
                else:
                    loss_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
                result: dict[str, str | float | int] = {
                    "variant": variant.name,
                    "label": label,
                    "implementation": variant.implementation,
                    "skip_label_logits": int(variant.skip_label_logits),
                    "fwd_xw_bf16": int(variant.fwd_xw_bf16),
                    "fwd_dot_accum_bf16": int(variant.fwd_dot_accum_bf16),
                    "fwd_full_h_dot": int(variant.fwd_full_h_dot),
                    "fwd_split_label_dot": int(variant.fwd_split_label_dot),
                    "fwd_split_label_bf16_mul": int(variant.fwd_split_label_bf16_mul),
                    "fwd_split_label_pallas": int(variant.fwd_split_label_pallas),
                    "fwd_inline_label_scalar": int(variant.fwd_inline_label_scalar),
                    "fwd_inline_label_take": int(variant.fwd_inline_label_take),
                    "fwd_lse_fori_loop": int(variant.fwd_lse_fori_loop),
                    "fwd_lse_fori_v_mult": int(variant.fwd_lse_fori_v_mult),
                    "fwd_lse_store_path": int(variant.fwd_lse_store_path),
                    "status": "failed",
                }
                try:
                    start = time.perf_counter()
                    if args.forward_only:
                        loss = loss_jit(x_raw, w_raw, y_raw)
                        jax.block_until_ready(loss)
                    else:
                        loss, out = loss_jit(x_raw, w_raw, y_raw)
                        jax.block_until_ready(out)
                    compile_time = time.perf_counter() - start

                    steps = args.steps
                    start = time.perf_counter()
                    for _ in range(steps):
                        if args.forward_only:
                            out = loss_jit(x_raw, w_raw, y_raw)
                            jax.block_until_ready(out)
                        else:
                            out = loss_jit(x_raw, w_raw, y_raw)
                            jax.block_until_ready(out)
                    steady_time = (time.perf_counter() - start) / steps

                    print("loss", float(loss))
                    print("compile_time_s", compile_time)
                    print("steady_time_s", steady_time)
                    print("tokens_per_s", tokens / steady_time)
                    result.update(
                        {
                            "status": "ok",
                            "loss": float(loss),
                            "compile_time_s": compile_time,
                            "steady_time_s": steady_time,
                            "tokens_per_s": tokens / steady_time,
                        }
                    )
                except Exception as exc:
                    error_summary, error_text = _summarize_error(exc)
                    print("failed", type(exc).__name__, error_summary)
                    result.update(
                        {
                            "error_type": type(exc).__name__,
                            "error": error_text,
                        }
                    )
                print("result_json", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
