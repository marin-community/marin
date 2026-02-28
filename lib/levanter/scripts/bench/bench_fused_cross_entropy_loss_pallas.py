# Copyright 2026 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)

_V5P_TFLOPS_BF16_PER_CHIP = 459e12
_V5P_HBM_BW_BYTES_PER_S_PER_CHIP = 2.765e12
_V4_TFLOPS_BF16_PER_CHIP = 275e12
_V4_HBM_BW_BYTES_PER_S_PER_CHIP = 1.2e12

_DISABLE_MEGACORE_ENV = "LEVANTER_PALLAS_TPU_DISABLE_MEGACORE"
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


@dataclass(frozen=True, slots=True)
class BenchVariant:
    name: str
    implementation: str
    disable_megacore: bool = False
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


def _estimate_v5p_roofline(
    *,
    batch: int,
    embed: int,
    vocab: int,
    dtype: jnp.dtype,
    num_devices: int,
) -> dict[str, float] | None:
    device_kind = jax.devices()[0].device_kind.lower() if jax.devices() else ""
    if not device_kind:
        return None
    if "v5p" not in device_kind and "v5" not in device_kind:
        return None

    # v5p JAX devices are chips (megacore), so use num_devices directly.
    chips = max(1, num_devices)
    peak_tflops = _V5P_TFLOPS_BF16_PER_CHIP * chips

    flop_count = 2.0 * batch * embed * vocab
    compute_time_s = flop_count / peak_tflops

    bytes_per_elem = jnp.dtype(dtype).itemsize
    memory_bytes = (batch * embed + embed * vocab + batch) * bytes_per_elem
    peak_bw = _V5P_HBM_BW_BYTES_PER_S_PER_CHIP * chips
    memory_time_s = memory_bytes / peak_bw

    return {
        "device_kind": device_kind,
        "chips": float(chips),
        "flops": float(flop_count),
        "compute_time_s": float(compute_time_s),
        "memory_time_s": float(memory_time_s),
        "tokens_per_s": float(batch / max(compute_time_s, memory_time_s)),
    }


def _estimate_v4_roofline(
    *,
    batch: int,
    embed: int,
    vocab: int,
    dtype: jnp.dtype,
    num_devices: int,
) -> dict[str, float] | None:
    device_kind = jax.devices()[0].device_kind.lower() if jax.devices() else ""
    if not device_kind:
        return None
    if "v4" not in device_kind:
        return None

    chips = max(1, num_devices)
    peak_tflops = _V4_TFLOPS_BF16_PER_CHIP * chips

    flop_count = 2.0 * batch * embed * vocab
    compute_time_s = flop_count / peak_tflops

    bytes_per_elem = jnp.dtype(dtype).itemsize
    memory_bytes = (batch * embed + embed * vocab + batch) * bytes_per_elem
    peak_bw = _V4_HBM_BW_BYTES_PER_S_PER_CHIP * chips
    memory_time_s = memory_bytes / peak_bw

    return {
        "device_kind": device_kind,
        "chips": float(chips),
        "flops": float(flop_count),
        "compute_time_s": float(compute_time_s),
        "memory_time_s": float(memory_time_s),
        "tokens_per_s": float(batch / max(compute_time_s, memory_time_s)),
    }


def _with_mesh(mesh):
    set_mesh = getattr(jax, "set_mesh", None)
    if set_mesh is None:
        return mesh
    set_mesh(mesh)
    return None


def _clear_mesh(mesh):
    del mesh


@contextmanager
def _variant_env(variant: BenchVariant):
    previous = {
        _DISABLE_MEGACORE_ENV: os.environ.get(_DISABLE_MEGACORE_ENV),
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
        _DISABLE_MEGACORE_ENV: "1" if variant.disable_megacore else None,
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


def _time_jitted(fn, *args, steps: int, warmup: int) -> tuple[float, float, jax.Array]:
    start = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        jax.block_until_ready(out)
    steady_time = (time.perf_counter() - start) / steps
    return compile_time, steady_time, out


def _build_variants(args: argparse.Namespace) -> list[BenchVariant]:
    if args.variant_sweep:
        variants: list[BenchVariant] = []
        if args.implementation == "pallas_tpu":
            variants.extend(
                [
                    BenchVariant(name="pallas_baseline", implementation="pallas_tpu"),
                    BenchVariant(name="pallas_no_megacore", implementation="pallas_tpu", disable_megacore=True),
                ]
            )
            if args.compare_no_label_logits:
                variants.append(
                    BenchVariant(name="pallas_no_label_logits", implementation="pallas_tpu", skip_label_logits=True)
                )
            if args.compare_fwd_xw_bf16:
                variants.append(BenchVariant(name="pallas_fwd_xw_bf16", implementation="pallas_tpu", fwd_xw_bf16=True))
            if args.compare_fwd_dot_accum_bf16:
                variants.append(
                    BenchVariant(
                        name="pallas_fwd_dot_accum_bf16", implementation="pallas_tpu", fwd_dot_accum_bf16=True
                    )
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
            if (
                args.compare_fwd_full_h_dot
                and args.compare_fwd_split_label_dot
                and args.compare_fwd_split_label_bf16_mul
            ):
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

    return [
        BenchVariant(
            name="single",
            implementation=args.implementation,
            disable_megacore=args.disable_megacore if args.implementation == "pallas_tpu" else False,
            skip_label_logits=args.skip_label_logits if args.implementation == "pallas_tpu" else False,
            fwd_xw_bf16=args.fwd_xw_bf16 if args.implementation == "pallas_tpu" else False,
            fwd_dot_accum_bf16=args.fwd_dot_accum_bf16 if args.implementation == "pallas_tpu" else False,
            fwd_full_h_dot=args.fwd_full_h_dot if args.implementation == "pallas_tpu" else False,
            fwd_split_label_dot=args.fwd_split_label_dot if args.implementation == "pallas_tpu" else False,
            fwd_split_label_bf16_mul=args.fwd_split_label_bf16_mul if args.implementation == "pallas_tpu" else False,
            fwd_split_label_pallas=args.fwd_split_label_pallas if args.implementation == "pallas_tpu" else False,
            fwd_inline_label_scalar=args.fwd_inline_label_scalar if args.implementation == "pallas_tpu" else False,
            fwd_inline_label_take=args.fwd_inline_label_take if args.implementation == "pallas_tpu" else False,
            fwd_lse_fori_loop=args.fwd_lse_fori_loop if args.implementation == "pallas_tpu" else False,
            fwd_lse_fori_v_mult=args.fwd_lse_fori_v_mult if args.implementation == "pallas_tpu" else 4,
            fwd_lse_store_path=args.fwd_lse_store_path if args.implementation == "pallas_tpu" else False,
        )
    ]


def _run_variant(
    *,
    x_raw: jax.Array,
    w_raw: jax.Array,
    y_raw: jax.Array,
    accum_dtype: jnp.dtype,
    block_sizes: BlockSizes | None,
    variant: BenchVariant,
    use_shard_map: bool,
    data_shards: int,
    steps: int,
    warmup: int,
    forward_only: bool,
) -> dict[str, float]:
    implementation = variant.implementation

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

    if use_shard_map:
        if data_shards <= 0:
            raise ValueError("data_shards must be positive when using --shard-map.")
        devices = jax.devices()[:data_shards]
        mesh = Mesh(np.array(devices), ("data",))

        def shard_loss(x_in, w_in, y_in):
            loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                x_in,
                y_in,
                w_in,
                reduction=None,
                logsumexp_weight=0.0,
                block_sizes=block_sizes,
                dtype=accum_dtype,
                logit_soft_cap=None,
                implementation=implementation,
            )
            local_sum = jnp.sum(loss)
            total_sum = jax.lax.psum(local_sum, "data")
            total_denom = jax.lax.psum(loss.shape[0], "data")
            return total_sum / total_denom

        loss_jit = jax.jit(
            jax.shard_map(
                shard_loss,
                in_specs=(P("data", None), P(None, None), P("data")),
                out_specs=P(),
                check_vma=False,
            )
        )
        if not forward_only:

            def shard_grad(x_in, w_in, y_in):
                def loss_inner(x_inner, w_inner, y_inner):
                    loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                        x_inner,
                        y_inner,
                        w_inner,
                        reduction=None,
                        logsumexp_weight=0.0,
                        block_sizes=block_sizes,
                        dtype=accum_dtype,
                        logit_soft_cap=None,
                        implementation=implementation,
                    )
                    local_sum = jnp.sum(loss)
                    total_sum = jax.lax.psum(local_sum, "data")
                    total_denom = jax.lax.psum(loss.shape[0], "data")
                    return total_sum / total_denom

                return jax.grad(loss_inner, argnums=(0, 1))(x_in, w_in, y_in)

            grad_jit = jax.jit(
                jax.shard_map(
                    shard_grad,
                    in_specs=(P("data", None), P(None, None), P("data")),
                    out_specs=(P("data", None), P(None, None)),
                    check_vma=False,
                )
            )

        token = _with_mesh(mesh)
        try:
            if token is None:
                compile_time, steady_time, loss_out = _time_jitted(
                    loss_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup
                )
                if not forward_only:
                    bwd_compile_time, bwd_steady_time, _ = _time_jitted(
                        grad_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup
                    )
            else:
                with token:
                    compile_time, steady_time, loss_out = _time_jitted(
                        loss_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup
                    )
                    if not forward_only:
                        bwd_compile_time, bwd_steady_time, _ = _time_jitted(
                            grad_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup
                        )
        finally:
            _clear_mesh(mesh)
    else:
        loss_jit = jax.jit(loss_fn)
        compile_time, steady_time, loss_out = _time_jitted(loss_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup)
        if not forward_only:
            grad_jit = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
            bwd_compile_time, bwd_steady_time, _ = _time_jitted(
                grad_jit, x_raw, w_raw, y_raw, steps=steps, warmup=warmup
            )

    result = {
        "loss": float(loss_out),
        "compile_time_s": compile_time,
        "steady_time_s": steady_time,
    }
    if not forward_only:
        result.update(
            {
                "bwd_compile_time_s": bwd_compile_time,
                "bwd_steady_time_s": bwd_steady_time,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pos", type=int, default=2048)
    parser.add_argument("--embed", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--input-dtype", type=str, default="bfloat16")
    parser.add_argument("--accum-dtype", type=str, default="float32")
    parser.add_argument("--implementation", type=str, default="pallas_tpu")
    parser.add_argument("--block-sizes", type=str, choices=("default", "infer"), default="default")
    parser.add_argument("--b-block-size", type=int, default=None)
    parser.add_argument("--h-block-size", type=int, default=None)
    parser.add_argument("--v-block-size", type=int, default=None)
    parser.add_argument("--shard-map", action="store_true")
    parser.add_argument("--data-shards", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--disable-megacore", action="store_true")
    parser.add_argument("--skip-label-logits", action="store_true")
    parser.add_argument("--fwd-xw-bf16", action="store_true")
    parser.add_argument("--fwd-dot-accum-bf16", action="store_true")
    parser.add_argument("--fwd-full-h-dot", action="store_true")
    parser.add_argument("--fwd-split-label-dot", action="store_true")
    parser.add_argument("--fwd-split-label-bf16-mul", action="store_true")
    parser.add_argument("--fwd-split-label-pallas", action="store_true")
    parser.add_argument("--fwd-inline-label-scalar", action="store_true")
    parser.add_argument("--fwd-inline-label-take", action="store_true")
    parser.add_argument("--fwd-lse-fori-loop", action="store_true")
    parser.add_argument("--fwd-lse-fori-v-mult", type=int, default=4)
    parser.add_argument("--fwd-lse-store-path", action="store_true")
    parser.add_argument("--variant-sweep", action="store_true")
    parser.add_argument("--compare-xla", action="store_true")
    parser.add_argument("--compare-no-label-logits", action="store_true")
    parser.add_argument("--compare-fwd-xw-bf16", action="store_true")
    parser.add_argument("--compare-fwd-dot-accum-bf16", action="store_true")
    parser.add_argument("--compare-fwd-full-h-dot", action="store_true")
    parser.add_argument("--compare-fwd-split-label-dot", action="store_true")
    parser.add_argument("--compare-fwd-split-label-bf16-mul", action="store_true")
    parser.add_argument("--compare-fwd-split-label-pallas", action="store_true")
    parser.add_argument("--compare-fwd-inline-label-scalar", action="store_true")
    parser.add_argument("--compare-fwd-inline-label-take", action="store_true")
    parser.add_argument("--compare-fwd-lse-fori-loop", action="store_true")
    parser.add_argument("--compare-fwd-lse-store-path", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

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
    if args.fwd_full_h_dot and not args.forward_only:
        raise ValueError("--fwd-full-h-dot is benchmark-only and requires --forward-only.")
    if args.compare_fwd_full_h_dot and not args.forward_only:
        raise ValueError("--compare-fwd-full-h-dot requires --forward-only.")
    if args.fwd_split_label_dot and not args.forward_only:
        raise ValueError("--fwd-split-label-dot is benchmark-only and requires --forward-only.")
    if args.compare_fwd_split_label_dot and not args.forward_only:
        raise ValueError("--compare-fwd-split-label-dot requires --forward-only.")
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
    if args.fwd_inline_label_scalar and args.fwd_split_label_dot:
        raise ValueError("--fwd-inline-label-scalar is incompatible with --fwd-split-label-dot.")
    if args.fwd_inline_label_take and not args.forward_only:
        raise ValueError("--fwd-inline-label-take is benchmark-only and requires --forward-only.")
    if args.compare_fwd_inline_label_take and not args.forward_only:
        raise ValueError("--compare-fwd-inline-label-take requires --forward-only.")
    if args.fwd_inline_label_take and args.fwd_split_label_dot:
        raise ValueError("--fwd-inline-label-take is incompatible with --fwd-split-label-dot.")
    if args.fwd_lse_fori_loop and not args.forward_only:
        raise ValueError("--fwd-lse-fori-loop is benchmark-only and requires --forward-only.")
    if args.compare_fwd_lse_fori_loop and not args.forward_only:
        raise ValueError("--compare-fwd-lse-fori-loop requires --forward-only.")
    if args.compare_fwd_lse_fori_loop and not args.compare_fwd_full_h_dot:
        raise ValueError("--compare-fwd-lse-fori-loop requires --compare-fwd-full-h-dot.")
    if args.compare_fwd_lse_fori_loop and not args.compare_fwd_split_label_dot:
        raise ValueError("--compare-fwd-lse-fori-loop requires --compare-fwd-split-label-dot.")
    if args.fwd_lse_fori_v_mult < 1:
        raise ValueError("--fwd-lse-fori-v-mult must be >= 1.")
    if args.fwd_lse_fori_loop and not args.fwd_full_h_dot:
        raise ValueError("--fwd-lse-fori-loop currently requires --fwd-full-h-dot.")
    if args.fwd_lse_fori_loop and not args.fwd_split_label_dot:
        raise ValueError("--fwd-lse-fori-loop currently requires --fwd-split-label-dot.")
    if args.fwd_lse_store_path and not args.forward_only:
        raise ValueError("--fwd-lse-store-path is benchmark-only and requires --forward-only.")
    if args.compare_fwd_lse_store_path and not args.forward_only:
        raise ValueError("--compare-fwd-lse-store-path requires --forward-only.")

    print("devices:", jax.devices())

    batch = args.batch
    pos = args.pos
    embed = args.embed
    vocab = args.vocab
    tokens = batch * pos
    input_dtype = jnp.dtype(args.input_dtype)
    accum_dtype = jnp.dtype(args.accum_dtype)
    use_shard_map = args.shard_map
    data_shards = args.data_shards or len(jax.devices())
    block_sizes = BlockSizes.get_default() if args.block_sizes == "default" else None
    explicit_block_sizes = any(
        value is not None for value in (args.b_block_size, args.h_block_size, args.v_block_size)
    )
    if explicit_block_sizes:
        if args.block_sizes != "default":
            raise ValueError("--b/h/v-block-size overrides require --block-sizes default.")
        defaults = BlockSizes.get_default()
        block_sizes = BlockSizes(
            b_block_size=args.b_block_size if args.b_block_size is not None else defaults.b_block_size,
            h_block_size=args.h_block_size if args.h_block_size is not None else defaults.h_block_size,
            v_block_size=args.v_block_size if args.v_block_size is not None else defaults.v_block_size,
        )

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x_raw = jax.random.normal(key_x, (tokens, embed), dtype=input_dtype)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=input_dtype)
    y_raw = jax.random.randint(key_y, (tokens,), 0, vocab, dtype=jnp.int32)

    roofline = _estimate_v5p_roofline(
        batch=tokens,
        embed=embed,
        vocab=vocab,
        dtype=accum_dtype,
        num_devices=len(jax.devices()),
    )
    if roofline is None:
        roofline = _estimate_v4_roofline(
            batch=tokens,
            embed=embed,
            vocab=vocab,
            dtype=accum_dtype,
            num_devices=len(jax.devices()),
        )
    if roofline is not None:
        print("roofline.device_kind", roofline["device_kind"])
        print("roofline.chips", roofline["chips"])
        print("roofline.flops", roofline["flops"])
        print("roofline.compute_time_s", roofline["compute_time_s"])
        print("roofline.memory_time_s", roofline["memory_time_s"])
        print("roofline.tokens_per_s", roofline["tokens_per_s"])

    variants = _build_variants(args)
    run_many = len(variants) > 1
    for variant in variants:
        if run_many:
            # Env vars are compile-time signals for the pallas kernel. Clear JAX
            # caches between variants so each one retraces under its own env.
            jax.clear_caches()

        result: dict[str, str | int | float] = {
            "variant": variant.name,
            "implementation": variant.implementation,
            "disable_megacore": int(variant.disable_megacore),
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
            "b_block_size": int(block_sizes.b_block_size) if block_sizes is not None else -1,
            "h_block_size": int(block_sizes.h_block_size) if block_sizes is not None else -1,
            "v_block_size": int(block_sizes.v_block_size) if block_sizes is not None else -1,
            "batch": batch,
            "pos": pos,
            "embed": embed,
            "vocab": vocab,
            "input_dtype": str(input_dtype),
            "accum_dtype": str(accum_dtype),
            "status": "failed",
        }
        try:
            with _variant_env(variant):
                metrics = _run_variant(
                    x_raw=x_raw,
                    w_raw=w_raw,
                    y_raw=y_raw,
                    accum_dtype=accum_dtype,
                    block_sizes=block_sizes,
                    variant=variant,
                    use_shard_map=use_shard_map,
                    data_shards=data_shards,
                    steps=args.steps,
                    warmup=args.warmup,
                    forward_only=args.forward_only,
                )

            fwd_tps = tokens / metrics["steady_time_s"]
            result.update(
                {
                    "status": "ok",
                    "loss": metrics["loss"],
                    "compile_time_s": metrics["compile_time_s"],
                    "steady_time_s": metrics["steady_time_s"],
                    "tokens_per_s": fwd_tps,
                }
            )
            if not args.forward_only:
                bwd_tps = tokens / metrics["bwd_steady_time_s"]
                result.update(
                    {
                        "bwd_compile_time_s": metrics["bwd_compile_time_s"],
                        "bwd_steady_time_s": metrics["bwd_steady_time_s"],
                        "bwd_tokens_per_s": bwd_tps,
                        "combined_tokens_per_s": tokens / (metrics["steady_time_s"] + metrics["bwd_steady_time_s"]),
                    }
                )
        except Exception as exc:  # pragma: no cover - backend/runtime dependent
            result.update({"error_type": type(exc).__name__, "error": str(exc)})
            print("variant_failed", variant.name, type(exc).__name__, exc)

        result = {key: value for key, value in result.items() if key != "error" or len(str(value)) < 800}
        print("variant", variant.name)
        print("implementation", variant.implementation)
        print("disable_megacore", int(variant.disable_megacore))
        print("skip_label_logits", int(variant.skip_label_logits))
        print("fwd_xw_bf16", int(variant.fwd_xw_bf16))
        print("fwd_dot_accum_bf16", int(variant.fwd_dot_accum_bf16))
        print("fwd_full_h_dot", int(variant.fwd_full_h_dot))
        print("fwd_split_label_dot", int(variant.fwd_split_label_dot))
        print("fwd_split_label_bf16_mul", int(variant.fwd_split_label_bf16_mul))
        print("fwd_split_label_pallas", int(variant.fwd_split_label_pallas))
        print("fwd_inline_label_scalar", int(variant.fwd_inline_label_scalar))
        print("fwd_inline_label_take", int(variant.fwd_inline_label_take))
        print("fwd_lse_fori_loop", int(variant.fwd_lse_fori_loop))
        print("fwd_lse_fori_v_mult", int(variant.fwd_lse_fori_v_mult))
        print("fwd_lse_store_path", int(variant.fwd_lse_store_path))
        print("b_block_size", result["b_block_size"])
        print("h_block_size", result["h_block_size"])
        print("v_block_size", result["v_block_size"])
        print("status", result["status"])
        print("batch", batch)
        print("pos", pos)
        print("embed", embed)
        print("vocab", vocab)
        print("input_dtype", input_dtype)
        print("accum_dtype", accum_dtype)
        if result["status"] == "ok":
            print("loss", result["loss"])
            print("compile_time_s", result["compile_time_s"])
            print("steady_time_s", result["steady_time_s"])
            print("tokens_per_s", result["tokens_per_s"])
            if not args.forward_only:
                print("bwd_compile_time_s", result["bwd_compile_time_s"])
                print("bwd_steady_time_s", result["bwd_steady_time_s"])
                print("bwd_tokens_per_s", result["bwd_tokens_per_s"])
                print("combined_tokens_per_s", result["combined_tokens_per_s"])
        else:
            print("error_type", result["error_type"])
            print("error", result.get("error", "<omitted>"))
        print("result_json", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
