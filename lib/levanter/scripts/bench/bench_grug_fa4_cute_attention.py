# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Grug FA4/CuTe attention on dense packed-segment shapes."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp

from levanter.grug.attention._fa4_cute import (
    _packed_segment_backward_block_sparse_indices_with_full,
    _packed_segment_causal_lower_bounds,
)
from levanter.grug.attention._fa4_cute_backend import (
    fa4_cute_attention_forward,
    segmented_flash_attention_forward,
    segmented_flash_attention_backward_sm90_native,
)
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config
from levanter.grug.attention._fa4_thd import (
    _jax_fa4_thd_attention,
    _cutlass_dtype,
    _import_upstream_fa4_cute,
    _thd_kernel_config,
    fa4_thd_attention_backward,
    fa4_thd_attention_forward,
)


@dataclass(frozen=True)
class BenchResult:
    label: str
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    sliding_window: int
    dtype: str
    backend: str
    config: dict[str, object]
    warmup: int
    iterations: int
    forward_ms: float
    backward_ms: float
    backward_with_forward_ms: float
    flops_forward: float
    flops_backward: float
    forward_tflops: float
    backward_tflops: float
    forward_sol: float
    backward_sol: float
    pass_mode: str
    native_mask_mode: str


def _dtype(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    raise ValueError(f"Unsupported dtype {name}.")


def _config(args: argparse.Namespace) -> Flash4CuteKernelConfig:
    config = flash4_cute_kernel_config(args.head_dim, arch=args.arch)
    forward_tile = config.forward_tile
    backward_tile = config.backward_tile
    num_threads = config.num_threads
    backward_arch = config.backward_arch

    if args.forward_tile is not None:
        forward_tile = _parse_tile(args.forward_tile)
    if args.backward_tile is not None:
        backward_tile = _parse_tile(args.backward_tile)
    if args.num_threads is not None:
        num_threads = args.num_threads
    if args.backward_arch is not None:
        backward_arch = None if args.backward_arch == "infer" else int(args.backward_arch)

    config = replace(
        config,
        forward_tile=forward_tile,
        backward_tile=backward_tile,
        num_threads=num_threads,
        backward_arch=backward_arch,
    )
    if config.sm90_backward is not None:
        sm90_backward = config.sm90_backward
        if args.sm90_tile is not None:
            sm90_backward = replace(sm90_backward, tile=_parse_tile(args.sm90_tile))
        if args.sm90_num_threads is not None:
            sm90_backward = replace(sm90_backward, num_threads=args.sm90_num_threads)
        if args.sm90_stages_q is not None:
            sm90_backward = replace(sm90_backward, num_stages_q=args.sm90_stages_q)
        if args.sm90_stages_do is not None:
            sm90_backward = replace(sm90_backward, num_stages_do=args.sm90_stages_do)
        if args.sm90_stages_pds is not None:
            sm90_backward = replace(sm90_backward, num_stages_pds=args.sm90_stages_pds)
        if args.sm90_sdp_swap_ab is not None:
            sm90_backward = replace(sm90_backward, sdp_swap_ab=_parse_bool(args.sm90_sdp_swap_ab))
        if args.sm90_dkv_swap_ab is not None:
            sm90_backward = replace(sm90_backward, dkv_swap_ab=_parse_bool(args.sm90_dkv_swap_ab))
        if args.sm90_dq_swap_ab is not None:
            sm90_backward = replace(sm90_backward, dq_swap_ab=_parse_bool(args.sm90_dq_swap_ab))
        if args.sm90_atom_layout_n_dkv is not None:
            sm90_backward = replace(sm90_backward, atom_layout_n_dkv=args.sm90_atom_layout_n_dkv)
        if args.sm90_atom_layout_m_dq is not None:
            sm90_backward = replace(sm90_backward, atom_layout_m_dq=args.sm90_atom_layout_m_dq)
        if args.sm90_dq_single_wg is not None:
            sm90_backward = replace(sm90_backward, dq_single_wg=_parse_bool(args.sm90_dq_single_wg))
        config = replace(config, sm90_backward=sm90_backward)
    return config


def _parse_tile(value: str) -> tuple[int, int]:
    left, sep, right = value.lower().partition("x")
    if not sep:
        raise ValueError(f"Tile must be MxN, got {value}.")
    return int(left), int(right)


def _parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value}.")


def _upstream_sm90_config() -> Flash4CuteKernelConfig:
    return Flash4CuteKernelConfig(
        forward_tile=(128, 128),
        backward_tile=(64, 128),
        num_threads=384,
        backward_arch=90,
    )


def _causal_local_options(args: argparse.Namespace) -> tuple[bool, bool, int | None, int | None]:
    if args.sliding_window >= args.seq_len:
        return True, False, None, None
    return False, True, args.sliding_window - 1, 0


def _upstream_dense_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    is_causal: bool,
    is_local: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[jax.Array, jax.Array]:
    modules = _import_upstream_fa4_cute()
    if modules.arch // 10 != 9:
        raise NotImplementedError(f"--backend upstream is currently an SM90 probe, got SM{modules.arch}.")

    cutlass = modules.cutlass
    cute = modules.cute
    cuda = modules.cuda
    cute_dtype = _cutlass_dtype(cutlass, q.dtype)
    qhead_per_kvhead = q.shape[2] // k.shape[2]
    flash_fwd = modules.FlashAttentionForward(
        cute_dtype,
        q.shape[-1],
        v.shape[-1],
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_local=is_local,
        pack_gqa=qhead_per_kvhead > 1,
        tile_m=kernel_config.forward_tile[0],
        tile_n=kernel_config.forward_tile[1],
        num_stages=2,
        num_threads=kernel_config.num_threads,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        q_subtile_factor=None,
        paged_kv_non_tma=False,
    )

    @cute.jit
    def launch(
        stream: cuda.CUstream,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        out_tensor: cute.Tensor,
        lse_tensor: cute.Tensor,
        *,
        scale: cutlass.Float32,
    ):
        flash_fwd(
            q_tensor,
            k_tensor,
            v_tensor,
            out_tensor,
            lse_tensor,
            scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            stream=stream,
        )

    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, 8), static=True)
    lse_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    out_shape_dtype = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_shape_dtype = jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32)
    call = modules.cjax.cutlass_call(
        launch,
        output_shape_dtype=(out_shape_dtype, lse_shape_dtype),
        input_spec=(qkv_spec, qkv_spec, qkv_spec),
        output_spec=(qkv_spec, lse_spec),
        use_static_tensors=True,
        compile_options="--enable-tvm-ffi",
        scale=softmax_scale,
    )
    return call(q, k, v)


def _time_call(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) * 1000.0 / iterations


def _attention_flops(batch: int, seq_len: int, q_heads: int, head_dim: int, sliding_window: int) -> float:
    dense_prefix = min(seq_len, sliding_window)
    per_head_pairs = dense_prefix * (dense_prefix + 1) / 2
    per_head_pairs += max(seq_len - sliding_window, 0) * sliding_window
    score_pairs = per_head_pairs * batch * q_heads
    qk_flops = 2.0 * score_pairs * head_dim
    pv_flops = 2.0 * score_pairs * head_dim
    return qk_flops + pv_flops


def _make_inputs(args: argparse.Namespace):
    dtype = _dtype(args.dtype)
    q_shape = (args.batch, args.seq_len, args.q_heads, args.head_dim)
    kv_shape = (args.batch, args.seq_len, args.kv_heads, args.head_dim)
    q_key, k_key, v_key, dout_key = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    q = jax.random.normal(q_key, q_shape, dtype=dtype)
    k = jax.random.normal(k_key, kv_shape, dtype=dtype)
    v = jax.random.normal(v_key, kv_shape, dtype=dtype)
    dout = jax.random.normal(dout_key, q_shape, dtype=dtype)
    segment_ids = jnp.broadcast_to(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], (args.batch, args.seq_len))
    segment_ids = jnp.zeros_like(segment_ids)
    lower_bounds, valid = _packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=args.batch,
        seq_len=args.seq_len,
        sliding_window=args.sliding_window,
    )
    return q, k, v, dout, lower_bounds, valid


def run(args: argparse.Namespace) -> BenchResult:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"bench_grug_fa4_cute_attention requires GPU backend, got {jax.default_backend()}.")

    q, k, v, dout, lower_bounds, valid = _make_inputs(args)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    if args.backend == "thd":
        config = _thd_kernel_config(args.head_dim)
        cu_seqlens = jnp.arange(args.batch + 1, dtype=jnp.int32) * args.seq_len
        q = q.reshape(args.batch * args.seq_len, args.q_heads, args.head_dim)
        k = k.reshape(args.batch * args.seq_len, args.kv_heads, args.head_dim)
        v = v.reshape(args.batch * args.seq_len, args.kv_heads, args.head_dim)
        dout = dout.reshape(args.batch * args.seq_len, args.q_heads, args.head_dim)

        def forward(q_arg, k_arg, v_arg):
            return _jax_fa4_thd_attention(
                q_arg,
                k_arg,
                v_arg,
                cu_seqlens,
                softmax_scale,
                config,
            )

        def direct_forward(q_arg, k_arg, v_arg):
            return fa4_thd_attention_forward(
                q_arg,
                k_arg,
                v_arg,
                cu_seqlens,
                softmax_scale=softmax_scale,
                kernel_config=config,
            )[0]

        def direct_backward(q_arg, k_arg, v_arg, out_arg, d_arg, lse_arg):
            return fa4_thd_attention_backward(
                q_arg,
                k_arg,
                v_arg,
                out_arg,
                d_arg,
                lse_arg,
                cu_seqlens,
                softmax_scale=softmax_scale,
                kernel_config=config,
            )

    elif args.backend == "upstream":
        config = _upstream_sm90_config()
        is_causal, is_local, window_size_left, window_size_right = _causal_local_options(args)

        def forward(q_arg, k_arg, v_arg):
            return _upstream_dense_forward(
                q_arg,
                k_arg,
                v_arg,
                softmax_scale=softmax_scale,
                kernel_config=config,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )[0]

        def direct_forward(q_arg, k_arg, v_arg):
            return forward(q_arg, k_arg, v_arg)

        def direct_backward(q_arg, k_arg, v_arg, out_arg, d_arg, lse_arg):
            del q_arg, k_arg, v_arg, out_arg, d_arg, lse_arg
            raise NotImplementedError("--pass direct-backward is not implemented for --backend upstream yet.")

    elif args.backend in ("cute", "cute-native"):
        config = _config(args)

        def forward(q_arg, k_arg, v_arg):
            return fa4_cute_attention_forward(
                q_arg,
                k_arg,
                v_arg,
                lower_bounds,
                valid,
                sm_scale=softmax_scale,
                kernel_config=config,
            )

        def direct_forward(q_arg, k_arg, v_arg):
            return forward(q_arg, k_arg, v_arg)

        def direct_backward(q_arg, k_arg, v_arg, out_arg, d_arg, lse_arg):
            if args.backend != "cute-native":
                raise NotImplementedError("--pass direct-backward is implemented for --backend thd or cute-native.")
            sm90_config = config.sm90_backward
            if sm90_config is None:
                raise NotImplementedError("--backend cute-native direct backward requires an SM90 native config.")
            mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = (
                _packed_segment_backward_block_sparse_indices_with_full(
                    lower_bounds,
                    valid,
                    tile_m=sm90_config.tile[0],
                    tile_n=sm90_config.tile[1],
                )
            )
            return segmented_flash_attention_backward_sm90_native(
                q_arg,
                k_arg,
                v_arg,
                out_arg,
                d_arg,
                lse_arg,
                lower_bounds,
                valid,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
                softmax_scale=softmax_scale,
                kernel_config=config,
                window_size_left=args.sliding_window - 1 if args.native_mask_mode == "builtin-local" else None,
            )

    else:
        raise ValueError(f"Unsupported backend {args.backend}.")

    forward_jit = jax.jit(forward)
    backward_with_forward_jit = jax.jit(
        lambda q_arg, k_arg, v_arg, d_arg: jax.vjp(forward, q_arg, k_arg, v_arg)[1](d_arg)
    )

    if args.pass_mode == "forward":
        forward_jit = jax.jit(direct_forward)
        jax.block_until_ready(forward_jit(q, k, v))
        for _ in range(args.warmup):
            jax.block_until_ready(forward_jit(q, k, v))
        forward_ms = _time_call(lambda: forward_jit(q, k, v), args.iterations)
        return _result(
            args,
            config,
            forward_ms=forward_ms,
            backward_ms=float("nan"),
            backward_with_forward_ms=float("nan"),
        )

    if args.pass_mode == "direct-backward":
        if args.backend == "thd":
            out, lse = jax.block_until_ready(
                fa4_thd_attention_forward(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    softmax_scale=softmax_scale,
                    kernel_config=config,
                )
            )
        elif args.backend == "cute-native":
            out, lse = jax.block_until_ready(
                segmented_flash_attention_forward(
                    q,
                    k,
                    v,
                    lower_bounds,
                    valid,
                    softmax_scale=softmax_scale,
                    kernel_config=config,
                )
            )
        else:
            raise ValueError("--pass direct-backward requires --backend thd or cute-native.")
        direct_backward_call = jax.jit(lambda d_arg: direct_backward(q, k, v, out, d_arg, lse))
        jax.block_until_ready(direct_backward_call(dout))
        for _ in range(args.warmup):
            jax.block_until_ready(direct_backward_call(dout))
        backward_ms = _time_call(lambda: direct_backward_call(dout), args.iterations)
        return _result(
            args,
            config,
            forward_ms=float("nan"),
            backward_ms=backward_ms,
            backward_with_forward_ms=float("nan"),
        )

    jax.block_until_ready(forward_jit(q, k, v))
    _, pullback = jax.vjp(forward_jit, q, k, v)
    backward_jit = jax.jit(lambda d_arg: pullback(d_arg))
    jax.block_until_ready(backward_jit(dout))
    jax.block_until_ready(backward_with_forward_jit(q, k, v, dout))

    for _ in range(args.warmup):
        jax.block_until_ready(forward_jit(q, k, v))
        jax.block_until_ready(backward_jit(dout))
        jax.block_until_ready(backward_with_forward_jit(q, k, v, dout))

    profile_dir = Path(args.profile_dir) if args.profile_dir else None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(profile_dir), create_perfetto_trace=True)

    try:
        forward_ms = _time_call(lambda: forward_jit(q, k, v), args.iterations)
        backward_ms = _time_call(lambda: backward_jit(dout), args.iterations)
        backward_with_forward_ms = _time_call(
            lambda: backward_with_forward_jit(q, k, v, dout),
            args.iterations,
        )
    finally:
        if profile_dir is not None:
            jax.profiler.stop_trace()

    return _result(
        args,
        config,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        backward_with_forward_ms=backward_with_forward_ms,
    )


def _result(
    args: argparse.Namespace,
    config: Flash4CuteKernelConfig,
    *,
    forward_ms: float,
    backward_ms: float,
    backward_with_forward_ms: float,
) -> BenchResult:
    flops_forward = _attention_flops(args.batch, args.seq_len, args.q_heads, args.head_dim, args.sliding_window)
    flops_backward = 2.5 * flops_forward
    peak_tflops = args.peak_tflops
    return BenchResult(
        label=args.label,
        batch=args.batch,
        seq_len=args.seq_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        sliding_window=args.sliding_window,
        dtype=args.dtype,
        backend=args.backend,
        config=asdict(config),
        warmup=args.warmup,
        iterations=args.iterations,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        backward_with_forward_ms=backward_with_forward_ms,
        flops_forward=flops_forward,
        flops_backward=flops_backward,
        forward_tflops=flops_forward / (forward_ms * 1e-3) / 1e12,
        backward_tflops=flops_backward / (backward_ms * 1e-3) / 1e12,
        forward_sol=flops_forward / (forward_ms * 1e-3) / 1e12 / peak_tflops,
        backward_sol=flops_backward / (backward_ms * 1e-3) / 1e12 / peak_tflops,
        pass_mode=args.pass_mode,
        native_mask_mode=args.native_mask_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="may208")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--backend", choices=("cute", "cute-native", "thd", "upstream"), default="cute")
    parser.add_argument("--pass", dest="pass_mode", choices=("full", "forward", "direct-backward"), default="full")
    parser.add_argument("--arch", type=int, default=90)
    parser.add_argument("--forward-tile")
    parser.add_argument("--backward-tile")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--backward-arch", help="Postprocess arch override, or 'infer' for legacy tile inference.")
    parser.add_argument("--sm90-tile")
    parser.add_argument("--sm90-num-threads", type=int)
    parser.add_argument("--sm90-stages-q", type=int)
    parser.add_argument("--sm90-stages-do", type=int)
    parser.add_argument("--sm90-stages-pds", type=int)
    parser.add_argument("--sm90-sdp-swap-ab")
    parser.add_argument("--sm90-dkv-swap-ab")
    parser.add_argument("--sm90-dq-swap-ab")
    parser.add_argument("--sm90-atom-layout-n-dkv", type=int)
    parser.add_argument("--sm90-atom-layout-m-dq", type=int)
    parser.add_argument("--sm90-dq-single-wg")
    parser.add_argument("--native-mask-mode", choices=("grug", "builtin-local"), default="grug")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=989.0)
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    result = run(args)
    payload = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
