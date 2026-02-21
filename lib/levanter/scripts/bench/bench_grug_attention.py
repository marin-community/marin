# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import time
import sys
import types
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp


def _load_attention_module():
    root = Path(__file__).resolve()
    attention_path = root.parents[2] / "src" / "levanter" / "grug" / "attention.py"
    package_root = attention_path.parents[1]
    fake_levanter = types.ModuleType("levanter")
    fake_levanter.__path__ = [str(package_root)]
    sys.modules["levanter"] = fake_levanter
    spec = importlib.util.spec_from_file_location("levanter_grug_attention", attention_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load attention module from {attention_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_once(
    fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, object], jnp.ndarray],
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: object,
    steps: int,
    do_backward: bool = False,
):
    if do_backward and q.dtype != jnp.float32:
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

    def _wait_for_grad(g):
        g[0].block_until_ready()
        g[1].block_until_ready()
        g[2].block_until_ready()

    start = time.perf_counter()
    if do_backward:

        def loss(xq: jnp.ndarray, xk: jnp.ndarray, xv: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(fn(xq, xk, xv, mask))

        grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        out = grad_fn(q, k, v)
        _wait_for_grad(out)
    else:
        out = fn(q, k, v, mask)
        out.block_until_ready()
    compile_s = time.perf_counter() - start

    start = time.perf_counter()
    if do_backward:
        for _ in range(steps):
            out = grad_fn(q, k, v)
            _wait_for_grad(out)
    else:
        for _ in range(steps):
            out = fn(q, k, v, mask)
            out.block_until_ready()
    steady_s = (time.perf_counter() - start) / steps
    return compile_s, steady_s, out


def _parse_block_values(raw: str, label: str) -> list[int | None]:
    values: list[int | None] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok == "auto":
            parsed = None
        else:
            parsed = int(tok)
            if parsed <= 0:
                raise ValueError(f"{label} must be positive, got {parsed}")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError(f"no valid values for {label}")
    return values


def _infer_auto_block_size(seq_len: int, block_override: int | None = None) -> int:
    max_block = block_override or 128
    if max_block <= 0:
        raise ValueError(f"auto block max must be positive, got {max_block}")

    cap = min(seq_len, max_block)
    for preferred in (16, 1):
        block = cap - (cap % preferred)
        while block >= preferred:
            if seq_len % block == 0:
                return block
            block -= preferred
    return 1


def _normalize_block_size(seq_len: int, requested: int) -> int:
    block = min(requested, seq_len)
    if block <= 0:
        raise ValueError(f"requested block size must be positive, got {requested}")
    while block > 1 and seq_len % block != 0:
        block -= 1
    return block


def _backward_fused_compatible(
    seq_q: int,
    seq_kv: int,
    bq: int | None,
    bk: int | None,
    bq_dkv: int | None,
    bk_dkv: int | None,
    bq_dq: int | None,
    bk_dq: int | None,
) -> bool:
    bq_norm = _infer_auto_block_size(seq_q, bq)
    bk_norm = _infer_auto_block_size(seq_kv, bk)
    bk_dkv_norm = _normalize_block_size(seq_kv, bk_dkv if bk_dkv is not None else bk_norm)
    bq_dq_norm = _normalize_block_size(seq_q, bq_dq if bq_dq is not None else bq_norm)
    _ = _normalize_block_size(seq_kv, bk_dq if bk_dq is not None else bk_norm)
    return seq_q // bq_dq_norm == seq_kv // bk_dkv_norm


def _segment_ids_for_benchmark(seq_len: int, batch: int, mode: str) -> jax.Array:
    base_segments = (jnp.arange(seq_len) // 32).astype(jnp.int32)

    if mode == "2d":
        return jnp.tile(base_segments[None, :], (batch, 1))

    if mode == "2d_diff":
        offsets = jnp.arange(batch, dtype=jnp.int32)[:, None]
        return (base_segments[None, :] + offsets) % 8

    if mode == "none":
        raise ValueError("segment-id mode cannot be none when building segment ids")

    raise ValueError(f"unsupported segment-id mode: {mode}")


def _benchmark_one(
    name: str,
    fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, object], jnp.ndarray],
    q,
    k,
    v,
    mask,
    steps: int,
    do_backward: bool,
):
    print(f"\n{name}")
    compile_s, steady_s, out = _run_once(fn, q, k, v, mask, steps=steps, do_backward=do_backward)
    print(f"  compile_s: {compile_s:.6f}")
    print(f"  steady_s:  {steady_s:.6f}")
    tokens = q.shape[0] * q.shape[1]
    metric_label = "fwd+bwd_tokens_per_s" if do_backward else "tokens_per_s"
    print(f"  {metric_label}: {tokens / steady_s:.2f}")
    return compile_s, steady_s, out


def _apply_regression_target(args: argparse.Namespace) -> None:
    if args.target is None:
        return

    target_overrides: dict[str, dict[str, object]] = {
        # Regression target for the previously failing backward path:
        # causal + sliding window + 2D segment ids.
        "gb10_segment_swa_backward": {
            "impls": "reference,pallas_gpu",
            "batch": 4,
            "seq": 4096,
            "heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "steps": 1,
            "dtype": "bf16",
            "benchmark_backward": True,
            "sliding_window": 1024,
            "segment_id_mode": "2d",
            "gpu_block_q": "auto",
            "gpu_block_k": "auto",
            "gpu_block_q_dkv": "auto",
            "gpu_block_kv_dkv": "auto",
            "gpu_block_q_dq": "auto",
            "gpu_block_kv_dq": "auto",
            "gpu_num_warps": None,
            "gpu_num_stages": None,
        },
        # Regression target for GB10 hd256 backward stability with auto dispatch.
        "gb10_hd256_backward_auto": {
            "impls": "reference,pallas_gpu",
            "batch": 8,
            "seq": 2048,
            "heads": 4,
            "kv_heads": 4,
            "head_dim": 256,
            "steps": 1,
            "dtype": "bf16",
            "benchmark_backward": True,
            "sliding_window": None,
            "segment_id_mode": "none",
            "gpu_block_q": "auto",
            "gpu_block_k": "auto",
            "gpu_block_q_dkv": "auto",
            "gpu_block_kv_dkv": "auto",
            "gpu_block_q_dq": "auto",
            "gpu_block_kv_dq": "auto",
            "gpu_num_warps": None,
            "gpu_num_stages": None,
        },
    }

    overrides = target_overrides[args.target]
    for key, value in overrides.items():
        setattr(args, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Grug attention implementations.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--seq", type=int, default=2048, help="Query/key/value sequence length.")
    parser.add_argument("--heads", type=int, default=4, help="Number of query heads.")
    parser.add_argument("--kv_heads", type=int, default=4, help="Number of KV heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension.")
    parser.add_argument("--steps", type=int, default=5, help="Steady-state timing steps.")
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16", help="Input dtype.")
    parser.add_argument(
        "--impls",
        default="reference,pallas_gpu",
        help="Comma-separated implementations to benchmark (e.g. reference,pallas_gpu,pallas_tpu).",
    )
    parser.add_argument(
        "--gpu-block-q",
        default="auto",
        help="Comma-separated block_q candidates for GPU attention (default: auto).",
    )
    parser.add_argument(
        "--gpu-block-k",
        default="auto",
        help="Comma-separated block_k candidates for GPU attention (default: auto).",
    )
    parser.add_argument(
        "--gpu-block-q-dkv",
        default="auto",
        help="Comma-separated backward dKV block_q candidates for GPU attention (default: auto).",
    )
    parser.add_argument(
        "--gpu-block-kv-dkv",
        default="auto",
        help="Comma-separated backward dKV block_kv candidates for GPU attention (default: auto).",
    )
    parser.add_argument(
        "--gpu-block-q-dq",
        default="auto",
        help="Comma-separated backward dQ block_q candidates for GPU attention (default: auto).",
    )
    parser.add_argument(
        "--gpu-block-kv-dq",
        default="auto",
        help="Comma-separated backward dQ block_kv candidates for GPU attention (default: auto).",
    )
    parser.add_argument("--gpu-num-warps", type=int, default=None, help="GPU attention num_warps override.")
    parser.add_argument("--gpu-num-stages", type=int, default=None, help="GPU attention num_stages override.")
    parser.add_argument(
        "--gpu-block-size-key",
        default="auto",
        help="Block-size dict key to use when passing GPU block configs (default: auto).",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Optional sliding-window size for masking. Set to 0 to disable.",
    )
    parser.add_argument(
        "--segment-id-mode",
        default="none",
        choices=["none", "2d", "2d_diff"],
        help="Segment-id mode for masked attention runs: none, 2d (same across batch), 2d_diff.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep forward+backward GPU block-size combinations.",
    )
    parser.add_argument(
        "--benchmark-backward",
        action="store_true",
        help="Benchmark forward+backward together instead of forward only.",
    )
    parser.add_argument(
        "--target",
        choices=["gb10_segment_swa_backward", "gb10_hd256_backward_auto"],
        default=None,
        help="Run a predefined regression benchmark target.",
    )
    args = parser.parse_args()
    _apply_regression_target(args)

    mod = _load_attention_module()
    attention = mod.attention
    AttentionMask = mod.AttentionMask
    AttentionBlockSizes = mod.BlockSizes

    dtype = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}[args.dtype]
    device = jax.default_backend()
    print(f"backend: {device}")
    print(f"jax devices: {jax.devices()}")
    print(
        f"config: batch={args.batch} seq={args.seq} heads={args.heads} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} dtype={args.dtype}"
    )
    if args.benchmark_backward and args.dtype != "f32":
        print("note: benchmark-backward upcasts inputs to f32 for stable grad capture")

    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads for grouped query handling in this benchmark.")

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (args.batch, args.seq, args.heads, args.head_dim), dtype=dtype)
    k = jax.random.normal(key, (args.batch, args.seq, args.kv_heads, args.head_dim), dtype=dtype)
    v = jax.random.normal(key, (args.batch, args.seq, args.kv_heads, args.head_dim), dtype=dtype)
    sliding_window = None if args.sliding_window in (None, 0) else args.sliding_window
    mask = AttentionMask.causal(sliding_window=sliding_window)
    if args.segment_id_mode != "none":
        segment_ids = _segment_ids_for_benchmark(args.seq, args.batch, args.segment_id_mode)
        mask = mask.with_segment_ids(segment_ids)

    print(f"segment-id mode: {args.segment_id_mode}")

    impl_names = [name.strip() for name in args.impls.split(",") if name.strip()]
    results = {}
    pallas_configs: list[
        tuple[tuple[int | None, int | None, int | None, int | None, int | None, int | None], float, float]
    ] = []

    for impl in impl_names:
        try:
            if impl == "pallas_gpu":
                gpu_block_q_values = _parse_block_values(args.gpu_block_q, "gpu_block_q")
                gpu_block_k_values = _parse_block_values(args.gpu_block_k, "gpu_block_k")
                if args.benchmark_backward:
                    gpu_block_q_dkv_values = _parse_block_values(args.gpu_block_q_dkv, "gpu_block_q_dkv")
                    gpu_block_kv_dkv_values = _parse_block_values(args.gpu_block_kv_dkv, "gpu_block_kv_dkv")
                    gpu_block_q_dq_values = _parse_block_values(args.gpu_block_q_dq, "gpu_block_q_dq")
                    gpu_block_kv_dq_values = _parse_block_values(args.gpu_block_kv_dq, "gpu_block_kv_dq")
                else:
                    gpu_block_q_dkv_values = [None]
                    gpu_block_kv_dkv_values = [None]
                    gpu_block_q_dq_values = [None]
                    gpu_block_kv_dq_values = [None]
            else:
                gpu_block_q_values = [None]
                gpu_block_k_values = [None]
                gpu_block_q_dkv_values = [None]
                gpu_block_kv_dkv_values = [None]
                gpu_block_q_dq_values = [None]
                gpu_block_kv_dq_values = [None]

            config_seen = False
            for bq in gpu_block_q_values:
                for bk in gpu_block_k_values:
                    for bq_dkv in gpu_block_q_dkv_values:
                        for bk_dkv in gpu_block_kv_dkv_values:
                            for bq_dq in gpu_block_q_dq_values:
                                for bk_dq in gpu_block_kv_dq_values:
                                    if impl == "pallas_gpu" and args.benchmark_backward:
                                        if not _backward_fused_compatible(
                                            args.seq,
                                            args.seq,
                                            bq,
                                            bk,
                                            bq_dkv,
                                            bk_dkv,
                                            bq_dq,
                                            bk_dq,
                                        ):
                                            continue
                                    if impl == "reference" or impl == "pallas_tpu":
                                        block_sizes = None
                                    else:
                                        block_cfg = AttentionBlockSizes(
                                            block_q=bq,
                                            block_k=bk,
                                            block_q_dkv=bq_dkv,
                                            block_kv_dkv=bk_dkv,
                                            block_q_dq=bq_dq,
                                            block_kv_dq=bk_dq,
                                            num_warps=args.gpu_num_warps,
                                            num_stages=args.gpu_num_stages,
                                        )
                                        block_sizes = {args.gpu_block_size_key: block_cfg}

                                    if impl == "reference":
                                        fn = lambda q_, k_, v_, m: attention(q_, k_, v_, m, implementation="reference")
                                        label = f"attention({impl})"
                                    elif impl == "pallas_gpu":
                                        fn = lambda q_, k_, v_, m, block_sizes_=block_sizes: attention(
                                            q_,
                                            k_,
                                            v_,
                                            m,
                                            implementation="pallas_gpu",
                                            block_sizes=block_sizes_,
                                        )
                                        label = (
                                            "attention("
                                            f"{impl}, bq={ 'auto' if bq is None else bq }, "
                                            f"bk={ 'auto' if bk is None else bk }, "
                                            f"bq_dkv={ 'auto' if bq_dkv is None else bq_dkv }, "
                                            f"bk_dkv={ 'auto' if bk_dkv is None else bk_dkv }, "
                                            f"bq_dq={ 'auto' if bq_dq is None else bq_dq }, "
                                            f"bk_dq={ 'auto' if bk_dq is None else bk_dq }"
                                            ")"
                                        )
                                    else:
                                        fn = lambda q_, k_, v_, m, impl_name=impl: attention(
                                            q_, k_, v_, m, implementation=impl_name
                                        )
                                        label = f"attention({impl})"

                                    compile_s, steady_s, out = _benchmark_one(
                                        label, fn, q, k, v, mask, steps=args.steps, do_backward=args.benchmark_backward
                                    )
                                    config_seen = True
                                    if impl == "pallas_gpu":
                                        pallas_configs.append(
                                            ((bq, bk, bq_dkv, bk_dkv, bq_dq, bk_dq), compile_s, steady_s)
                                        )
                                    else:
                                        results[impl] = {"compile_s": compile_s, "steady_s": steady_s}

            if not config_seen:
                print("  failed: no valid config combinations")
                continue
        except Exception as exc:  # pragma: no cover - backend-specific failures are expected
            print(f"  failed: {type(exc).__name__}: {exc}")
            continue

    if "reference" in results and len(results) > 1:
        ref = results["reference"]["steady_s"]
        for impl, metric in results.items():
            if impl == "reference":
                continue
            speedup = ref / metric["steady_s"]
            print(f"\nSpeedup {impl} vs reference: {speedup:.3f}x")

    if "pallas_gpu" in impl_names and pallas_configs:
        best = min(pallas_configs, key=lambda x: x[2])
        bq, bk, bq_dkv, bk_dkv, bq_dq, bk_dq = best[0]
        results["pallas_gpu"] = {"compile_s": best[1], "steady_s": best[2]}
        print(
            f"\nBest pallas_gpu config: bq={ 'auto' if bq is None else bq }, "
            f"bk={ 'auto' if bk is None else bk }, "
            f"bq_dkv={ 'auto' if bq_dkv is None else bq_dkv }, "
            f"bk_dkv={ 'auto' if bk_dkv is None else bk_dkv }, "
            f"bq_dq={ 'auto' if bq_dq is None else bq_dq }, "
            f"bk_dq={ 'auto' if bk_dq is None else bk_dq }, "
            f"compile={best[1]:.6f}s, steady={best[2]:.6f}s"
        )

        if "reference" in results:
            ref = results["reference"]["steady_s"]
            print(f"Best pallas_gpu speedup vs reference: {ref / best[2]:.3f}x")


if __name__ == "__main__":
    main()
