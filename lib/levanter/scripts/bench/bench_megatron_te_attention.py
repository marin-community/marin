# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import statistics
from typing import Any, Literal


Backend = Literal["auto", "fused", "flash"]
Mode = Literal["forward", "forward_backward"]

BACKEND_ENV: dict[Backend, dict[str, str]] = {
    "auto": {
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_UNFUSED_ATTN": "1",
    },
    "fused": {
        "NVTE_FLASH_ATTN": "0",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_UNFUSED_ATTN": "0",
    },
    "flash": {
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
    },
}

BACKEND_SELECTION_ENV_KEYS = (
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_FUSED_ATTN_BACKEND",
    "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT",
    "NVTE_FUSED_ATTN_USE_FAv2_BWD",
)


@dataclass(frozen=True, slots=True)
class Shape:
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    backend: Backend
    mode: Mode
    batch: int
    seq_len: int
    total_tokens: int
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    max_seqlen: int
    segments: int
    mean_ms: float
    median_ms: float
    min_ms: float
    peak_memory_mib: float
    max_abs_vs_reference: float | None
    grad_max_abs_vs_reference: float | None


def _set_backend_env(backend: Backend, *, debug: bool) -> None:
    for key in BACKEND_SELECTION_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in BACKEND_ENV[backend].items():
        os.environ[key] = value
    if debug:
        os.environ["NVTE_DEBUG"] = "1"
        os.environ["NVTE_DEBUG_LEVEL"] = "2"


def _torch_dtype(torch: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _segment_lengths(shape: Shape, target_segment_len: int) -> list[int]:
    if target_segment_len <= 0:
        raise ValueError(f"target_segment_len must be positive, got {target_segment_len}")
    pattern = [
        max(1, target_segment_len - 37),
        target_segment_len + 19,
        max(1, target_segment_len // 2 + 11),
        target_segment_len + 73,
    ]
    lengths: list[int] = []
    for batch_idx in range(shape.batch):
        remaining = shape.seq_len
        segment_idx = 0
        while remaining > 0:
            length = min(remaining, pattern[(batch_idx + segment_idx) % len(pattern)])
            lengths.append(length)
            remaining -= length
            segment_idx += 1
    return lengths


def _cu_seqlens(torch: Any, lengths: list[int]) -> Any:
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + int(length))
    return torch.tensor(cumulative, dtype=torch.int32, device="cuda")


def _new_leaf(torch: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    tensor = torch.randn(shape, device="cuda", dtype=dtype)
    tensor.requires_grad_(True)
    return tensor


def _normalize_output(torch: Any, out: Any, shape: Shape) -> Any:
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 3:
        return out
    if out.ndim == 2:
        return out.reshape(out.shape[0], shape.q_heads, -1)
    raise ValueError(f"Unexpected TE output shape: {tuple(out.shape)}")


def _reference_thd(torch: Any, q: Any, k: Any, v: Any, lengths: list[int]) -> Any:
    import torch.nn.functional as F

    outputs = []
    start = 0
    repeat = q.shape[1] // k.shape[1]
    scale = 1.0 / (q.shape[-1] ** 0.5)
    for length in lengths:
        end = start + length
        q_i = q[start:end].transpose(0, 1).unsqueeze(0)
        k_i = k[start:end].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        v_i = v[start:end].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0, is_causal=True, scale=scale)
        outputs.append(out_i.squeeze(0).transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


def _call_attention(module: Any, q: Any, k: Any, v: Any, cu_seqlens: Any, max_seqlen: int) -> Any:
    return module(
        q,
        k,
        v,
        None,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        attn_mask_type="padding_causal",
        window_size=(-1, 0),
    )


def _time_cuda(torch: Any, fn, *, steps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def _benchmark_mode(
    torch: Any,
    module: Any,
    *,
    backend: Backend,
    mode: Mode,
    shape: Shape,
    dtype: Any,
    dtype_name: str,
    lengths: list[int],
    cu_seqlens: Any,
    steps: int,
    warmup: int,
    check_reference: bool,
) -> BenchmarkResult:
    total_tokens = shape.batch * shape.seq_len
    max_seqlen = max(lengths)
    q = _new_leaf(torch, (total_tokens, shape.q_heads, shape.head_dim), dtype)
    k = _new_leaf(torch, (total_tokens, shape.kv_heads, shape.head_dim), dtype)
    v = _new_leaf(torch, (total_tokens, shape.kv_heads, shape.head_dim), dtype)
    with torch.no_grad():
        sample_out = _normalize_output(torch, _call_attention(module, q, k, v, cu_seqlens, max_seqlen), shape)
        dout = torch.randn_like(sample_out)

    max_abs = None
    grad_max_abs = None
    if check_reference:
        with torch.no_grad():
            actual = _normalize_output(torch, _call_attention(module, q, k, v, cu_seqlens, max_seqlen), shape)
            expected = _reference_thd(torch, q, k, v, lengths)
            max_abs = float((actual.float() - expected.float()).abs().max().item())
            reference_cotangent = torch.randn_like(expected)
        actual = _normalize_output(torch, _call_attention(module, q, k, v, cu_seqlens, max_seqlen), shape)
        actual_loss = (actual.float() * reference_cotangent.float()).sum()
        actual_grads = torch.autograd.grad(actual_loss, (q, k, v))
        expected = _reference_thd(torch, q, k, v, lengths)
        expected_loss = (expected.float() * reference_cotangent.float()).sum()
        expected_grads = torch.autograd.grad(expected_loss, (q, k, v))
        grad_max_abs = max(
            float((actual_grad.float() - expected_grad.float()).abs().max().item())
            for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True)
        )

    def zero_grads() -> None:
        for tensor in (q, k, v):
            tensor.grad = None

    def forward() -> None:
        out = _call_attention(module, q, k, v, cu_seqlens, max_seqlen)
        if isinstance(out, tuple):
            out = out[0]
        # Keep grad-enabled forward timing representative without retaining outputs.
        out.sum().detach()

    def forward_backward() -> None:
        zero_grads()
        out = _normalize_output(torch, _call_attention(module, q, k, v, cu_seqlens, max_seqlen), shape)
        loss = (out.float() * dout.float()).sum()
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    times_ms = _time_cuda(
        torch,
        forward if mode == "forward" else forward_backward,
        steps=steps,
        warmup=warmup,
    )
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return BenchmarkResult(
        backend=backend,
        mode=mode,
        batch=shape.batch,
        seq_len=shape.seq_len,
        total_tokens=total_tokens,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        dtype=dtype_name,
        max_seqlen=max_seqlen,
        segments=len(lengths),
        mean_ms=statistics.fmean(times_ms),
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        peak_memory_mib=peak_memory_mib,
        max_abs_vs_reference=max_abs,
        grad_max_abs_vs_reference=grad_max_abs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Megatron-Core comparison target: Transformer Engine "
            "DotProductAttention in THD packed-sequence mode."
        )
    )
    parser.add_argument("--backend", choices=("auto", "fused", "flash"), default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--segment-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--debug-backend-selection", action="store_true")
    args = parser.parse_args()

    backend: Backend = args.backend
    _set_backend_env(backend, debug=args.debug_backend_selection)

    try:
        import torch
        from transformer_engine.pytorch import DotProductAttention
    except ImportError as exc:
        raise RuntimeError(
            "bench_megatron_te_attention.py requires PyTorch and transformer-engine with CUDA support."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    shape = Shape(
        batch=args.batch,
        seq_len=args.seq_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    dtype = _torch_dtype(torch, args.dtype)
    lengths = _segment_lengths(shape, args.segment_len)
    cu = _cu_seqlens(torch, lengths)

    module = DotProductAttention(
        num_attention_heads=shape.q_heads,
        kv_channels=shape.head_dim,
        num_gqa_groups=shape.kv_heads,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        window_size=(-1, 0),
    ).cuda()
    module.train()

    header = {
        "backend": backend,
        "env": {key: os.environ[key] for key in BACKEND_SELECTION_ENV_KEYS if key in os.environ},
        "device": torch.cuda.get_device_name(),
        "shape": asdict(shape),
        "segment_len": args.segment_len,
        "segments": len(lengths),
        "max_seqlen": max(lengths),
    }
    print(json.dumps(header))

    for mode in ("forward", "forward_backward"):
        result = _benchmark_mode(
            torch,
            module,
            backend=backend,
            mode=mode,
            shape=shape,
            dtype=dtype,
            dtype_name=args.dtype,
            lengths=lengths,
            cu_seqlens=cu,
            steps=args.steps,
            warmup=args.warmup,
            check_reference=args.check_reference,
        )
        print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
