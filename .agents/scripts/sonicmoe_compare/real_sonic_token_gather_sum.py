# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated upstream SonicMoE Triton token gather/sum kernel.

This mirrors ``sonicmoe/functional/reduction_over_k_gather.py`` at
Dao-AILab/sonic-moe commit ``cfbd65f39b980b85b878b3cccdacb09191e24993``.
It is intentionally kept separate from the JAX/Pallas port so source and IR
comparisons have a clean "real Sonic" side.
"""

from __future__ import annotations

import argparse
import inspect
import socket
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from typing import Any

from common import (
    TokenGatherSumConfig,
    add_common_arguments,
    config_from_args,
    emit_record,
    make_selected_experts,
    reverse_scatter_from_selected_experts,
    timing_stats,
)

SONICMOE_COMMIT = "cfbd65f39b980b85b878b3cccdacb09191e24993"

torch, triton, tl = None, None, None


def _run(command: list[str]) -> None:
    emit_record({"event": "run", "command": command})
    subprocess.run(command, check=True)


def install_dependencies() -> None:
    if find_spec("pip") is not None:
        base = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    else:
        uv = which("uv")
        if uv is None:
            raise RuntimeError("Neither python -m pip nor uv is available for dependency install")
        base = [uv, "pip", "install", "--python", sys.executable]
    _run([*base, "torch"])


def _powers_of_2(start: int, end: int) -> list[int]:
    values: list[int] = []
    value = 1
    while value < start:
        value *= 2
    while value <= end:
        values.append(value)
        value *= 2
    return values


def _load_torch_and_triton():
    import torch as torch_module
    import triton as triton_module
    import triton.language as tl_module

    return torch_module, triton_module, tl_module


def define_token_gather_sum_kernel() -> object:
    global torch, triton, tl
    torch, triton, tl = _load_torch_and_triton()

    def autotune_configs() -> list[object]:
        configs = []
        for block_h in _powers_of_2(256, 4096):
            for block_k in _powers_of_2(1, 128):
                for num_warps in (4, 8):
                    if block_k * block_h <= 32768:
                        configs.append(
                            triton.Config({"BLOCK_H": block_h, "BLOCK_K": block_k}, num_warps=num_warps, num_stages=4)
                        )
        return configs

    def prune_configs(configs, nargs, **kw):
        pruned_configs = []
        for config in configs:
            block_h = config.kwargs["BLOCK_H"]
            block_k = config.kwargs["BLOCK_K"]
            hidden = kw["H"]
            max_k = kw["MAX_K"]
            if (
                block_h <= triton.next_power_of_2(hidden)
                and block_k <= triton.next_power_of_2(max_k)
                and min(hidden * max_k, 1024) <= (block_h * block_k)
            ):
                pruned_configs.append(config)
        return configs if len(pruned_configs) == 0 else pruned_configs

    @triton.autotune(
        configs=autotune_configs(),
        key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
        prune_configs_by={"early_config_prune": prune_configs},
    )
    @triton.jit
    def token_gather_sum_kernel(
        x_ptr,
        w_ptr,
        m_perm_ptr,
        m_offset_ptr,
        repeat_offsets_ptr,
        out_ptr,
        T,
        H: tl.constexpr,
        MAX_K: tl.constexpr,
        stride_xM: tl.constexpr,
        stride_xH: tl.constexpr,
        stride_outT: tl.constexpr,
        stride_outH: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        w_is_None: tl.constexpr,
        is_varlen_K: tl.constexpr,
        REPEAT: tl.constexpr,
    ):
        pid_t = tl.program_id(axis=0)
        t_idx = pid_t.to(tl.int64)

        if is_varlen_K:
            m_start = tl.load(m_offset_ptr + t_idx).to(tl.int64)
            m_end = tl.load(m_offset_ptr + t_idx + 1).to(tl.int64)
            k_this_token = m_end - m_start
        else:
            m_start = MAX_K * t_idx
            k_this_token: tl.constexpr = MAX_K

        for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
            h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
            h_mask = h_idx < H
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)

            for repeat_index in tl.range(0, REPEAT):
                repeat_offset = tl.load(repeat_offsets_ptr + repeat_index).to(tl.int64)
                for k_tile in tl.range(tl.cdiv(k_this_token, BLOCK_K)):
                    k_offset = k_tile * BLOCK_K
                    k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)
                    k_mask = k_idx < k_this_token
                    m_abs = m_start + k_idx
                    perm_idx = tl.load(m_perm_ptr + m_abs, mask=k_mask, other=0).to(tl.int64) + repeat_offset

                    x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
                    x_mask = k_mask[:, None] & h_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
                    if w_is_None:
                        acc += tl.sum(x_vals, axis=0)
                    else:
                        w_vals = tl.load(w_ptr + m_abs, mask=k_mask, other=0.0).to(tl.float32)
                        acc += tl.sum(x_vals * w_vals[:, None], axis=0)

            out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
            tl.store(out_ptrs, acc * (1.0 / REPEAT), mask=h_mask)

    return token_gather_sum_kernel


def _time_kernel(
    kernel,
    *,
    x,
    weights,
    m_perm,
    m_offset,
    repeat_offsets,
    out,
    config: TokenGatherSumConfig,
) -> tuple[list[float], list[float]]:
    assert torch is not None

    def call() -> None:
        kernel[(config.tokens,)](
            x,
            weights,
            m_perm,
            m_offset,
            repeat_offsets,
            out,
            T=config.tokens,
            H=config.hidden,
            MAX_K=config.topk,
            stride_xM=x.stride(0),
            stride_xH=x.stride(1),
            stride_outT=out.stride(0),
            stride_outH=out.stride(1),
            w_is_None=(weights is None),
            is_varlen_K=False,
            REPEAT=config.kernel_repeat,
        )

    for _ in range(config.warmup):
        call()
    torch.cuda.synchronize()

    event_timings = []
    wall_timings = []
    for _ in range(config.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        wall_timings.append(time.perf_counter() - wall_start)
        event_timings.append(start.elapsed_time(end) / 1000.0)
    return event_timings, wall_timings


def _write_artifact(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(str(value), encoding="utf-8")


def _dump_triton_ir(kernel: object, directory: Path) -> list[dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    try:
        source = inspect.getsource(kernel.fn)
    except Exception:
        source = repr(kernel)
    source_path = directory / "real_sonic_token_gather_sum_kernel.py"
    _write_artifact(source_path, source)
    written.append({"path": str(source_path), "kind": "source", "bytes": source_path.stat().st_size})

    seen: set[int] = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 8:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        asm = getattr(value, "asm", None)
        if isinstance(asm, dict):
            for key, artifact in asm.items():
                artifact_path = directory / f"{path}.asm.{key}".replace("/", "_").replace(" ", "_")
                _write_artifact(artifact_path, artifact)
                written.append({"path": str(artifact_path), "kind": f"asm.{key}", "bytes": artifact_path.stat().st_size})

        if isinstance(value, dict):
            for index, (_key, child) in enumerate(value.items()):
                visit(child, f"{path}.dict_{index}", depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{type(value).__name__}_{index}", depth + 1)
            return
        try:
            attrs = vars(value)
        except TypeError:
            return
        for name, child in attrs.items():
            if name.startswith("__") or name in {"src", "module"}:
                continue
            visit(child, f"{path}.{name}", depth + 1)

    visit(kernel, "kernel", 0)
    return written


def run_case(config: TokenGatherSumConfig, *, write_ir_dir: Path | None = None) -> dict[str, Any]:
    kernel = define_token_gather_sum_kernel()
    assert torch is not None
    assert triton is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float32
    base_x = torch.randn(config.assignments, config.hidden, device=device, dtype=dtype)
    x = base_x
    if config.replicate_input and config.kernel_repeat > 1:
        x = torch.cat([base_x] * config.kernel_repeat, dim=0).contiguous()

    selected_experts = make_selected_experts(config)
    m_perm = torch.from_numpy(reverse_scatter_from_selected_experts(selected_experts)).to(device=device)
    m_offset = torch.empty(config.tokens + 1, device=device, dtype=torch.int32)
    weights = None
    if config.weighted:
        logits = torch.randn(config.tokens, config.topk, device=device, dtype=torch.float32)
        weights = torch.softmax(logits, dim=-1).to(dtype).reshape(-1).contiguous()
    if config.replicate_input:
        repeat_offsets = torch.arange(config.kernel_repeat, device=device, dtype=torch.int32) * config.assignments
    else:
        repeat_offsets = torch.zeros(config.kernel_repeat, device=device, dtype=torch.int32)
    out = torch.empty(config.tokens, config.hidden, device=device, dtype=dtype)

    event_timings, wall_timings = _time_kernel(
        kernel,
        x=x,
        weights=weights,
        m_perm=m_perm,
        m_offset=m_offset,
        repeat_offsets=repeat_offsets,
        out=out,
        config=config,
    )

    with torch.no_grad():
        gathered = base_x[m_perm.long()].reshape(config.tokens, config.topk, config.hidden)
        if weights is None:
            reference = gathered.sum(dim=1)
        else:
            reference = (gathered * weights.reshape(config.tokens, config.topk, 1)).sum(dim=1)
        max_abs = (out - reference).abs().max().item()
        mean_abs = (out - reference).abs().float().mean().item()

    record: dict[str, Any] = {
        "backend": "real_sonicmoe_triton_token_gather_sum",
        "source_commit": SONICMOE_COMMIT,
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        **config.as_record(),
        **timing_stats(event_timings),
        "wall": timing_stats(wall_timings),
        "max_abs_vs_reference": float(max_abs),
        "mean_abs_vs_reference": float(mean_abs),
    }
    if write_ir_dir is not None:
        record["ir_files"] = _dump_triton_ir(kernel, write_ir_dir / "real_sonic")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--write-ir-dir", type=Path)
    add_common_arguments(parser)
    args = parser.parse_args()

    if args.install_deps:
        install_dependencies()
    emit_record(run_case(config_from_args(args), write_ir_dir=args.write_ir_dir))


if __name__ == "__main__":
    main()
