# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX/Pallas port side for SonicMoE token gather/sum comparisons.

This file adapts the production port in ``levanter.grug.sonic_moe`` to the
same fixed-top-k reverse-scatter contract used by
``real_sonic_token_gather_sum.py``. It is deliberately separate from the real
Sonic file so the source boundary for the purported port is easy to inspect.
"""

from __future__ import annotations

import argparse
import inspect
import socket
from pathlib import Path
from typing import Any, Literal

from common import (
    TokenGatherSumConfig,
    add_common_arguments,
    config_from_args,
    emit_record,
    make_selected_experts,
    reverse_scatter_from_selected_experts,
    time_blocking_call,
)

PallasBackend = Literal[
    "xla",
    "pallas_triton",
    "pallas_triton_faithful",
    "pallas_triton_token_loop",
    "pallas_triton_token_kblock",
]


def _jax_dtype(dtype: str):
    import jax.numpy as jnp

    return jnp.bfloat16 if dtype == "bf16" else jnp.float32


def _make_inputs(config: TokenGatherSumConfig):
    import jax
    import jax.numpy as jnp

    key = jax.random.key(config.seed)
    k_x, k_logits = jax.random.split(key)
    dtype = _jax_dtype(config.dtype)
    dispatch_output = jax.random.normal(k_x, (config.assignments, config.hidden), dtype=dtype)
    selected_experts = make_selected_experts(config)
    dispatch_positions_np = reverse_scatter_from_selected_experts(selected_experts).reshape(config.tokens, config.topk)
    dispatch_positions = jnp.asarray(dispatch_positions_np, dtype=jnp.int32)
    if config.weighted:
        logits = jax.random.normal(k_logits, (config.tokens, config.topk), dtype=jnp.float32)
        combine_weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
    else:
        combine_weights = jnp.ones((config.tokens, config.topk), dtype=dtype)
    return dispatch_output, dispatch_positions, combine_weights


def _call_port(
    *,
    backend: PallasBackend,
    dispatch_output,
    dispatch_positions,
    combine_weights,
    config: TokenGatherSumConfig,
    token_block_size: int,
    hidden_block_size: int,
    k_block_size: int,
    num_warps: int,
    use_inline_fma: bool,
):
    import jax.numpy as jnp
    from levanter.grug.sonic_moe import (
        SonicGatherSumBlockSizes,
        _sonic_gather_sum_pallas_triton_faithful_call,
        _sonic_gather_sum_pallas_triton_token_kblock_call,
        _sonic_gather_sum_pallas_triton_token_loop_call,
        sonic_gather_sum,
        sonic_gather_sum_reference,
    )

    block_sizes = SonicGatherSumBlockSizes(
        token_block_size=token_block_size,
        hidden_block_size=hidden_block_size,
        k_block_size=k_block_size,
        kernel_repeat=config.kernel_repeat,
        num_warps=num_warps,
        use_inline_fma=use_inline_fma,
    )

    repeated_output = dispatch_output
    repeat_offsets = None
    if config.replicate_input:
        repeated_output = jnp.concatenate([dispatch_output] * config.kernel_repeat, axis=0)
        repeat_offsets = jnp.arange(config.kernel_repeat, dtype=jnp.int32) * config.assignments

    if backend == "xla":
        return sonic_gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
    if backend == "pallas_triton":
        return sonic_gather_sum(
            repeated_output,
            dispatch_positions,
            combine_weights,
            implementation="pallas_triton",
            block_sizes=block_sizes,
        )
    if backend == "pallas_triton_faithful":
        weights = combine_weights if config.weighted else None
        return _sonic_gather_sum_pallas_triton_faithful_call(
            repeated_output,
            dispatch_positions,
            weights,
            block_sizes=block_sizes,
            interpret=False,
            repeat_offsets=repeat_offsets,
        )
    if backend == "pallas_triton_token_loop":
        return _sonic_gather_sum_pallas_triton_token_loop_call(
            repeated_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=False,
            repeat_offsets=repeat_offsets,
        )
    if backend == "pallas_triton_token_kblock":
        return _sonic_gather_sum_pallas_triton_token_kblock_call(
            repeated_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=False,
            repeat_offsets=repeat_offsets,
        )
    raise ValueError(f"Unknown Pallas comparison backend: {backend!r}")


def _write_port_sources(directory: Path, *, backend: PallasBackend) -> list[dict[str, object]]:
    from levanter.grug import sonic_moe

    directory.mkdir(parents=True, exist_ok=True)
    source_map = {
        "xla": ["sonic_gather_sum_reference"],
        "pallas_triton": [
            "sonic_gather_sum",
            "sonic_gather_sum_pallas_triton",
            "_sonic_gather_sum_pallas_triton_call",
            "_gather_sum_pallas_triton_kernel",
        ],
        "pallas_triton_faithful": [
            "sonic_gather_sum_pallas_triton_faithful",
            "_sonic_gather_sum_pallas_triton_faithful_call",
            "_gather_sum_pallas_triton_faithful_kernel",
        ],
        "pallas_triton_token_loop": [
            "_sonic_gather_sum_pallas_triton_token_loop_call",
            "_gather_sum_pallas_triton_token_loop_kernel",
        ],
        "pallas_triton_token_kblock": [
            "_sonic_gather_sum_pallas_triton_token_kblock_call",
            "_gather_sum_pallas_triton_token_kblock_kernel",
        ],
    }
    written = []
    for name in source_map[backend]:
        target = directory / f"{name}.py"
        target.write_text(inspect.getsource(getattr(sonic_moe, name)), encoding="utf-8")
        written.append({"path": str(target), "kind": "source", "bytes": target.stat().st_size})
    return written


def _write_xla_dump_hint(directory: Path, *, backend: PallasBackend) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    hint = directory / "xla_dump_hint.txt"
    hint.write_text(
        'Set XLA_FLAGS="--xla_dump_to=/tmp/xla-sonic-pallas --xla_dump_hlo_as_text '
        '--xla_gpu_dump_llvmir" before running this script to capture the compiled artifacts '
        f"for backend={backend}. Current XLA builds reject --xla_gpu_dump_ptx; PTX is emitted "
        "alongside LLVM IR when GPU IR dumping is enabled.\n",
        encoding="utf-8",
    )
    return {"path": str(hint), "kind": "xla_dump_hint", "bytes": hint.stat().st_size}


def run_case(
    config: TokenGatherSumConfig,
    *,
    backend: PallasBackend,
    token_block_size: int,
    hidden_block_size: int,
    k_block_size: int,
    num_warps: int,
    use_inline_fma: bool,
    write_ir_dir: Path | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from levanter.grug.sonic_moe import sonic_gather_sum_reference

    dispatch_output, dispatch_positions, combine_weights = _make_inputs(config)
    reference = sonic_gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)

    def call():
        return _call_port(
            backend=backend,
            dispatch_output=dispatch_output,
            dispatch_positions=dispatch_positions,
            combine_weights=combine_weights,
            config=config,
            token_block_size=token_block_size,
            hidden_block_size=hidden_block_size,
            k_block_size=k_block_size,
            num_warps=num_warps,
            use_inline_fma=use_inline_fma,
        )

    jitted = jax.jit(call)
    compile_inclusive, timing, out = time_blocking_call(jitted, warmup=config.warmup, steps=config.steps)
    diff = out.astype(jnp.float32) - reference.astype(jnp.float32)
    record: dict[str, Any] = {
        "backend": f"ported_jax_{backend}",
        "source_path": "lib/levanter/src/levanter/grug/sonic_moe.py",
        "hostname": socket.gethostname(),
        "jax": jax.__version__,
        "platform": jax.default_backend(),
        **config.as_record(),
        **timing,
        "compile_inclusive_s": compile_inclusive,
        "token_block_size": token_block_size,
        "hidden_block_size": hidden_block_size,
        "k_block_size": k_block_size,
        "num_warps": num_warps,
        "use_inline_fma": use_inline_fma,
        "max_abs_vs_reference": float(jnp.max(jnp.abs(diff))),
        "mean_abs_vs_reference": float(jnp.mean(jnp.abs(diff))),
    }
    if write_ir_dir is not None:
        source_dir = write_ir_dir / f"ported_pallas_{backend}"
        record["source_files"] = _write_port_sources(source_dir, backend=backend)
        record["xla_dump_hint"] = _write_xla_dump_hint(source_dir, backend=backend)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "xla",
            "pallas_triton",
            "pallas_triton_faithful",
            "pallas_triton_token_loop",
            "pallas_triton_token_kblock",
        ),
        default="pallas_triton_token_kblock",
    )
    parser.add_argument("--token-block", type=int, default=16)
    parser.add_argument("--hidden-block", type=int, default=64)
    parser.add_argument("--k-block", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--use-inline-fma", action="store_true")
    parser.add_argument("--write-ir-dir", type=Path)
    add_common_arguments(parser)
    args = parser.parse_args()

    emit_record(
        run_case(
            config_from_args(args),
            backend=args.backend,
            token_block_size=args.token_block,
            hidden_block_size=args.hidden_block,
            k_block_size=args.k_block,
            num_warps=args.num_warps,
            use_inline_fma=args.use_inline_fma,
            write_ir_dir=args.write_ir_dir,
        )
    )


if __name__ == "__main__":
    main()
