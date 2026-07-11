# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate QuACK versus Pallas-Triton grouped-MLP BF16 numerics on one GPU."""

import argparse
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.sonic_quack import (
    _quack_gated_impl,
    _quack_grouped_concat_impl,
    _quack_grouped_impl,
    _require_quack,
)

_BF16_RTOL = 0.1
_BF16_ATOL = 2e-4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-xla", action="store_true")
    return parser


def _error_metrics(actual: jax.Array, expected: jax.Array) -> dict[str, float | int | bool]:
    actual_host = np.asarray(jax.device_get(actual), dtype=np.float32)
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    difference = np.abs(actual_host - expected_host)
    close = np.isclose(actual_host, expected_host, rtol=_BF16_RTOL, atol=_BF16_ATOL, equal_nan=False)
    reference_l2 = np.linalg.norm(expected_host.astype(np.float64).reshape(-1))
    difference_l2 = np.linalg.norm(difference.astype(np.float64).reshape(-1))
    return {
        "allclose": bool(np.all(close)),
        "mismatch_count": int(np.size(close) - np.count_nonzero(close)),
        "mismatch_fraction": float(1.0 - np.mean(close)),
        "mean_abs_error": float(np.mean(difference)),
        "max_abs_error": float(np.max(difference)),
        "reference_l2": float(reference_l2),
        "relative_l2_error": float(difference_l2 / reference_l2) if reference_l2 else float(difference_l2),
    }


def _pallas_mlp(
    x: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    preact = ragged_dot(x, w13, group_sizes, implementation="triton")
    gate, up = jnp.split(preact, 2, axis=-1)
    hidden = jax.nn.silu(gate) * up
    return preact, hidden, ragged_dot(hidden, w2, group_sizes, implementation="triton")


def _xla_mlp(
    x: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    preact = ragged_dot(x, w13, group_sizes, implementation="xla")
    gate, up = jnp.split(preact, 2, axis=-1)
    hidden = jax.nn.silu(gate) * up
    return preact, hidden, ragged_dot(hidden, w2, group_sizes, implementation="xla")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"reproducer requires exactly one GPU, found {jax.devices()}")
    _require_quack()
    for name in ("experts", "rows_per_expert", "hidden_dim", "intermediate_dim"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    key_x, key_w13, key_w2 = jax.random.split(jax.random.key(args.seed), 3)
    rows = args.experts * args.rows_per_expert
    x = jax.random.normal(key_x, (rows, args.hidden_dim), dtype=jnp.bfloat16)
    w13 = 0.02 * jax.random.normal(
        key_w13,
        (args.experts, args.hidden_dim, 2 * args.intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w2 = 0.02 * jax.random.normal(
        key_w2,
        (args.experts, args.intermediate_dim, args.hidden_dim),
        dtype=jnp.bfloat16,
    )
    group_sizes = jnp.full((args.experts,), args.rows_per_expert, dtype=jnp.int32)

    pallas_preact, pallas_hidden, pallas_output = jax.jit(_pallas_mlp)(x, w13, w2, group_sizes)
    _, quack_hidden = jax.jit(_quack_gated_impl)(x, w13, group_sizes)
    quack_preact = jax.jit(_quack_grouped_concat_impl)(x, w13, group_sizes)
    quack_output = jax.jit(_quack_grouped_impl)(quack_hidden, w2, group_sizes)

    quack_gate, quack_up = jnp.split(quack_preact, 2, axis=-1)
    quack_preact_jax_hidden = jax.nn.silu(quack_gate) * quack_up
    pallas_down_from_quack_hidden = jax.jit(
        lambda hidden, weights, sizes: ragged_dot(hidden, weights, sizes, implementation="triton")
    )(quack_hidden, w2, group_sizes)
    quack_down_from_pallas_hidden = jax.jit(_quack_grouped_impl)(pallas_hidden, w2, group_sizes)

    comparisons = {
        "w13_preact_quack_vs_pallas": _error_metrics(quack_preact, pallas_preact),
        "swiglu_quack_fused_vs_jax_from_quack_preact": _error_metrics(quack_hidden, quack_preact_jax_hidden),
        "hidden_quack_vs_pallas": _error_metrics(quack_hidden, pallas_hidden),
        "w2_quack_vs_pallas_shared_quack_hidden": _error_metrics(quack_output, pallas_down_from_quack_hidden),
        "w2_quack_vs_pallas_shared_pallas_hidden": _error_metrics(quack_down_from_pallas_hidden, pallas_output),
        "full_output_quack_vs_pallas": _error_metrics(quack_output, pallas_output),
    }
    if args.include_xla:
        xla_preact, xla_hidden, xla_output = jax.jit(_xla_mlp)(x, w13, w2, group_sizes)
        comparisons.update(
            {
                "w13_pallas_vs_xla": _error_metrics(pallas_preact, xla_preact),
                "w13_quack_vs_xla": _error_metrics(quack_preact, xla_preact),
                "hidden_pallas_vs_xla": _error_metrics(pallas_hidden, xla_hidden),
                "hidden_quack_vs_xla": _error_metrics(quack_hidden, xla_hidden),
                "full_output_pallas_vs_xla": _error_metrics(pallas_output, xla_output),
                "full_output_quack_vs_xla": _error_metrics(quack_output, xla_output),
            }
        )

    jax.block_until_ready((quack_output, pallas_output))
    return {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "shape": {
            "experts": args.experts,
            "rows_per_expert": args.rows_per_expert,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
        },
        "dtype": str(x.dtype),
        "activation": "silu(gate) * up / QuACK swiglu",
        "weight_layout": {
            "public_w13": "[experts, hidden, 2 * intermediate]",
            "public_w2": "[experts, intermediate, hidden]",
            "quack_storage": "swapaxes(weight, 1, 2); kernel view restores [K, N, experts]",
        },
        "accumulation": {
            "pallas_triton": "float32 accumulator, bfloat16 output",
            "quack": "Float32 accumulator, BFloat16 output",
        },
        "quack_activation_math": "FP32 fast exp2 plus approximate reciprocal for sigmoid, then BF16 output",
        "tolerances": {"rtol": _BF16_RTOL, "atol": _BF16_ATOL},
        "comparisons": comparisons,
    }


def main() -> None:
    result = _run(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
