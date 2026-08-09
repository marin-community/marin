#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace one natural JAX reverse entry at XLA PRE_SCHEDULER."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.jax_streaming_attention_backward_ffi import (
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import recover_stablehlo_streaming_attention_backward
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    plan_streaming_attention_backward_hlo_replacement,
    replace_streaming_attention_backward_entry_with_custom_call,
)

_PASS_NAME = "shuttle_streaming_reverse_entry_v1"


def _error(actual: jax.Array, expected: jax.Array) -> dict[str, float]:
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def _hash(values: tuple[jax.Array, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Compile and execute baseline and transformed natural JAX VJPs."""
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("streaming reverse HLO replacement requires a CUDA JAX device")
    scale = args.head_dimension**-0.5
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=args.sequence,
        key_length=args.sequence,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        head_dimension=args.head_dimension,
        scale=scale,
    )
    source_stablehlo = export_debug_streaming_attention_backward(config)
    graph = import_stablehlo(source_stablehlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
    recovered = recover_stablehlo_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(
            query_tile_size=args.block_m,
            key_value_tile_size=args.block_n,
            pipeline_depth=args.num_stages,
        ),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=args.block_m,
        key_value_tile_size=args.block_n,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    target = (
        f"shuttle.streaming_reverse.entry_s{args.sequence}_d{args.head_dimension}_"
        f"bm{args.block_m}_bn{args.block_n}_v1"
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=target,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    compiled = compile_streaming_attention_backward_ffi(
        generated,
        repository=args.repository,
        directory=args.build_directory,
        nvcc=args.nvcc,
        architecture=args.architecture,
        triton_target=args.triton_target,
    )
    register_streaming_attention_backward_ffi(compiled)
    key = jax.random.key(20260809)
    arguments = tuple(
        jax.random.normal(fold_key, specification.shape, dtype=jnp.bfloat16)
        for fold_key, specification in zip(
            jax.random.split(key, len(generated.inputs)),
            generated.inputs,
            strict=True,
        )
    )
    reverse = jax.jit(causal_gqa_attention_vjp(config))
    jax.config.update("jax_enable_compilation_cache", False)
    baseline = reverse.lower(*arguments).compile()
    expected = baseline(*arguments)
    jax.block_until_ready(expected)

    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    original_modules: list[str] = []
    transformed_modules: list[str] = []
    plans: list[Any] = []

    def replace(serialized_module: bytes) -> bytes:
        module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
        original = module.to_string()
        plan = plan_streaming_attention_backward_hlo_replacement(original, program, generated)
        transformed_text = replace_streaming_attention_backward_entry_with_custom_call(
            original,
            plan,
            target=target,
        )
        transformed = hlo.hlo_module_from_text(transformed_text)
        original_modules.append(original)
        transformed_modules.append(transformed.to_string())
        plans.append(plan)
        return transformed.as_serialized_hlo_module_proto()

    xla.register_hlo_module_transformation(
        replace,
        name=_PASS_NAME,
        stage=xla.PipelineStage.PRE_SCHEDULER,
        platforms="cuda",
    )
    jax.clear_caches()
    try:
        transformed = jax.jit(causal_gqa_attention_vjp(config)).lower(*arguments).compile()
    finally:
        xla.clear_hlo_module_transformation(
            _PASS_NAME,
            stage=xla.PipelineStage.PRE_SCHEDULER,
            platforms="cuda",
        )
    actual = transformed(*arguments)
    jax.block_until_ready(actual)
    if len(original_modules) != 1 or len(transformed_modules) != 1 or len(plans) != 1:
        raise RuntimeError("expected exactly one whole-entry streaming reverse transformation")
    call_count = compiled.library.shuttle_streaming_attention_backward_ffi_call_count
    call_count.restype = ctypes.c_int
    executions = int(call_count())
    if executions < 1:
        raise RuntimeError("generated streaming reverse handler did not execute")

    plan = plans[0]
    result = {
        "kind": "xla_streaming_attention_backward_pre_scheduler_replacement",
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": args.architecture,
        "natural_frontend": "ordinary JAX tensor algebra differentiated by JAX",
        "state_policy": plan.state_policy.value,
        "maximum_vjp": plan.maximum_vjp,
        "reassociation": plan.reassociation,
        "semantic_fingerprint": plan.semantic_fingerprint,
        "score_scale": plan.provenance.score_scale,
        "contract_count": 1 + len(plan.provenance.reverse_contracts),
        "additive_fold_count": len(plan.provenance.additive_folds),
        "domain_restriction_recovered": plan.provenance.domain_restriction is not None,
        "input_roles": [value.role.value for value in plan.inputs],
        "output_roles": [value.role.value for value in plan.outputs],
        "custom_call_occurrences": transformed_modules[0].count(target),
        "handler_executions": executions,
        "errors": {
            role: _error(output, reference)
            for role, output, reference in zip(
                ("query", "key", "value"),
                actual,
                expected,
                strict=True,
            )
        },
        "baseline_hash": _hash(expected),
        "transformed_hash": _hash(actual),
        "runtime_imports": {
            "torch": "torch" in sys.modules,
            "triton": "triton" in sys.modules,
        },
    }
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    (args.artifact_directory / "source-vjp-stablehlo.mlir.bc").write_bytes(source_stablehlo)
    (args.artifact_directory / "original-pre-scheduler-hlo.txt.gz").write_bytes(
        gzip.compress(original_modules[0].encode(), mtime=0)
    )
    (args.artifact_directory / "transformed-pre-scheduler-hlo.txt.gz").write_bytes(
        gzip.compress(transformed_modules[0].encode(), mtime=0)
    )
    (args.artifact_directory / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--triton-target")
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_smoke(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
