#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax==0.11.0",
#   "jaxlib==0.11.0",
# ]
# ///

"""Replace one recovered pair-Map region inside a natural Grug train step.

This extends the isolated entry-replacement smoke to an honest region-local
replacement in a 58-leaf training-step result. It remains a disposable CPU
proof: HLO text mutation and the legacy custom-call ABI are forbidden in the
production bridge.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import _make_train_step, initial_state
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _PASS_NAME,
    _TARGET_NAME,
    _compile_handler,
    _register_legacy_custom_call,
    _shape,
    generate_cpu_handler,
    recover_region_local_rewrite,
    replace_region_instruction_with_custom_call,
    write_gzip_text,
)
from tile_lifetime.xla_hlo_recovery import recover_pair_map_regions


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(
        devices.reshape((1, devices.size, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _natural_train_step() -> tuple[Any, Any, Any]:
    os.environ["RAGGED_DOT_IMPL"] = "xla"
    config = GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    optimizer = optax.sgd(1e-3)
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    state = initial_state(
        config,
        optimizer=optimizer,
        mp=policy,
        key=jax.random.PRNGKey(0),
        ema_beta=None,
    )
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
    batch = GrugLmExample(
        tokens=tokens % 64,
        loss_weight=jnp.ones(tokens.shape, dtype=jnp.float32),
        attn_mask=AttentionMask.causal(),
    )
    train_step = _make_train_step(
        optimizer,
        policy,
        z_loss_weight=0.0,
        ema_beta=None,
        watch_config=None,
    )
    return train_step, state, batch


def _forward_region_index(hlo_text: str) -> tuple[int, tuple[str, ...]]:
    """Select the row-preserving consumer structurally, not by source names."""
    report = recover_pair_map_regions(hlo_text)
    candidates: list[int] = []
    rejected: list[str] = []
    for index in range(len(report.regions)):
        try:
            rewrite = recover_region_local_rewrite(hlo_text, index)
        except ValueError as error:
            rejected.append(f"region {index}: {error}")
            continue
        _, output_dims = _shape(rewrite.target_shape)
        if output_dims[0] == rewrite.program.rows:
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError(f"expected one row-preserving pair-Map consumer, found {candidates}")
    return candidates[0], tuple(rejected)


def _compare_outputs(expected: Any, actual: Any) -> dict[str, Any]:
    expected_leaves = jax.tree.leaves(expected)
    actual_leaves = jax.tree.leaves(actual)
    if len(expected_leaves) != len(actual_leaves):
        raise RuntimeError("transformed train step changed the result tree")
    maximum = 0.0
    total = 0.0
    count = 0
    exact_leaves = 0
    nan_values = 0
    infinity_values = 0
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        expected_array = np.asarray(expected_leaf)
        actual_array = np.asarray(actual_leaf)
        if expected_array.shape != actual_array.shape or expected_array.dtype != actual_array.dtype:
            raise RuntimeError("transformed train step changed a result leaf type")
        if expected_array.tobytes() == actual_array.tobytes():
            exact_leaves += 1
        if np.issubdtype(expected_array.dtype, np.inexact):
            expected_nan = np.isnan(expected_array)
            actual_nan = np.isnan(actual_array)
            if not np.array_equal(expected_nan, actual_nan):
                raise RuntimeError("transformed train step changed NaN positions")
            expected_infinity = np.isinf(expected_array)
            actual_infinity = np.isinf(actual_array)
            if not np.array_equal(expected_infinity, actual_infinity):
                raise RuntimeError("transformed train step changed infinity positions")
            if not np.array_equal(expected_array[expected_infinity], actual_array[actual_infinity]):
                raise RuntimeError("transformed train step changed infinity values")
            nan_values += int(expected_nan.sum())
            infinity_values += int(expected_infinity.sum())
            finite = np.isfinite(expected_array) & np.isfinite(actual_array)
            difference = np.abs(actual_array[finite].astype(np.float64) - expected_array[finite].astype(np.float64))
            maximum = max(maximum, float(difference.max(initial=0.0)))
            total += float(difference.sum())
            count += difference.size
            if not np.allclose(actual_array, expected_array, rtol=2e-4, atol=2e-5, equal_nan=True):
                raise RuntimeError(f"floating result leaf mismatch: max_abs={float(difference.max())}")
        elif not np.array_equal(actual_array, expected_array):
            raise RuntimeError("integer or Boolean result leaf changed")
    return {
        "result_leaf_count": len(expected_leaves),
        "bitwise_equal_leaf_count": exact_leaves,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": total / count if count else 0.0,
        "matching_nan_value_count": nan_values,
        "matching_infinity_value_count": infinity_values,
    }


def run_smoke(artifact_directory: Path | None) -> dict[str, Any]:
    """Compile and execute unmodified and region-replaced train steps."""
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    original_modules: list[str] = []
    transformed_modules: list[str] = []
    rewrites: list[Any] = []
    rejected_regions: list[str] = []
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-region-call-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory
    try:
        with set_mesh(_mesh()):
            train_step, state, batch = _natural_train_step()
            transformed_state = jax.tree.map(jnp.array, state)
            transformed_batch = jax.tree.map(jnp.array, batch)
            baseline = train_step.lower(state, batch, compute_watch=False).compile()
            expected = baseline(state, batch)
            holder: dict[str, Any] = {}

            def replace(serialized_module: bytes) -> bytes | None:
                module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                region_index, rejected = _forward_region_index(original)
                rewrite = recover_region_local_rewrite(original, region_index)
                source = generate_cpu_handler(rewrite.program)
                library = _compile_handler(source, directory)
                _register_legacy_custom_call(library)
                holder["library"] = library
                original_modules.append(original)
                rewrites.append(rewrite)
                rejected_regions.extend(rejected)
                transformed_text = replace_region_instruction_with_custom_call(
                    original,
                    rewrite,
                    _TARGET_NAME,
                )
                transformed_module = hlo.hlo_module_from_text(transformed_text)
                transformed_modules.append(transformed_module.to_string())
                return transformed_module.as_serialized_hlo_module_proto()

            xla.register_hlo_module_transformation(
                replace,
                name=_PASS_NAME,
                stage=xla.PipelineStage.PRE_SCHEDULER,
                platforms="cpu",
            )
            jax.clear_caches()
            try:
                transformed = train_step.lower(state, batch, compute_watch=False).compile()
            finally:
                xla.clear_hlo_module_transformation(
                    _PASS_NAME,
                    stage=xla.PipelineStage.PRE_SCHEDULER,
                    platforms="cpu",
                )
            actual = transformed(transformed_state, transformed_batch)
            jax.block_until_ready(actual)
            library = holder["library"]
            call_count_function = library.shuttle_pair_map_smoke_call_count
            call_count_function.restype = ctypes.c_int
            call_count = int(call_count_function())
        if artifact_directory is not None:
            (directory / "generated_pair_map_handler.so").unlink()
    finally:
        if temporary is not None:
            temporary.cleanup()

    if len(original_modules) != 1 or len(transformed_modules) != 1 or len(rewrites) != 1:
        raise RuntimeError("expected exactly one Grug train-step replacement")
    comparison = _compare_outputs(expected, actual)
    transformed_hlo = transformed_modules[0]
    target_occurrences = transformed_hlo.count(_TARGET_NAME)
    if target_occurrences != 1 or call_count != 1:
        raise RuntimeError(
            f"region-local custom call evidence mismatch: occurrences={target_occurrences}, executions={call_count}"
        )
    rewrite = rewrites[0]
    generated_source = generate_cpu_handler(rewrite.program)
    if artifact_directory is not None:
        write_gzip_text(artifact_directory / "original-pre-scheduler-hlo.txt.gz", original_modules[0])
        write_gzip_text(artifact_directory / "transformed-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_pre_scheduler_grug_region_custom_call_smoke",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cpu",
        "device_kind": jax.devices("cpu")[0].device_kind,
        "natural_frontend": "ordinary one-layer Grug train step",
        "selection": "unique structurally recovered row-preserving two-Contract/Map/Contract region",
        "recovered_region_count": len(recover_pair_map_regions(original_modules[0]).regions),
        "unsupported_recovered_regions": rejected_regions,
        "replaced_entry_instruction": rewrite.target_instruction,
        "physical_operands": rewrite.operand_instructions,
        "physical_operand_shapes": rewrite.operand_shapes,
        "preserved_map_cast_boundary_count": len(rewrite.preserved_map_casts),
        "generated_scalar_expression": rewrite.program.scalar_expression,
        "generated_handler_sha256": hashlib.sha256(generated_source.encode()).hexdigest(),
        "custom_call_target": _TARGET_NAME,
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "custom_call_handler_executions": call_count,
        **comparison,
        "outputs_match": True,
        "explicit_warning": (
            "Disposable proof only: region-local but still an HLO text edit using the removed legacy CPU ABI."
        ),
        "production_blockers": (
            "typed C++ instruction replacement and dead-subgraph cleanup",
            "supported XLA FFI ABI with plan fingerprint and buffer metadata",
            "explicit sharding/aliasing/side-effect transfer for arbitrary target instructions",
            "supported multi-result FFI for the separately validated backward region",
            "GPU lowering from the recovered scalar AST",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke(args.artifact_directory)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    elif args.artifact_directory is not None:
        (args.artifact_directory / "summary.json").write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
