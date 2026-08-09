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

"""Replace the connected multi-output pair-Map reverse region in Grug."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
from haliax.partitioning import set_mesh

from lib.tile_lifetime.benchmarks.xla_grug_pair_map_custom_call_smoke import (
    _compare_outputs,
    _mesh,
    _natural_train_step,
)
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _PASS_NAME,
    _TARGET_NAME,
    _compile_handler,
    _register_legacy_custom_call,
    generate_cpu_multi_output_handler,
    recover_multi_output_region_rewrite,
    replace_multi_output_region_with_custom_call,
    write_gzip_text,
)
from tile_lifetime.xla_hlo_recovery import form_pair_map_entry_region, recover_pair_map_regions


def _multi_output_region_index(hlo_text: str) -> int:
    report = recover_pair_map_regions(hlo_text)
    candidates = tuple(
        index
        for index, region in enumerate(report.regions)
        if len(form_pair_map_entry_region(hlo_text, region).outputs) > 1
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one multi-output pair-Map region, found {candidates}")
    return candidates[0]


def run_smoke(artifact_directory: Path | None) -> dict[str, Any]:
    """Execute a generic tuple-output replacement of one backward region."""
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    jax.config.update("jax_enable_compilation_cache", False)
    original_modules: list[str] = []
    transformed_modules: list[str] = []
    rewrites: list[Any] = []
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-backward-region-")
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
                rewrite = recover_multi_output_region_rewrite(
                    original,
                    _multi_output_region_index(original),
                )
                source = generate_cpu_multi_output_handler(rewrite.program)
                library = _compile_handler(source, directory)
                _register_legacy_custom_call(library)
                holder["library"] = library
                original_modules.append(original)
                rewrites.append(rewrite)
                transformed_text = replace_multi_output_region_with_custom_call(
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
        raise RuntimeError("expected exactly one Grug backward-region replacement")
    comparison = _compare_outputs(expected, actual)
    rewrite = rewrites[0]
    transformed_hlo = transformed_modules[0]
    target_occurrences = transformed_hlo.count(_TARGET_NAME)
    tuple_gets = transformed_hlo.count("get-tuple-element(%shuttle_generated_multi_output_region)")
    if target_occurrences != 1 or tuple_gets != len(rewrite.boundary.outputs) or call_count != 1:
        raise RuntimeError(
            "multi-output custom-call evidence mismatch: "
            f"targets={target_occurrences}, tuple_gets={tuple_gets}, executions={call_count}"
        )
    source = generate_cpu_multi_output_handler(rewrite.program)
    if artifact_directory is not None:
        write_gzip_text(artifact_directory / "original-pre-scheduler-hlo.txt.gz", original_modules[0])
        write_gzip_text(artifact_directory / "transformed-pre-scheduler-hlo.txt.gz", transformed_hlo)
    return {
        "kind": "xla_pre_scheduler_grug_backward_multi_output_custom_call_smoke",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cpu",
        "device_kind": jax.devices("cpu")[0].device_kind,
        "natural_frontend": "ordinary one-layer Grug train step",
        "selection": "unique generic pair-Map region with multiple externally used pointwise outputs",
        "region_inputs": tuple((value.instruction, value.shape) for value in rewrite.boundary.inputs),
        "region_outputs": tuple((value.instruction, value.shape) for value in rewrite.boundary.outputs),
        "external_users": rewrite.boundary.external_users,
        "internal_instruction_count": len(rewrite.boundary.internal_instructions),
        "has_explicit_sharding": rewrite.boundary.has_explicit_sharding,
        "has_side_effect": rewrite.boundary.has_side_effect,
        "generated_scalar_expressions": rewrite.program.scalar_expressions,
        "generated_handler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "custom_call_target": _TARGET_NAME,
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "tuple_get_element_count": tuple_gets,
        "custom_call_handler_executions": call_count,
        **comparison,
        "outputs_match": True,
        "explicit_warning": (
            "Disposable proof only: tuple ABI works here but still uses text mutation and removed legacy CPU ABI."
        ),
        "production_blockers": (
            "typed C++ connected-region replacement and dead instruction deletion",
            "supported multi-result XLA FFI handler",
            "generic output alias/sharding/effect transfer",
            "GPU lowering from the same multi-output semantic AST",
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
