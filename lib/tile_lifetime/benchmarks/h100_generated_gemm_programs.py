# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark generated QuACK GEMM programs recovered from natural StableHLO."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import torch
from h100_plan_runtime import H100DenseBackend, _input_bindings, _tensor
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.library import rms_partial_epi, rstd_swiglu_epi, swiglu_mod
from quack.epilogue.ops import ColVecLoad
from quack.epilogue.rotary import rotary_cos_sin_load
from quack.operand_transform import a_transform, transform_a_operand

from shuttle.ir import DType
from tile_lifetime import (
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    compile_stablehlo_dense_transformer_region,
)
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.reference import DENSE_REGION_INPUT_NAMES, DenseDebugConfig, export_debug_dense_region
from tile_lifetime.runtime import RuntimeBufferSpec, TensorBinding


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile_fp32"})
def _named_scale_a(activation, inverse_rms):
    return activation * inverse_rms


@gemm_epilogue(ops={"cs": rotary_cos_sin_load("cs")}, mode="acc_pair")
def _named_rope_table(acc, cs):
    x, y = unpack(acc)
    cosine, sine = unpack(cs)
    return {"D": pack(x * cosine - y * sine, x * sine + y * cosine)}


@gemm_epilogue(
    ops={"rstd": ColVecLoad("rstd"), "cs": rotary_cos_sin_load("cs")},
    mode="acc_pair",
)
def _named_rstd_rope_table(acc, rstd, cs):
    x, y = unpack(acc * rstd)
    cosine, sine = unpack(cs)
    return {"D": pack(x * cosine - y * sine, x * sine + y * cosine)}


HISTORICAL_ORACLE_MS = {
    "qkv": 0.1272,
    "output_projection": 0.1037,
    "gate_up.consumer_prologue": 0.6509,
    "gate_up.consumer_epilogue": 0.6430,
    "down_projection": 0.3085,
    "next_qkv.consumer_prologue": 0.1467,
    "next_qkv.consumer_epilogue": 0.1357,
}


def _bindings(plan, config: DenseDebugConfig, backend: H100DenseBackend) -> dict[str, TensorBinding]:
    result = _input_bindings(plan, config)
    records = {record.value: record for record in plan.materializations}
    generator = torch.Generator(device="cuda").manual_seed(17)
    for record in plan.materializations:
        if record.disposition not in {
            MaterializationDisposition.MATERIALIZE,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
        }:
            continue
        binding = backend.allocate(RuntimeBufferSpec(record.value, record.shape, record.dtype))
        tensor = _tensor({record.value: binding}, record.value)
        if record.dtype is DType.BF16:
            tensor.copy_(torch.randn(record.shape, dtype=torch.bfloat16, device="cuda", generator=generator))
        else:
            tensor.zero_()
        result[record.value] = binding
    for skeleton in plan.skeletons:
        if not isinstance(skeleton, ReductionSkeleton):
            continue
        rows = records[skeleton.input].shape[0]
        binding = backend.allocate(RuntimeBufferSpec(skeleton.output, (rows,), DType.FP32))
        _tensor({skeleton.output: binding}, skeleton.output).uniform_(0.25, 1.75, generator=generator)
        result[skeleton.output] = binding
    aliases = {
        record.value: record
        for record in plan.materializations
        if record.disposition is MaterializationDisposition.ALIAS
    }
    changed = True
    while changed:
        changed = False
        for name, record in aliases.items():
            if name in result or record.alias_of not in result:
                continue
            assert record.alias_of is not None
            result[name] = backend.alias(
                RuntimeBufferSpec(name, record.shape, record.dtype),
                result[record.alias_of],
            )
            changed = True
    return result


def _benchmark_pair(
    generated: Callable[[], None],
    named: Callable[[], None],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    variants = (("generated", generated), ("named_same_semantics", named))
    for _ in range(warmups):
        for _, function in variants:
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    measurements = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
        }
        for name, values in samples.items()
    }
    measurements["ratio_generated_to_named"] = float(measurements["generated"]["median_ms"]) / float(
        measurements["named_same_semantics"]["median_ms"]
    )
    return measurements


def _named_oracle(
    skeleton: GemmSkeleton,
    placement: RowScalePlacement,
    bindings: Mapping[str, TensorBinding],
    backend: H100DenseBackend,
) -> Callable[[], None]:
    tile_shape = skeleton.physical_tile_shape
    cluster_shape = skeleton.cluster_shape
    pingpong = skeleton.pingpong
    assert tile_shape is not None and cluster_shape is not None and pingpong is not None
    activation = _tensor(bindings, skeleton.input).view(skeleton.shape[0], skeleton.shape[2])
    weight = _tensor(bindings, skeleton.weight)
    output = _tensor(bindings, skeleton.output).view(skeleton.shape[0], -1)
    common = {
        "tile_M": tile_shape[0],
        "tile_N": tile_shape[1],
        "cluster_M": cluster_shape[0],
        "cluster_N": cluster_shape[1],
        "pingpong": pingpong,
    }
    epilogue_operations = {attachment.operation for attachment in skeleton.epilogue}
    if {"residual_add", "multiply_gamma", "partial_sum_square"} <= epilogue_operations or {
        "add",
        "multiply",
        "partial_sum_square",
    } <= epilogue_operations:
        residual, gamma, partial_attachment = skeleton.epilogue
        residual_tensor = _tensor(bindings, residual.inputs[1]).view(skeleton.shape[0], skeleton.shape[1])
        epi_args = {
            "weight": _tensor(bindings, gamma.inputs[1]),
            "resid_out": _tensor(bindings, residual.outputs[0]).view(skeleton.shape[0], -1),
            "sqsum": _tensor(bindings, partial_attachment.outputs[0]).view(skeleton.shape[0], -1),
        }
        return partial(
            rms_partial_epi.gemm,
            activation,
            weight,
            output,
            residual_tensor,
            epi_args=epi_args,
            **common,
        )
    if "pairwise_swiglu" in epilogue_operations or "pairwise_map" in epilogue_operations:
        activated = output
        inverse_rms = _tensor(
            bindings,
            (
                skeleton.prologue[0].inputs[1]
                if placement is RowScalePlacement.CONSUMER_PROLOGUE
                else skeleton.epilogue[0].inputs[1]
            ),
        )
        if placement is RowScalePlacement.CONSUMER_EPILOGUE:
            return partial(
                rstd_swiglu_epi.gemm,
                activation,
                weight,
                None,
                epi_args={"rstd": inverse_rms, "postact": activated},
                **common,
            )
        strip = torch.empty(
            skeleton.shape[2] // tile_shape[2],
            skeleton.shape[0],
            dtype=torch.float32,
            device=activation.device,
        )
        bundle = transform_a_operand(
            _named_scale_a,
            activation,
            {"inverse_rms": strip},
            tile_shape[0],
            tile_shape[2],
        )

        def launch_gate_up() -> None:
            strip.copy_(inverse_rms[None, :])
            swiglu_mod.gemm(
                bundle,
                weight,
                None,
                epi_args={"postact": activated},
                transform_a=_named_scale_a,
                **common,
            )

        return launch_gate_up
    sine = _tensor(bindings, "rope_sine")
    cosine = _tensor(bindings, "rope_cosine")
    rotary_table = backend._rotary_table(sine, cosine)
    if "scale_row" in epilogue_operations:
        inverse_rms = _tensor(bindings, skeleton.epilogue[0].inputs[1])
        return partial(
            _named_rstd_rope_table.gemm,
            activation,
            weight,
            output,
            epi_args={"rstd": inverse_rms, "cs": rotary_table},
            **common,
        )
    if skeleton.prologue:
        inverse_rms = _tensor(bindings, skeleton.prologue[0].inputs[1])
        strip = torch.empty(
            skeleton.shape[2] // tile_shape[2],
            skeleton.shape[0],
            dtype=torch.float32,
            device=activation.device,
        )
        bundle = transform_a_operand(
            _named_scale_a,
            activation,
            {"inverse_rms": strip},
            tile_shape[0],
            tile_shape[2],
        )

        def launch_qkv() -> None:
            strip.copy_(inverse_rms[None, :])
            _named_rope_table.gemm(
                bundle,
                weight,
                output,
                epi_args={"cs": rotary_table},
                transform_a=_named_scale_a,
                **common,
            )

        return launch_qkv
    return partial(
        _named_rope_table.gemm,
        activation,
        weight,
        output,
        epi_args={"cs": rotary_table},
        **common,
    )


def _finite_outputs(skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> dict[str, object]:
    names = _output_names(skeleton, bindings)
    summary = {}
    for name in sorted(names):
        tensor = _tensor(bindings, name)
        finite = torch.isfinite(tensor).all().item()
        if not finite:
            raise AssertionError(f"generated program produced a non-finite value in {name}")
        summary[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    return summary


def _output_names(skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> tuple[str, ...]:
    names = {skeleton.output}
    names.update(output for attachment in skeleton.epilogue for output in attachment.outputs if output in bindings)
    return tuple(sorted(names))


def _clone_outputs(
    skeleton: GemmSkeleton,
    bindings: Mapping[str, TensorBinding],
) -> dict[str, torch.Tensor]:
    return {name: _tensor(bindings, name).clone() for name in _output_names(skeleton, bindings)}


def _compare_outputs(
    generated: Mapping[str, torch.Tensor],
    named: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    comparisons = {}
    for name in generated:
        difference = (generated[name].float() - named[name].float()).abs()
        comparisons[name] = {
            "maximum_absolute_error": difference.max().item(),
            "mean_absolute_error": difference.mean().item(),
        }
    return comparisons


def _component_name(index: int, placement: RowScalePlacement) -> str:
    if index == 0:
        return "qkv"
    if index == 2:
        return "output_projection"
    if index == 4:
        return f"gate_up.{placement.value}"
    if index == 5:
        return "down_projection"
    if index == 7:
        return f"next_qkv.{placement.value}"
    raise ValueError(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--quack-revision", required=True)
    args = parser.parse_args()
    config = DenseDebugConfig(
        sequence=args.sequence,
        hidden=4096,
        intermediate=14336,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
    )
    artifact = export_debug_dense_region(config)
    timings: dict[str, object] = {}
    correctness: dict[str, object] = {}
    generated_sources: dict[str, str] = {}
    for placement in (RowScalePlacement.CONSUMER_PROLOGUE, RowScalePlacement.CONSUMER_EPILOGUE):
        plan = compile_stablehlo_dense_transformer_region(
            artifact,
            input_names=DENSE_REGION_INPUT_NAMES,
            gemm_accumulation_dtype=DType.FP32,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
            rms_scale_placement=placement,
        ).plan
        backend = H100DenseBackend(config)
        bindings = _bindings(plan, config, backend)
        for index in (0, 2, 4, 5, 7):
            if placement is RowScalePlacement.CONSUMER_EPILOGUE and index in (0, 2, 5):
                continue
            skeleton = plan.skeletons[index]
            assert isinstance(skeleton, GemmSkeleton)
            name = _component_name(index, placement)
            function = partial(backend.run_gemm, skeleton, bindings)
            named_oracle = _named_oracle(skeleton, placement, bindings, backend)
            function()
            first_hashes = _finite_outputs(skeleton, bindings)
            generated_outputs = _clone_outputs(skeleton, bindings)
            function()
            second_hashes = _finite_outputs(skeleton, bindings)
            if first_hashes != second_hashes:
                raise AssertionError(f"generated program {name} is not bitwise deterministic")
            named_oracle()
            named_outputs = _clone_outputs(skeleton, bindings)
            comparison = _compare_outputs(generated_outputs, named_outputs)
            is_pairwise_scalar_ast = skeleton.epilogue[-1].operation == "pairwise_map"
            exceeds_tolerance = any(
                value["maximum_absolute_error"] > (0.125 if is_pairwise_scalar_ast else 0.0)
                or value["mean_absolute_error"] > (1e-4 if is_pairwise_scalar_ast else 0.0)
                for value in comparison.values()
            )
            if exceeds_tolerance:
                raise AssertionError(
                    f"generated program {name} differs from the same-semantics named QuACK oracle: {comparison}"
                )
            measurement = _benchmark_pair(
                function,
                named_oracle,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            oracle_ms = HISTORICAL_ORACLE_MS[name] if args.sequence == 2048 else None
            generated_measurement = measurement["generated"]
            assert isinstance(generated_measurement, dict)
            generated_measurement["historical_oracle_ms"] = oracle_ms
            generated_measurement["ratio_to_historical_oracle"] = (
                float(generated_measurement["median_ms"]) / oracle_ms if oracle_ms is not None else None
            )
            timings[name] = measurement
            correctness[name] = {
                "generated_deterministic_outputs": first_hashes,
                "generated_vs_named_same_semantics": comparison,
            }
        generated_sources.update(backend.generated_sources)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    source_root = args.json_output.parent / f"{args.json_output.stem}-generated-source"
    source_root.mkdir(parents=True, exist_ok=True)
    for digest, source in generated_sources.items():
        (source_root / f"shuttle_quack_{digest}.py").write_text(source)
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    result = {
        "schema_version": 1,
        "workload": {
            "sequence": config.sequence,
            "hidden": config.hidden,
            "intermediate": config.intermediate,
            "query_heads": config.query_heads,
            "key_value_heads": config.key_value_heads,
            "head_dimension": config.head_dimension,
            "dtype": "bfloat16",
            "accumulation_dtype": "float32",
        },
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "timings": timings,
        },
        "correctness": correctness,
        "sources": {
            "shuttle_revision": args.shuttle_revision,
            "quack_revision": args.quack_revision,
            "quack_patch_sha256": "40318b9b390e111c38f4838a50cf8913695c9f94142122b374bf09c220cfd9a1",
            "stablehlo_sha256": hashlib.sha256(artifact).hexdigest(),
            "generated_source_sha256": sorted(generated_sources),
        },
        "environment": {
            "gpu_telemetry": telemetry,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: timings[name] for name in sorted(timings)}, indent=2))


if __name__ == "__main__":
    main()
