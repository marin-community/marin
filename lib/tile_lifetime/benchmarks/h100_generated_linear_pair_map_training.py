# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark generated linear pair-Map forward/backward against pinned CODA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import torch
from coda.core.elementwise.functional import dswiglu_backward as coda_dswiglu_backward
from coda.core.gemm.functional import gemm as coda_gemm
from coda.core.gemm.functional import gemm_swiglu as coda_gemm_swiglu
from h100_plan_runtime import H100DenseBackend

from tile_lifetime import (
    DType,
    PairMapSavePolicy,
    TensorBinding,
    build_linear_pair_map_program,
    compile_linear_pair_map_training,
    pair_silu_product_expression,
    pair_tanh_product_expression,
)
from tile_lifetime.cute_pair_map_codegen import GeneratedCutePairMapVjp, generate_cute_pair_map_vjp
from tile_lifetime.reference import DenseDebugConfig

PINNED_CODA_REVISION = "8fa88065e541f6a5b52fb400d94d4be02f18c543"
CODA_COMPATIBLE_QUACK_REVISION = "02c7f69881737731173a6a009aeb6f032e449b61"
SHUTTLE_QUACK_BASE_REVISION = "84ef91df9bec87c7e4938517234fafb07ef844dd"
SHUTTLE_QUACK_PATCH_SHA256 = "40318b9b390e111c38f4838a50cf8913695c9f94142122b374bf09c220cfd9a1"


def _parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(dimension) for dimension in value.split("x"))
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxI")
    return dimensions


def _binding(tensor: torch.Tensor) -> TensorBinding:
    dtype = DType.BF16 if tensor.dtype is torch.bfloat16 else DType.FP32
    return TensorBinding(handle=tensor, shape=tuple(tensor.shape), dtype=dtype)


def _load_generated_pair_vjp(
    generated: GeneratedCutePairMapVjp,
    source_directory: Path,
) -> tuple[ModuleType, Path]:
    source_directory.mkdir(parents=True, exist_ok=True)
    module_name = f"shuttle_pair_vjp_{generated.digest}"
    source_path = source_directory / f"{module_name}.py"
    source_path.write_text(generated.source)
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import generated pair-Map VJP {source_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module, source_path


def _benchmark_variants(
    variants: tuple[tuple[str, Callable[[], None]], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
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
    records = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
        }
        for name, values in samples.items()
    }
    measurements: dict[str, object] = dict(records)
    generated_ms = records["generated"]["median_ms"]
    for name, _ in variants:
        if name == "generated":
            continue
        oracle_ms = records[name]["median_ms"]
        measurements[f"ratio_generated_to_{name}"] = generated_ms / oracle_ms
    return measurements


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
    }


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _torch_reference(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output_cotangent: torch.Tensor,
    *,
    mutation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preactivation_fp32 = torch.mm(activation, weight.mT, out_dtype=torch.float32)
    saved_preactivation = preactivation_fp32.bfloat16()
    forward_pairs = preactivation_fp32.view(activation.shape[0], weight.shape[0] // 2, 2)
    saved_pairs = saved_preactivation.float().view(activation.shape[0], weight.shape[0] // 2, 2)
    left_forward, right_forward = forward_pairs.unbind(dim=-1)
    left_saved, right_saved = saved_pairs.unbind(dim=-1)
    if mutation:
        output = torch.tanh(left_forward) * right_forward
        dleft = output_cotangent.float() * right_saved * (1.0 - torch.tanh(left_saved) ** 2)
        dright = output_cotangent.float() * torch.tanh(left_saved)
    else:
        sigmoid_forward = torch.sigmoid(left_forward)
        output = left_forward * sigmoid_forward * right_forward
        sigmoid_saved = torch.sigmoid(left_saved)
        silu_saved = left_saved * sigmoid_saved
        dleft = output_cotangent.float() * right_saved * (sigmoid_saved + silu_saved * (1.0 - sigmoid_saved))
        dright = output_cotangent.float() * silu_saved
    preactivation_cotangent = torch.stack((dleft, dright), dim=-1).flatten(-2).bfloat16()
    input_gradient = torch.mm(preactivation_cotangent, weight)
    weight_gradient = torch.mm(preactivation_cotangent.mT, activation)
    return saved_preactivation, output.bfloat16(), input_gradient, weight_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=_parse_shape, default=(2048, 4096, 14336))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mutation", choices=("silu", "tanh"), default="silu")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, default=Path("/tmp/shuttle-linear-pair-map"))
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--coda-revision", default=PINNED_CODA_REVISION)
    parser.add_argument("--shuttle-quack-base-revision", default=SHUTTLE_QUACK_BASE_REVISION)
    parser.add_argument("--shuttle-quack-patch-sha256", default=SHUTTLE_QUACK_PATCH_SHA256)
    parser.add_argument("--coda-compatible-quack-revision", default=CODA_COMPATIBLE_QUACK_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the generated linear pair-Map benchmark requires H100 CUDA")
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 9:
        raise RuntimeError(
            f"pinned CODA oracle requires compute capability 9.x, got {properties.major}.{properties.minor}"
        )

    m, k, features = args.shape
    pair_expression = pair_silu_product_expression() if args.mutation == "silu" else pair_tanh_product_expression()
    source = build_linear_pair_map_program(
        rows=m,
        reduction=k,
        features=features,
        pair_expression=pair_expression,
    )
    saved_training = compile_linear_pair_map_training(
        source,
        save_policy=PairMapSavePolicy.SAVE_PREACTIVATION,
    )
    recompute_training = compile_linear_pair_map_training(
        source,
        save_policy=PairMapSavePolicy.RECOMPUTE_PREACTIVATION,
    )
    recompute_contract = recompute_training.recompute
    if recompute_contract is None:
        raise AssertionError("recompute policy did not emit its required Contract")
    generated_vjp = generate_cute_pair_map_vjp(saved_training.pair_vjp)
    vjp_module, vjp_source_path = _load_generated_pair_vjp(generated_vjp, args.source_directory)

    generator = torch.Generator(device="cuda").manual_seed(20260808)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    left_weight = torch.randn((k, features), dtype=torch.bfloat16, device="cuda", generator=generator) / k**0.5
    right_weight = torch.randn((k, features), dtype=torch.bfloat16, device="cuda", generator=generator) / k**0.5
    interleaved_weight_transpose = torch.stack((left_weight, right_weight), dim=-1).reshape(k, 2 * features)
    interleaved_weight = interleaved_weight_transpose.mT.contiguous()
    output_cotangent = torch.randn((m, features), dtype=torch.bfloat16, device="cuda", generator=generator)
    preactivation = torch.empty((m, 2 * features), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, features), dtype=torch.bfloat16, device="cuda")
    preactivation_cotangent = torch.empty_like(preactivation)
    input_gradient = torch.empty_like(activation)
    weight_gradient = torch.empty_like(interleaved_weight)

    bindings: dict[str, TensorBinding] = {
        saved_training.activation.name: _binding(activation),
        saved_training.physical_interleaved_weight: _binding(interleaved_weight),
        saved_training.physical_interleaved_weight_transpose: _binding(interleaved_weight.mT),
        saved_training.preactivation: _binding(preactivation),
        saved_training.output.name: _binding(output),
        saved_training.output_cotangent: _binding(output_cotangent),
        saved_training.preactivation_cotangent: _binding(preactivation_cotangent),
        f"transpose.{saved_training.preactivation_cotangent}": _binding(preactivation_cotangent.mT),
        saved_training.input_gradient.output: _binding(input_gradient),
        saved_training.weight_gradient.output: _binding(weight_gradient),
    }
    config = DenseDebugConfig(
        sequence=m,
        hidden=k,
        intermediate=features,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
    )
    backend = H100DenseBackend(config, generated_source_directory=args.source_directory)

    def generated_saved_forward() -> None:
        backend.run_gemm(saved_training.forward, bindings)

    def generated_saved_backward() -> None:
        vjp_module.generated_pair_vjp(preactivation, output_cotangent, preactivation_cotangent)
        backend.run_gemm(saved_training.input_gradient, bindings)
        backend.run_gemm(saved_training.weight_gradient, bindings)

    def generated_saved_forward_backward() -> None:
        generated_saved_forward()
        generated_saved_backward()

    def generated_recompute_forward() -> None:
        backend.run_gemm(recompute_training.forward, bindings)

    def generated_recompute_backward() -> None:
        backend.run_gemm(recompute_contract, bindings)
        vjp_module.generated_pair_vjp(preactivation, output_cotangent, preactivation_cotangent)
        backend.run_gemm(recompute_training.input_gradient, bindings)
        backend.run_gemm(recompute_training.weight_gradient, bindings)

    def generated_recompute_forward_backward() -> None:
        generated_recompute_forward()
        generated_recompute_backward()

    coda_preactivation = torch.empty_like(preactivation)
    coda_output = torch.empty_like(output)
    coda_dpreactivation = torch.empty_like(preactivation)
    coda_input_gradient = torch.empty_like(activation)
    coda_weight_gradient = torch.empty_like(interleaved_weight)

    def coda_forward() -> None:
        coda_gemm_swiglu(
            activation,
            interleaved_weight.mT,
            pre_act=coda_preactivation,
            post_act=coda_output,
        )

    def coda_backward() -> None:
        coda_dswiglu_backward(coda_preactivation, output_cotangent, grad_pre=coda_dpreactivation)
        coda_gemm(coda_dpreactivation, interleaved_weight, out=coda_input_gradient)
        coda_gemm(coda_dpreactivation.mT, activation, out=coda_weight_gradient)

    def coda_forward_backward() -> None:
        coda_forward()
        coda_backward()

    generated_saved_forward_backward()
    first_hashes = {
        "output": _sha256(output),
        "input_gradient": _sha256(input_gradient),
        "weight_gradient": _sha256(weight_gradient),
    }
    generated_saved_forward_backward()
    second_hashes = {
        "output": _sha256(output),
        "input_gradient": _sha256(input_gradient),
        "weight_gradient": _sha256(weight_gradient),
    }
    if first_hashes != second_hashes:
        raise AssertionError("generated linear pair-Map training path is not deterministic")
    reference_preactivation, reference_output, reference_dx, reference_dw = _torch_reference(
        activation,
        interleaved_weight,
        output_cotangent,
        mutation=args.mutation == "tanh",
    )
    saved_correctness = {
        "preactivation": _error(preactivation, reference_preactivation),
        "output": _error(output, reference_output),
        "input_gradient": _error(input_gradient, reference_dx),
        "weight_gradient": _error(weight_gradient, reference_dw),
        "deterministic_hashes": first_hashes,
    }
    if args.mutation == "silu":
        coda_forward_backward()
        saved_correctness["generated_vs_coda"] = {
            "preactivation": _error(preactivation, coda_preactivation),
            "output": _error(output, coda_output),
            "input_gradient": _error(input_gradient, coda_input_gradient),
            "weight_gradient": _error(weight_gradient, coda_weight_gradient),
        }
    generated_recompute_forward_backward()
    recompute_first_hashes = {
        "output": _sha256(output),
        "input_gradient": _sha256(input_gradient),
        "weight_gradient": _sha256(weight_gradient),
    }
    generated_recompute_forward_backward()
    recompute_second_hashes = {
        "output": _sha256(output),
        "input_gradient": _sha256(input_gradient),
        "weight_gradient": _sha256(weight_gradient),
    }
    if recompute_first_hashes != recompute_second_hashes:
        raise AssertionError("generated recompute linear pair-Map path is not deterministic")
    recompute_correctness = {
        "preactivation": _error(preactivation, reference_preactivation),
        "output": _error(output, reference_output),
        "input_gradient": _error(input_gradient, reference_dx),
        "weight_gradient": _error(weight_gradient, reference_dw),
        "deterministic_hashes": recompute_first_hashes,
    }
    correctness = {
        PairMapSavePolicy.SAVE_PREACTIVATION.value: saved_correctness,
        PairMapSavePolicy.RECOMPUTE_PREACTIVATION.value: recompute_correctness,
    }

    benchmark_arguments = {
        "warmups": args.warmups,
        "repeats": args.repeats,
        "iterations": args.iterations,
    }
    if args.mutation == "silu":
        generated_saved_forward()
        coda_forward()
        measurements = {
            "forward_saved": _benchmark_variants(
                (("generated", generated_saved_forward), ("coda_matched", coda_forward)),
                **benchmark_arguments,
            ),
            "backward_saved_preactivation_available": _benchmark_variants(
                (("generated", generated_saved_backward), ("coda_matched", coda_backward)),
                **benchmark_arguments,
            ),
            "full_saved": _benchmark_variants(
                (("generated", generated_saved_forward_backward), ("coda_matched", coda_forward_backward)),
                **benchmark_arguments,
            ),
            "backward_recompute": _benchmark_variants(
                (("generated", generated_recompute_backward), ("coda_matched", coda_backward)),
                **benchmark_arguments,
            ),
            "full_recompute": _benchmark_variants(
                (("generated", generated_recompute_forward_backward), ("coda_matched", coda_forward_backward)),
                **benchmark_arguments,
            ),
        }
    else:
        measurements = {
            "full_saved_mutation": _benchmark_variants(
                (("generated", generated_saved_forward_backward),),
                **benchmark_arguments,
            ),
            "full_recompute_mutation": _benchmark_variants(
                (("generated", generated_recompute_forward_backward),),
                **benchmark_arguments,
            ),
        }
    acceptance: dict[str, object] | None = None
    if args.mutation == "silu":
        full_saved = measurements["full_saved"]
        full_recompute = measurements["full_recompute"]
        assert isinstance(full_saved, dict) and isinstance(full_recompute, dict)
        saved_ratio = float(full_saved["ratio_generated_to_coda_matched"])
        recompute_ratio = float(full_recompute["ratio_generated_to_coda_matched"])
        acceptance = {
            "matched_denominator": "coda_components_preallocated.full_forward_backward",
            "denominator_operations": (
                "CODA gemm_swiglu forward",
                "CODA dswiglu_backward",
                "CODA gemm dX",
                "CODA gemm dW",
            ),
            "excludes": ("allocation", "autograd dispatch", "optimizer update"),
            "primary_numerator": "generated.save_preactivation.full_forward_backward",
            "primary_ratio": saved_ratio,
            "recompute_candidate_ratio": recompute_ratio,
            "threshold": 1.2,
            "primary_passes": saved_ratio <= 1.2,
            "recompute_passes": recompute_ratio <= 1.2,
        }
    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    generated_sources: Mapping[str, str] = backend.generated_sources
    result = {
        "schema_version": 1,
        "workload": {
            "rows": m,
            "reduction": k,
            "features": features,
            "physical_projection_width": 2 * features,
            "map": args.mutation,
            "dtype": "bfloat16",
            "accumulation_dtype": "float32",
            "save_policies": [
                saved_training.save_policy.value,
                recompute_training.save_policy.value,
            ],
        },
        "natural_source": {
            "operations": [type(operation).__name__ for operation in source.operations],
            "output": source.outputs[0].name,
            "numerical_policy": "real_algebra_equivalent",
        },
        "generated_plan": {
            "saved_forward": saved_training.forward.name,
            "recompute_forward": recompute_training.forward.name,
            "recompute_backward_prefix": recompute_contract.name,
            "pair_vjp_digest": generated_vjp.digest,
            "input_gradient": saved_training.input_gradient.name,
            "weight_gradient": saved_training.weight_gradient.name,
        },
        "correctness": correctness,
        "measurements": measurements,
        "acceptance": acceptance,
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced_order": True,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": properties.name,
            "telemetry": telemetry,
        },
        "revisions": {
            "shuttle": args.shuttle_revision,
            "coda": args.coda_revision,
            "executed_quack": {
                "base_revision": args.shuttle_quack_base_revision,
                "patch_sha256": args.shuttle_quack_patch_sha256,
                "used_by": ("generated Shuttle", "CODA matched denominator in this single-process environment"),
            },
            "coda_compatible_quack_revision": args.coda_compatible_quack_revision,
            "coda_compatible_quack_note": (
                "source-lineage pin only; the matched denominator executes under the same patched QuACK runtime "
                "as Shuttle to avoid toolchain confounding"
            ),
        },
        "generated_source": {
            "pair_vjp": str(vjp_source_path),
            "pair_vjp_sha256": generated_vjp.digest,
            "quack_modules": sorted(generated_sources),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
