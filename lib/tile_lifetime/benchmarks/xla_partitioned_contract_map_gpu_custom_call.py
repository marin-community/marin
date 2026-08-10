# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the generic partitioned-Contract correctness skeleton on one H100."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from tile_lifetime.cuda_partitioned_gemm_codegen import (
    audit_cuda_partitioned_gemm_source,
    generate_cuda_partitioned_gemm_ffi,
)
from tile_lifetime.jax_partitioned_gemm_ffi import (
    call_cuda_partitioned_gemm_ffi,
    compile_cuda_partitioned_gemm_ffi,
    evaluate_partitioned_gemm_jax,
    register_cuda_partitioned_gemm_ffi,
)
from tile_lifetime.partitioned_gemm_reference import (
    evaluate_partitioned_gemm_reference,
    partitioned_gemm_error_metrics,
)
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
)
from tile_lifetime.xla_partitioned_contract_map import plan_attached_partitioned_contract_maps

_DEFAULT_HLO = (
    Path(__file__).parent / "artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
    "transformed-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET_PREFIX = "shuttle.generic.partitioned_contract_map.h100"


def run_benchmark(
    *,
    hlo_gzip: Path,
    nvcc: Path,
    architecture: str,
    artifact_directory: Path,
    seed: int,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, Any]:
    """Compile, validate, and time generated versus decomposed ordinary JAX."""
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError(f"partitioned Contract benchmark requires exactly one GPU, found {devices}")
    hlo = gzip.decompress(hlo_gzip.read_bytes()).decode()
    gated = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix="shuttle.generic.low_rank_contract_map.generated.forward.partitioned_harness",
        reverse_target_prefix="shuttle.generic.low_rank_contract_map.generated.reverse.partitioned_harness",
    )
    post_gated = replace_generated_low_rank_contract_map_training(hlo, gated)
    plan = plan_attached_partitioned_contract_maps(post_gated, target_prefix=_TARGET_PREFIX)
    if len(plan.families) != 1:
        raise ValueError(f"expected one attached partitioned Contract family, found {len(plan.families)}")
    family = plan.families[0]
    generated = generate_cuda_partitioned_gemm_ffi(family.program, target=family.target)
    source_audit = audit_cuda_partitioned_gemm_source(generated)
    if not source_audit.command_buffer_eligible or source_audit.opaque_semantic_dependencies:
        raise ValueError(f"generated source failed clean-boundary audit: {source_audit}")

    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "generated_partitioned_contract.cu").write_text(generated.source + "\n")
    (artifact_directory / "post-gated-pre-scheduler-hlo.txt.gz").write_bytes(gzip.compress(post_gated.encode()))
    with tempfile.TemporaryDirectory(prefix="shuttle-partitioned-contract-build-") as temporary:
        library = compile_cuda_partitioned_gemm_ffi(
            generated,
            directory=Path(temporary),
            nvcc=nvcc,
            architecture=architecture,
        )
        register_cuda_partitioned_gemm_ffi(generated, library)

        rng = np.random.default_rng(seed)
        host_operands = tuple(
            np.asarray(rng.normal(0.0, 0.25, size=buffer.shape), dtype=ml_dtypes.bfloat16)
            for buffer in generated.abi.inputs
        )
        expected = evaluate_partitioned_gemm_reference(family.program, host_operands)
        operands = tuple(jnp.asarray(value) for value in host_operands)

        generated_function = jax.jit(lambda *values: call_cuda_partitioned_gemm_ffi(generated, values))
        natural_function = jax.jit(lambda *values: evaluate_partitioned_gemm_jax(family.program, values))
        generated_executable = generated_function.lower(*operands).compile()
        natural_executable = natural_function.lower(*operands).compile()
        actual = tuple(np.asarray(value) for value in generated_executable(*operands))
        natural = tuple(np.asarray(value) for value in natural_executable(*operands))
        repeated = tuple(np.asarray(value) for value in generated_executable(*operands))
        jax.block_until_ready(repeated)

        reference_metrics = partitioned_gemm_error_metrics(actual, expected)
        natural_metrics = partitioned_gemm_error_metrics(actual, natural)
        deterministic = tuple(np.array_equal(left, right) for left, right in zip(actual, repeated, strict=True))
        if not all(metric["maximum_absolute_error"] <= 0.03125 for metric in reference_metrics):
            raise ValueError(f"generated partitioned Contract disagrees with ordered CPU reference: {reference_metrics}")
        if not all(deterministic):
            raise ValueError("generated partitioned Contract is not bitwise deterministic")

        for _ in range(warmup):
            jax.block_until_ready(generated_executable(*operands))
            jax.block_until_ready(natural_executable(*operands))
        generated_samples: list[float] = []
        natural_samples: list[float] = []
        for sample in range(samples):
            first, second = (
                ((generated_executable, generated_samples), (natural_executable, natural_samples))
                if sample % 2 == 0
                else ((natural_executable, natural_samples), (generated_executable, generated_samples))
            )
            for executable, destination in (first, second):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    result = executable(*operands)
                jax.block_until_ready(result)
                destination.append((time.perf_counter_ns() - start) / iterations / 1.0e6)

    generated_median = float(np.median(generated_samples))
    natural_median = float(np.median(natural_samples))
    summary = {
        "kind": "generated_generic_partitioned_contract_map_h100_correctness_skeleton",
        "accepted_performance_result": False,
        "reason": (
            "bounded one-CTA scalar mainloop is a correctness skeleton; QuACK/CuTe remains the throughput candidate"
        ),
        "revision": _git_revision(),
        "device": str(devices[0]),
        "architecture": architecture,
        "seed": seed,
        "warmup": warmup,
        "iterations": iterations,
        "samples": samples,
        "program_semantic_digest": family.program.semantic_digest,
        "generated_semantic_digest": generated.semantic_digest,
        "generated_source_digest": generated.source_digest,
        "source_audit": source_audit.__dict__,
        "input_hashes": tuple(_array_digest(value) for value in host_operands),
        "reference_error": reference_metrics,
        "natural_jax_error": natural_metrics,
        "deterministic_outputs": deterministic,
        "generated_latency_ms": {
            "median": generated_median,
            "raw": generated_samples,
        },
        "ordinary_jax_latency_ms": {
            "median": natural_median,
            "raw": natural_samples,
        },
        "generated_over_ordinary_jax": generated_median / natural_median,
    }
    (artifact_directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlo-gzip", type=Path, default=_DEFAULT_HLO)
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Run the manual H100 correctness/performance harness."""
    arguments = _parse_args()
    summary = run_benchmark(
        hlo_gzip=arguments.hlo_gzip,
        nvcc=arguments.nvcc,
        architecture=arguments.architecture,
        artifact_directory=arguments.artifact_directory,
        seed=arguments.seed,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        samples=arguments.samples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
