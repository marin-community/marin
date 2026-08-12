# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate non-scorecard Transformer Engine oracle runs."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from target1_expert_oracle import SOURCE_COMMIT, load_contract, verify_source_checkout
from target1_numerical_oracle import fixed_inputs, independent_reference
from target1_prerun_comparison import (
    CONTRACT_ID as COMPARISON_CONTRACT_ID,
)
from target1_prerun_comparison import (
    load_contract as load_comparison_contract,
)
from target1_prerun_comparison import (
    require_reference_qualified,
)

RUNNER = Path(__file__).with_name("target1_te_oracle_runner.cpp")
BUILD_MANIFEST = Path(__file__).with_name("target1-te-oracle-runner-build-v1.json")
EXPERT_CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-te-2.17-expert-oracle-v1.json")
COMPARISON_CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-prerun-comparison-v1.json")
COMPARISON_CONTRACT_SHA256 = "3886fe875da9b6445a45d1e03eb32dec242bb59d0ec58a28854e319bbdb31845"
PLAN_ID = "shape_boundary_backend_alternating_v1"
SHAPES = ((2048, 4096), (7, 13))
BOUNDARIES = ("forward", "backward_recompute", "composed")
BACKEND_PAIRS = (
    ("transformer_engine", "transformer_engine"),
    ("cudnn", "cudnn"),
    ("transformer_engine", "cudnn"),
    ("cudnn", "transformer_engine"),
)
OUTPUT_ROLES = {
    "forward": ("y",),
    "backward_recompute": ("dx", "dgamma"),
    "composed": ("y", "dx", "dgamma"),
}
EXPECTED_RUNNER_SHA256 = "c0b1f96546ddad9d6fc7de0dfb7dee00fc7c8eccf99fcb29878a7c4d4995d536"
EXPECTED_BUILD_MANIFEST_SHA256 = "47ec616ed9d261999f9665fba9e970a2eea7943dd9dfc3ec9e3341d29d0f26ad"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes(), object_pairs_hook=_strict_object)


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields drifted")
    return value


def _exact_int(value: object, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} drifted")


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def validate_build_manifest(path: Path = BUILD_MANIFEST) -> dict[str, Any]:
    """Validate the pinned source/build state without claiming CUDA execution."""
    if _sha256(RUNNER) != EXPECTED_RUNNER_SHA256:
        raise ValueError("runner source digest drifted")
    if _sha256(path) != EXPECTED_BUILD_MANIFEST_SHA256:
        raise ValueError("runner build manifest digest drifted")
    document = _closed(
        _load_json(path),
        {
            "schema_version",
            "manifest_id",
            "runner",
            "provider",
            "comparison_contract",
            "static_abi_gate",
            "build",
            "execution",
        },
        "build manifest",
    )
    _exact_int(document["schema_version"], 1, "schema_version")
    expected = {
        "schema_version": 1,
        "manifest_id": "target1_te_2_17_oracle_runner_build_v1",
        "runner": {
            "path": "lib/shuttle/mlir/jax_patch/target1_te_oracle_runner.cpp",
            "sha256": EXPECTED_RUNNER_SHA256,
            "language": "c++20",
            "dispatch": "boundary_shape_and_independent_backend_flags_only",
        },
        "provider": {
            "distribution": "transformer_engine",
            "version": "2.17.0",
            "tag": "v2.17",
            "commit": SOURCE_COMMIT,
            "headers": [
                "transformer_engine/common/include/transformer_engine/transformer_engine.h",
                "transformer_engine/common/include/transformer_engine/normalization.h",
            ],
        },
        "comparison_contract": {
            "id": COMPARISON_CONTRACT_ID,
            "path": "lib/shuttle/mlir/jax_patch/target1-rowwise-bf16-prerun-comparison-v1.json",
            "sha256": COMPARISON_CONTRACT_SHA256,
        },
        "static_abi_gate": {
            "status": "passed_on_clean_official_checkout",
            "scope": "c++20_syntax_and_exact_public_header_signatures_no_link",
            "host": "darwin_arm64",
            "compiler_family": "apple_clang",
            "cuda_declarations": "test_only_stub_no_runtime_implementation",
        },
        "build": {
            "cuda_link_status": "blocked_not_attempted_on_macos",
            "required_libraries": ["transformer_engine", "cudart"],
            "resolved_compiler": None,
            "build_flags": None,
            "target_architectures": None,
            "runner_binary_sha256": None,
            "transformer_engine_library_sha256": None,
        },
        "execution": {
            "status": "blocked_not_executed",
            "hardware_results": [],
            "oracle_relative_thresholds": None,
            "scorecard_status_changed": False,
        },
    }
    if document != expected:
        raise ValueError("runner build manifest drifted")
    return document


def _write_bfloat16(path: Path, value: Any) -> None:
    path.write_bytes(value.view("uint16").tobytes(order="C"))


def prepare_run_plan(output: Path, runner_binary: Path) -> dict[str, Any]:
    """Write pinned BF16 inputs/references and a deterministic 24-run plan."""
    validate_build_manifest()
    load_contract(EXPERT_CONTRACT)
    load_comparison_contract(COMPARISON_CONTRACT)
    if output.exists():
        raise ValueError("run-plan output must not exist")
    output.mkdir(parents=True)
    cases = output / "cases"
    cases.mkdir()

    for rows, features in SHAPES:
        case = cases / f"{rows}x{features}"
        case.mkdir()
        x, gamma, dy = fixed_inputs(rows, features, "backward")
        y, dx, dgamma = independent_reference("composed", (x, gamma, dy))
        for name, value in (("x", x), ("gamma", gamma), ("dy", dy)):
            _write_bfloat16(case / f"{name}.bf16", value)
        for name, value in (("y", y), ("dx", dx), ("dgamma", dgamma)):
            _write_bfloat16(case / f"reference_{name}.bf16", value)

    runs = []
    position = 0
    for group, ((rows, features), boundary) in enumerate(
        shape_boundary for shape in SHAPES for shape_boundary in ((shape, item) for item in BOUNDARIES)
    ):
        ordered_pairs = BACKEND_PAIRS if group % 2 == 0 else tuple(reversed(BACKEND_PAIRS))
        for forward_backend, backward_backend in ordered_pairs:
            result = f"result-{position:02d}.json"
            arguments = [
                "--boundary",
                boundary,
                "--forward-backend",
                forward_backend,
                "--backward-backend",
                backward_backend,
                "--rows",
                str(rows),
                "--features",
                str(features),
                "--case-directory",
                f"cases/{rows}x{features}",
                "--output",
                result,
                "--counterbalance-id",
                PLAN_ID,
                "--counterbalance-position",
                str(position),
                "--comparison-contract-id",
                COMPARISON_CONTRACT_ID,
                "--comparison-contract-sha256",
                COMPARISON_CONTRACT_SHA256,
            ]
            runs.append(
                {
                    "position": position,
                    "shape": [rows, features],
                    "boundary": boundary,
                    "forward_backend": forward_backend,
                    "backward_backend": backward_backend,
                    "argv": [str(runner_binary), *arguments],
                    "result": result,
                }
            )
            position += 1
    plan = {
        "schema_version": 1,
        "status": "prepared_not_executed",
        "counterbalance_plan_id": PLAN_ID,
        "runner_binary": str(runner_binary),
        "runner_binary_sha256": None,
        "comparison_contract": {
            "id": COMPARISON_CONTRACT_ID,
            "path": COMPARISON_CONTRACT.name,
            "sha256": COMPARISON_CONTRACT_SHA256,
        },
        "runs": runs,
        "scorecard_status_changed": False,
    }
    (output / "run-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _validate_workspace(value: object, name: str) -> None:
    workspace = _closed(value, {"shape", "dtype", "byte_count"}, name)
    shape = workspace["shape"]
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError(f"{name}.shape drifted")
    if workspace["dtype"] != "byte":
        raise ValueError(f"{name}.dtype drifted")
    if type(workspace["byte_count"]) is not int or workspace["byte_count"] <= 0:
        raise ValueError(f"{name}.byte_count drifted")
    if workspace["byte_count"] != shape[0]:
        raise ValueError(f"{name}.byte_count does not match byte workspace shape")


def validate_result(path: Path, expected_run: dict[str, Any]) -> dict[str, Any]:
    """Validate one unsealed hardware observation against its planned invocation."""
    comparison_contract = load_comparison_contract(COMPARISON_CONTRACT)
    result = _closed(
        _load_json(path),
        {
            "schema_version",
            "status",
            "boundary",
            "shape",
            "tensor_contract",
            "backends",
            "counterbalance",
            "workspace_queries",
            "timing",
            "repeatability",
            "comparison",
            "provenance",
        },
        "result",
    )
    _exact_int(result["schema_version"], 1, "result.schema_version")
    if result["status"] != "unsealed_hardware_observation":
        raise ValueError("result status drifted")
    if result["boundary"] != expected_run["boundary"] or result["shape"] != expected_run["shape"]:
        raise ValueError("result invocation identity drifted")
    if result["backends"] != {
        "forward": expected_run["forward_backend"],
        "backward": expected_run["backward_backend"],
    }:
        raise ValueError("result backend identity drifted")
    if result["counterbalance"] != {
        "plan_id": PLAN_ID,
        "position": expected_run["position"],
        "execution_unit": "single_backend_pair",
    }:
        raise ValueError("result counterbalance identity drifted")
    _rows, features = expected_run["shape"]
    if result["tensor_contract"] != {
        "dtype": "bfloat16",
        "layout": "row_major_contiguous",
        "matrix_strides_elements": [features, 1],
        "vector_strides_elements": [1],
        "rsigma_dtype": "float32",
    }:
        raise ValueError("result tensor contract drifted")
    workspaces = _closed(result["workspace_queries"], {"forward", "backward"}, "workspace_queries")
    _validate_workspace(workspaces["forward"], "workspace_queries.forward")
    if expected_run["boundary"] == "forward":
        if workspaces["backward"] is not None:
            raise ValueError("forward result must not contain a backward workspace")
    else:
        _validate_workspace(workspaces["backward"], "workspace_queries.backward")
    timing = _closed(
        result["timing"],
        {
            "warmup_invocations",
            "measured_invocations",
            "synchronization",
            "raw_cuda_event_milliseconds",
            "median_cuda_event_milliseconds",
        },
        "timing",
    )
    _exact_int(timing["warmup_invocations"], 10, "timing.warmup_invocations")
    _exact_int(timing["measured_invocations"], 50, "timing.measured_invocations")
    if timing["synchronization"] != "cudaEventSynchronize(stop) per sample":
        raise ValueError("timing synchronization drifted")
    samples = timing["raw_cuda_event_milliseconds"]
    if not isinstance(samples, list) or len(samples) != 50:
        raise ValueError("timing raw sample count drifted")
    checked_samples = [_nonnegative_finite(value, "timing sample") for value in samples]
    recorded_median = _nonnegative_finite(timing["median_cuda_event_milliseconds"], "timing median")
    if recorded_median != statistics.median(checked_samples):
        raise ValueError("timing median drifted")
    if result["repeatability"] != {
        "post_timing_invocations": 3,
        "comparison": "bitwise_all_public_outputs",
        "placement": "outside_cuda_event_intervals_after_50_measured_invocations",
        "all_outputs_bitwise_equal": True,
    }:
        raise ValueError("result repeatability evidence drifted")
    comparison = _closed(
        result["comparison"],
        {
            "contract",
            "subject_id",
            "reference",
            "relative_scale_floor",
            "outputs",
            "qualification_status",
        },
        "comparison",
    )
    if comparison["contract"] != {
        "id": COMPARISON_CONTRACT_ID,
        "sha256": COMPARISON_CONTRACT_SHA256,
    }:
        raise ValueError("comparison contract identity drifted")
    if comparison["subject_id"] != "transformer_engine_2_17_exact_c_api":
        raise ValueError("comparison subject identity drifted")
    if comparison["reference"] != "independent_numpy_binary64_closed_form_then_bfloat16_outputs":
        raise ValueError("comparison reference drifted")
    if comparison["relative_scale_floor"] != 0.0078125:
        raise ValueError("comparison relative scale floor drifted")
    if comparison["qualification_status"] != "unsealed_runner_metrics_require_contract_validator":
        raise ValueError("comparison qualification status drifted")
    outputs = comparison["outputs"]
    if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(
        OUTPUT_ROLES[expected_run["boundary"]]
    ):
        raise ValueError("comparison output roles drifted")
    for output in outputs:
        record = _closed(output, {"role", "metrics"}, "comparison output")
        metrics = _closed(
            record["metrics"],
            {
                "max_absolute_error",
                "mean_absolute_error",
                "relative_linf_error",
                "max_bfloat16_ulp_error",
            },
            "comparison metrics",
        )
        checked_metrics = {
            name: _nonnegative_finite(metrics[name], f"comparison metrics.{name}")
            for name in (
                "max_absolute_error",
                "mean_absolute_error",
                "relative_linf_error",
            )
        }
        if checked_metrics["mean_absolute_error"] > checked_metrics["max_absolute_error"]:
            raise ValueError("comparison metrics.mean_absolute_error exceeds max_absolute_error")
        if type(metrics["max_bfloat16_ulp_error"]) is not int or metrics["max_bfloat16_ulp_error"] < 0:
            raise ValueError("comparison metrics.max_bfloat16_ulp_error drifted")
        try:
            require_reference_qualified(
                metrics,
                shape=f"{expected_run['shape'][0]}x{expected_run['shape'][1]}",
                role=record["role"],
                contract=comparison_contract,
            )
        except AssertionError as error:
            raise ValueError("comparison metric exceeds predeclared reference limit") from error
    provenance = _closed(
        result["provenance"],
        {
            "marin_revision",
            "adapter_sha256",
            "transformer_engine",
            "toolchain",
            "cuda",
            "device",
        },
        "provenance",
    )
    provider = _closed(
        provenance["transformer_engine"],
        {
            "version",
            "source_tag",
            "source_commit",
            "resolved_library_path",
            "library_sha256",
            "elf_build_id",
            "resolved_shared_library_dependencies",
        },
        "provenance.transformer_engine",
    )
    if (provider["version"], provider["source_tag"], provider["source_commit"]) != (
        "2.17.0",
        "v2.17",
        SOURCE_COMMIT,
    ):
        raise ValueError("result Transformer Engine source identity drifted")
    for field in (
        "resolved_library_path",
        "library_sha256",
        "elf_build_id",
        "resolved_shared_library_dependencies",
    ):
        if provider[field] is not None:
            raise ValueError(f"unreviewed provenance.transformer_engine.{field} is forbidden")
    if provenance["marin_revision"] is not None or provenance["adapter_sha256"] is not None:
        raise ValueError("unreviewed harness provenance is forbidden")
    toolchain = _closed(
        provenance["toolchain"],
        {"compiler", "build_flags", "target_architectures"},
        "provenance.toolchain",
    )
    if any(value is not None for value in toolchain.values()):
        raise ValueError("unreviewed toolchain provenance is forbidden")
    cuda = _closed(
        provenance["cuda"],
        {"toolkit_version", "nvcc_version", "driver_version", "runtime_version"},
        "provenance.cuda",
    )
    if cuda["toolkit_version"] is not None or cuda["nvcc_version"] is not None:
        raise ValueError("unreviewed CUDA build provenance is forbidden")
    for field in ("driver_version", "runtime_version"):
        if type(cuda[field]) is not int or cuda[field] <= 0:
            raise ValueError(f"provenance.cuda.{field} drifted")
    device = _closed(
        provenance["device"],
        {"ordinal", "model", "uuid", "compute_capability", "physical_sm_count"},
        "provenance.device",
    )
    if type(device["ordinal"]) is not int or device["ordinal"] < 0:
        raise ValueError("provenance.device.ordinal drifted")
    if type(device["physical_sm_count"]) is not int or device["physical_sm_count"] <= 0:
        raise ValueError("provenance.device.physical_sm_count drifted")
    for field in ("model", "uuid", "compute_capability"):
        if device[field] is not None:
            raise ValueError(f"unreviewed provenance.device.{field} is forbidden")
    return result


CUDA_STUB = """\
#ifndef CUDA_RUNTIME_API_H
#define CUDA_RUNTIME_API_H
#include <stddef.h>
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount };
static const int cudaSuccess = 0;
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaRuntimeGetVersion(int*);
#endif
"""


def static_abi_check(te_source: Path, compiler: str) -> None:
    """Syntax-check the runner against the exact official v2.17 public headers."""
    validate_build_manifest()
    verify_source_checkout(te_source)
    resolved_compiler = shutil.which(compiler)
    if resolved_compiler is None:
        raise ValueError(f"compiler is unavailable: {compiler}")
    with tempfile.TemporaryDirectory(prefix="target1-te-cuda-stub-") as directory:
        stub = Path(directory)
        (stub / "cuda_runtime_api.h").write_text(CUDA_STUB)
        subprocess.run(
            [
                resolved_compiler,
                "-std=c++20",
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{stub}",
                "-isystem",
                str(te_source / "transformer_engine/common/include"),
                str(RUNNER),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--runner-binary", type=Path, required=True)
    static = subparsers.add_parser("static-abi-check")
    static.add_argument("--te-source", type=Path, required=True)
    static.add_argument("--compiler", default="clang++")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        plan = prepare_run_plan(arguments.output, arguments.runner_binary)
        print(json.dumps({"status": plan["status"], "runs": len(plan["runs"])}, sort_keys=True))
    else:
        static_abi_check(arguments.te_source, arguments.compiler)
        print(
            json.dumps(
                {
                    "status": "passed_syntax_only_no_link",
                    "host": f"{platform.system().lower()}_{platform.machine().lower()}",
                    "te_commit": SOURCE_COMMIT,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
