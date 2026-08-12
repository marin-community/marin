# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed NVIDIA Transformer Engine contract for Target 1 evaluation."""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_CONTRACT_BYTES = 128 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"
BOUNDARIES = ("forward", "backward_recompute", "composed")
SHAPES = ((2048, 4096), (7, 13))

SOURCE_FILES = {
    ".gitmodules": "83d5c78811ca5b445b4ef2c5118a9a6ae691adf6339492cf2f495446df72b63d",
    "build_tools/VERSION.txt": "b7c9900f90f674500897abbc5467c0478cd7a7a7dc2e4e7797a85ae1cebe4ae6",
    "docs/installation.rst": "376f2cdb375c17fb68523b34135baaf543eb3bf3dfbc31a17dafdf9f043d2cc5",
    "setup.py": "a0330e68c1c7c6e21d0d9f39bd79fd41aa94287bb4dcb3641cdc5555cc3ff9d1",
    "transformer_engine/common/include/transformer_engine/normalization.h": (
        "9533ef14eb870236c3200c9f143973e870ef52c4d07ddeb74fc2c4acd948fbc2"
    ),
    "transformer_engine/common/include/transformer_engine/transformer_engine.h": (
        "29963fa4df150956b7775b0e80e88e657def80b22397eff7710a33c40b9a4c33"
    ),
    "transformer_engine/common/normalization/common.cpp": (
        "f7a24af63ca0110a888ebbf2a3ab5c209035be5973e708186a376d5990bf37a7"
    ),
    "transformer_engine/common/normalization/rmsnorm/rmsnorm_api.cpp": (
        "c04a5a420a703ca2bef415d53a6d828c4562932178f049e7222c2b9eca9cdf7f"
    ),
    "transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu": (
        "f48e50c92a98dff555900e7d162e5eca566c007e20644a54c4391542d980940d"
    ),
    "transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu": (
        "b12ba36f0cb5472e129552d2bd17dc3fd3b8031f967d1f9527bb181465308f4a"
    ),
}
SUBMODULES = {
    "3rdparty/cudnn-frontend": "e46d7082450264ce05cf898f8740011c4896f817",
    "3rdparty/cutlass": "57e3cfb47a2d9e0d46eb6335c3dc411498efa198",
    "3rdparty/googletest": "f8d7d77c06936315286eb55f8de22cd23c188571",
    "3rdparty/nccl": "a6b5de08b6af4f938cef541ae6e4d405632f89a4",
}
API_SIGNATURES = {
    "nvte_enable_cudnn_norm_bwd": {
        "declaration": "void nvte_enable_cudnn_norm_bwd(bool enable);",
        "sha256": "e9eae4d1eb0be3e100f0708a83a6219a344b10ae30420eff0d07cb718872e1db",
    },
    "nvte_enable_cudnn_norm_fwd": {
        "declaration": "void nvte_enable_cudnn_norm_fwd(bool enable);",
        "sha256": "ec72fa17f51a067b9bfd11305ffd86abd982651809d373607b9bbd828ccdc544",
    },
    "nvte_rmsnorm_bwd": {
        "declaration": (
            "void nvte_rmsnorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor rsigma, "
            "const NVTETensor gamma, NVTETensor dx, NVTETensor dgamma, NVTETensor workspace, "
            "const int multiprocessorCount, const bool zero_centered_gamma, cudaStream_t stream);"
        ),
        "sha256": "369e8d212738eb2bd7d47e81b1a17c3c05beeeb09551cf90cba2234476c22f98",
    },
    "nvte_rmsnorm_fwd": {
        "declaration": (
            "void nvte_rmsnorm_fwd(const NVTETensor x, const NVTETensor gamma, const float epsilon, "
            "NVTETensor z, NVTETensor rsigma, NVTETensor workspace, const int multiprocessorCount, "
            "const bool zero_centered_gamma, cudaStream_t stream);"
        ),
        "sha256": "f26efcdd25dc65c9e765044db89b2dae7c567c917f33f2dbcb26df5b5fa803c5",
    },
}

BACKEND_CONTROLS = {
    "dispatch_scope": "process_global_set_before_query_warmup_and_timing",
    "forward": {
        "api": "nvte_enable_cudnn_norm_fwd",
        "environment_alias": "NVTE_NORM_FWD_USE_CUDNN",
        "false_backend": "transformer_engine",
        "true_backend": "cudnn",
    },
    "backward": {
        "api": "nvte_enable_cudnn_norm_bwd",
        "environment_alias": "NVTE_NORM_BWD_USE_CUDNN",
        "false_backend": "transformer_engine",
        "true_backend": "cudnn",
    },
    "independence": "forward_and_backward_values_are_set_and_recorded_separately",
}

TIMING = {
    "warmup_invocations": 10,
    "measured_invocations": 50,
    "statistic": "median_cuda_event_elapsed_time",
    "stream": "one_explicit_nondefault_cuda_stream_per_process",
    "warmup_synchronization": "cudaStreamSynchronize_after_all_warmup_invocations",
    "measurement": [
        "cudaEventRecord(start,stream)",
        "one_complete_public_boundary_invocation",
        "cudaEventRecord(stop,stream)",
        "cudaEventSynchronize(stop)",
        "cudaEventElapsedTime(start,stop)",
    ],
    "excluded": [
        "adapter_and_library_compilation",
        "normalization_plan_construction",
        "workspace_queries",
        "workspace_allocation",
        "input_and_output_allocation",
        "warmup",
        "host_to_device_and_device_to_host_copies",
        "numerical_comparison",
    ],
}

COMPARISON = {
    "contract": {
        "id": "target1_rowwise_bf16_prerun_comparison_v1",
        "path": "lib/shuttle/mlir/jax_patch/target1-rowwise-bf16-prerun-comparison-v1.json",
    },
    "input_contract": "target1_rowwise_bf16_numerical_oracle_v1",
    "input_identity": "same_pinned_bfloat16_input_digests",
    "reference": "independent_numpy_binary64_closed_form_then_bfloat16_outputs",
    "outputs": {
        "forward": ["y"],
        "backward_recompute": ["dx", "dgamma"],
        "composed": ["y", "dx", "dgamma"],
    },
    "metrics": [
        "max_absolute_error",
        "mean_absolute_error",
        "relative_linf_error",
        "max_bfloat16_ulp_error",
    ],
    "subjects": ["transformer_engine", "shuttle_source_ordered", "shuttle_fast"],
    "acceptance_rule": "shuttle_error_le_max_matched_oracle_error_or_predeclared_dtype_floor",
    "threshold_status": "predeclared_source_ordered_and_identity_fast_nonidentity_fast_unresolved",
}


def _shape_tensors(rows: int, features: int) -> dict[str, dict[str, Any]]:
    matrix = {"shape": [rows, features], "strides_elements": [features, 1], "layout": "row_major_contiguous"}
    vector = {"shape": [features], "strides_elements": [1], "layout": "contiguous"}
    return {
        "x": {**matrix, "dtype": "bfloat16", "role": "input"},
        "gamma": {**vector, "dtype": "bfloat16", "role": "input"},
        "dy": {**matrix, "dtype": "bfloat16", "role": "input"},
        "y": {**matrix, "dtype": "bfloat16", "role": "output"},
        "dx": {**matrix, "dtype": "bfloat16", "role": "output"},
        "dgamma": {**vector, "dtype": "bfloat16", "role": "output"},
        "rsigma": {
            "shape": [rows],
            "strides_elements": [1],
            "layout": "contiguous",
            "dtype": "float32",
            "role": "adapter_private_forward_state",
        },
        "throwaway_z": {**matrix, "dtype": "bfloat16", "role": "adapter_private_recompute_output"},
    }


TENSORS = {
    "scaling_mode": "NVTE_DELAYED_TENSOR_SCALING",
    "storage": "device_contiguous_nonoverlapping",
    "shapes": {f"{rows}x{features}": _shape_tensors(rows, features) for rows, features in SHAPES},
}

WORKSPACE = {
    "query_protocol": "call_api_with_empty_workspace_numel_zero_then_allocate_returned_exact_shape_and_kNVTEByte_dtype",
    "query_phase": "once_per_api_backend_shape_device_before_warmup_and_timing",
    "queries": {
        "forward": {
            "api": "nvte_rmsnorm_fwd",
            "bindings": [
                "x",
                "gamma",
                "epsilon_float32",
                "z",
                "rsigma",
                "empty_workspace_numel_zero",
                "multiprocessor_count",
                False,
                "stream",
            ],
        },
        "backward": {
            "api": "nvte_rmsnorm_bwd",
            "bindings": [
                "dy",
                "x",
                "rsigma",
                "gamma",
                "dx",
                "dgamma",
                "empty_workspace_numel_zero",
                "multiprocessor_count",
                False,
                "stream",
            ],
        },
    },
    "metadata_rule": "each_call_uses_its_exact_queried_shape_even_if_raw_storage_capacity_is_shared",
    "artifact_fields": ["shape", "dtype", "byte_count"],
}

BOUNDARY_CALLS = {
    "forward": {
        "public_signature": "forward(x,gamma)->y",
        "timed_calls": [
            {
                "api": "nvte_rmsnorm_fwd",
                "bindings": [
                    "x",
                    "gamma",
                    "epsilon_float32",
                    "y",
                    "rsigma",
                    "forward_workspace",
                    "multiprocessor_count",
                    False,
                    "stream",
                ],
            }
        ],
        "state_policy": "rsigma_is_adapter_private_and_not_returned",
    },
    "backward_recompute": {
        "public_signature": "backward(x,gamma,dy)->(dx,dgamma)",
        "numerical_reference_boundary": "backward",
        "timed_calls": [
            {
                "api": "nvte_rmsnorm_fwd",
                "bindings": [
                    "x",
                    "gamma",
                    "epsilon_float32",
                    "throwaway_z",
                    "rsigma",
                    "forward_workspace",
                    "multiprocessor_count",
                    False,
                    "stream",
                ],
            },
            {
                "api": "nvte_rmsnorm_bwd",
                "bindings": [
                    "dy",
                    "x",
                    "rsigma",
                    "gamma",
                    "dx",
                    "dgamma",
                    "backward_workspace",
                    "multiprocessor_count",
                    False,
                    "stream",
                ],
            },
        ],
        "state_policy": "no_saved_public_state_recompute_rsigma_inside_timing_and_write_required_throwaway_z",
    },
    "composed": {
        "public_signature": "composed(x,gamma,dy)->(y,dx,dgamma)",
        "numerical_reference_boundary": "composed",
        "timed_calls": [
            {
                "api": "nvte_rmsnorm_fwd",
                "bindings": [
                    "x",
                    "gamma",
                    "epsilon_float32",
                    "y",
                    "rsigma",
                    "forward_workspace",
                    "multiprocessor_count",
                    False,
                    "stream",
                ],
            },
            {
                "api": "nvte_rmsnorm_bwd",
                "bindings": [
                    "dy",
                    "x",
                    "rsigma",
                    "gamma",
                    "dx",
                    "dgamma",
                    "backward_workspace",
                    "multiprocessor_count",
                    False,
                    "stream",
                ],
            },
        ],
        "state_policy": "save_rsigma_only_within_this_timed_composed_invocation_then_consume_in_backward",
    },
}

HARDWARE_RESULTS = {
    "status": "blocked_not_executed",
    "required_hardware": ["h100", "gb200_or_b200"],
    "runs": [],
    "missing": [
        "hardware_execution",
        "resolved_binary_and_dependency_identity",
        "queried_workspace_records",
        "numerical_results",
        "latency_results",
    ],
}


def validate_contract(document: object) -> None:
    """Reject any drift in the closed expert-oracle contract."""
    root = _closed_mapping(
        document,
        "contract",
        {
            "schema_version",
            "contract_id",
            "dispatch",
            "provider",
            "api",
            "constants",
            "tensors",
            "workspace",
            "backend_controls",
            "boundaries",
            "timing",
            "comparison",
            "artifact_provenance",
            "hardware_results",
            "scorecard_effect",
        },
    )
    _equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _equal(root["contract_id"], "target1_rowwise_bf16_te_2_17_expert_oracle_v1", "contract_id")
    _equal(
        root["dispatch"],
        {"key": "boundary", "accepted_values": list(BOUNDARIES), "workload_name_dispatch": False},
        "dispatch",
    )
    provider = _closed_mapping(
        root["provider"], "provider", {"distribution", "version", "source", "source_files", "submodules"}
    )
    _equal(provider["distribution"], "transformer_engine", "provider.distribution")
    _equal(provider["version"], "2.17.0", "provider.version")
    _equal(
        provider["source"],
        {
            "repository": "https://github.com/NVIDIA/TransformerEngine.git",
            "tag": "v2.17",
            "commit": SOURCE_COMMIT,
            "audit_checkout_state": "clean_detached_exact_commit_with_uninitialized_submodules",
        },
        "provider.source",
    )
    _equal(provider["source_files"], SOURCE_FILES, "provider.source_files")
    _equal(provider["submodules"], SUBMODULES, "provider.submodules")

    api = _closed_mapping(root["api"], "api", {"header", "signatures", "invariants"})
    _equal(api["header"], "transformer_engine/common/include/transformer_engine/normalization.h", "api.header")
    _equal(api["signatures"], API_SIGNATURES, "api.signatures")
    for name, signature in API_SIGNATURES.items():
        _equal(
            hashlib.sha256(signature["declaration"].encode()).hexdigest(),
            signature["sha256"],
            f"api.signatures.{name}.sha256",
        )
    _equal(
        api["invariants"],
        {
            "x_rank": 2,
            "epsilon_nonnegative": True,
            "dz_dtype_equals_gamma_dtype": True,
            "dx_dtype_equals_x_dtype": True,
            "dgamma_dtype_equals_gamma_dtype": True,
            "rsigma_dtype": "float32",
            "zero_centered_gamma": False,
        },
        "api.invariants",
    )
    _equal(root["constants"], {"epsilon_float32": 1e-5, "multiprocessor_count": "physical_device_sm_count"}, "constants")
    _equal(root["tensors"], TENSORS, "tensors")
    _equal(root["workspace"], WORKSPACE, "workspace")
    _equal(root["backend_controls"], BACKEND_CONTROLS, "backend_controls")
    _equal(root["boundaries"], BOUNDARY_CALLS, "boundaries")
    _equal(root["timing"], TIMING, "timing")
    _equal(root["comparison"], COMPARISON, "comparison")

    provenance = _closed_mapping(
        root["artifact_provenance"],
        "artifact_provenance",
        {"harness", "transformer_engine", "cuda", "cudnn", "device", "execution"},
    )
    expected_provenance = {
        "harness": [
            "marin_revision",
            "adapter_sha256",
            "jax_version",
            "jaxlib_version",
            "cuda_plugin_identity",
            "pjrt_identity",
            "xla_build_identity",
        ],
        "transformer_engine": [
            "distribution",
            "version",
            "source_tag",
            "source_commit",
            "submodule_commits",
            "wheel_or_source_build_identity",
            "build_flags",
            "compiler",
            "target_architectures",
            "resolved_library_path",
            "library_sha256",
            "elf_build_id",
            "soname",
            "resolved_shared_library_dependencies",
        ],
        "cuda": ["toolkit_version", "nvcc_version", "driver_version", "runtime_version"],
        "cudnn": ["compile_time_version", "runtime_version", "resolved_library_identity"],
        "device": ["model", "uuid", "compute_capability", "physical_sm_count", "multiprocessor_count_argument"],
        "execution": [
            "tensor_shapes",
            "tensor_strides",
            "tensor_layouts",
            "tensor_dtypes",
            "scaling_mode",
            "epsilon",
            "zero_centered_gamma",
            "stream_policy",
            "forward_backend",
            "backward_backend",
            "forward_workspace",
            "backward_workspace",
            "warmup",
            "timing",
            "synchronization",
            "boundary_call_sequence",
        ],
    }
    _equal(provenance, expected_provenance, "artifact_provenance")
    _equal(root["hardware_results"], HARDWARE_RESULTS, "hardware_results")
    _equal(
        root["scorecard_effect"],
        {
            "status_changed": False,
            "reason": (
                "Contract provenance is pinned, but no H100 or GB200/B200 numerical or performance artifact exists."
            ),
        },
        "scorecard_effect",
    )


def load_contract(path: Path) -> Mapping[str, Any]:
    """Load the contract while rejecting duplicate keys and oversized input."""
    payload = path.read_bytes()
    if len(payload) > MAX_CONTRACT_BYTES:
        raise ValueError("expert-oracle contract exceeds the byte limit")
    document = json.loads(payload, object_pairs_hook=_unique_object)
    validate_contract(document)
    return document


def verify_source_checkout(path: Path) -> None:
    """Verify the locally available official checkout against the pinned source."""
    commit = _git(path, "rev-parse", "HEAD^{commit}")
    if commit != SOURCE_COMMIT:
        raise ValueError("Transformer Engine source commit drifted")
    if _git(path, "status", "--porcelain"):
        raise ValueError("Transformer Engine source checkout is dirty")
    if "v2.17" not in _git(path, "tag", "--points-at", "HEAD").splitlines():
        raise ValueError("Transformer Engine source tag drifted")
    submodules = {}
    for line in _git(path, "submodule", "status", "--recursive").splitlines():
        status_commit, relative_path, *_ = line.split()
        if not status_commit.startswith("-"):
            raise ValueError("Transformer Engine audit submodule state drifted")
        submodules[relative_path] = status_commit[1:]
    if submodules != SUBMODULES:
        raise ValueError("Transformer Engine submodule commits drifted")
    if (path / "build_tools/VERSION.txt").read_text().strip() != "2.17.0":
        raise ValueError("Transformer Engine version drifted")
    for relative_path, expected_digest in SOURCE_FILES.items():
        source = path / relative_path
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"Transformer Engine source file drifted: {relative_path}")
    header = " ".join(
        (path / "transformer_engine/common/include/transformer_engine/normalization.h").read_text().split()
    )
    for name, signature in API_SIGNATURES.items():
        if signature["declaration"] not in header:
            raise ValueError(f"Transformer Engine API declaration drifted: {name}")


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _closed_mapping(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} fields drifted")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} drifted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
