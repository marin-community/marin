# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the non-executing Target 1 TE oracle harness."""

import copy
import json
from pathlib import Path

import pytest
from target1_te_oracle_harness import (
    BACKEND_PAIRS,
    BOUNDARIES,
    COMPARISON_CONTRACT_ID,
    COMPARISON_CONTRACT_SHA256,
    PLAN_ID,
    SHAPES,
    prepare_run_plan,
    validate_build_manifest,
    validate_result,
)


def test_build_manifest_keeps_link_execution_and_thresholds_blocked() -> None:
    manifest = validate_build_manifest()

    assert manifest["static_abi_gate"]["status"] == "passed_on_clean_official_checkout"
    assert manifest["build"]["cuda_link_status"] == "blocked_not_attempted_on_macos"
    assert manifest["build"]["runner_binary_sha256"] is None
    assert manifest["execution"] == {
        "status": "blocked_not_executed",
        "hardware_results": [],
        "oracle_relative_thresholds": None,
        "scorecard_status_changed": False,
    }


def test_prepared_plan_covers_shapes_boundaries_and_independent_backends(
    tmp_path: Path,
) -> None:
    plan = prepare_run_plan(tmp_path / "plan", Path("./target1_te_oracle_runner"))
    runs = plan["runs"]

    assert len(runs) == 24
    assert {tuple(run["shape"]) for run in runs} == set(SHAPES)
    assert {run["boundary"] for run in runs} == set(BOUNDARIES)
    for shape in SHAPES:
        for boundary in BOUNDARIES:
            matching = [
                (run["forward_backend"], run["backward_backend"])
                for run in runs
                if tuple(run["shape"]) == shape and run["boundary"] == boundary
            ]
            assert set(matching) == set(BACKEND_PAIRS)
    assert [run["position"] for run in runs] == list(range(24))
    assert [(runs[index]["forward_backend"], runs[index]["backward_backend"]) for index in range(0, 8)] == [
        *BACKEND_PAIRS,
        *reversed(BACKEND_PAIRS),
    ]
    assert all("workload" not in " ".join(run["argv"]) for run in runs)
    assert all(
        run["argv"][-8:]
        == [
            "--counterbalance-id",
            PLAN_ID,
            "--counterbalance-position",
            str(run["position"]),
            "--comparison-contract-id",
            COMPARISON_CONTRACT_ID,
            "--comparison-contract-sha256",
            COMPARISON_CONTRACT_SHA256,
        ]
        for run in runs
    )
    assert plan["comparison_contract"] == {
        "id": COMPARISON_CONTRACT_ID,
        "path": "target1-rowwise-bf16-prerun-comparison-v1.json",
        "sha256": COMPARISON_CONTRACT_SHA256,
    }
    large_case = tmp_path / "plan/cases/2048x4096"
    assert (large_case / "x.bf16").stat().st_size == 2048 * 4096 * 2
    assert (large_case / "gamma.bf16").stat().st_size == 4096 * 2
    assert (large_case / "dy.bf16").stat().st_size == 2048 * 4096 * 2
    assert (large_case / "reference_y.bf16").stat().st_size == 2048 * 4096 * 2
    assert (large_case / "reference_dx.bf16").stat().st_size == 2048 * 4096 * 2
    assert (large_case / "reference_dgamma.bf16").stat().st_size == 4096 * 2


def _valid_result(run: dict) -> dict:
    _, features = run["shape"]
    roles = {
        "forward": ["y"],
        "backward_recompute": ["dx", "dgamma"],
        "composed": ["y", "dx", "dgamma"],
    }[run["boundary"]]
    metrics = {
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_linf_error": 0.0,
        "max_bfloat16_ulp_error": 0,
    }
    return {
        "schema_version": 1,
        "status": "unsealed_hardware_observation",
        "boundary": run["boundary"],
        "shape": run["shape"],
        "tensor_contract": {
            "dtype": "bfloat16",
            "layout": "row_major_contiguous",
            "matrix_strides_elements": [features, 1],
            "vector_strides_elements": [1],
            "rsigma_dtype": "float32",
        },
        "backends": {
            "forward": run["forward_backend"],
            "backward": run["backward_backend"],
        },
        "counterbalance": {
            "plan_id": PLAN_ID,
            "position": run["position"],
            "execution_unit": "single_backend_pair",
        },
        "workspace_queries": {
            "forward": {"shape": [1024], "dtype": "byte", "byte_count": 1024},
            "backward": None if run["boundary"] == "forward" else {"shape": [2048], "dtype": "byte", "byte_count": 2048},
        },
        "timing": {
            "warmup_invocations": 10,
            "measured_invocations": 50,
            "synchronization": "cudaEventSynchronize(stop) per sample",
            "raw_cuda_event_milliseconds": [float(index + 1) for index in range(50)],
            "median_cuda_event_milliseconds": 25.5,
        },
        "repeatability": {
            "post_timing_invocations": 3,
            "comparison": "bitwise_all_public_outputs",
            "placement": "outside_cuda_event_intervals_after_50_measured_invocations",
            "all_outputs_bitwise_equal": True,
        },
        "comparison": {
            "contract": {
                "id": COMPARISON_CONTRACT_ID,
                "sha256": COMPARISON_CONTRACT_SHA256,
            },
            "subject_id": "transformer_engine_2_17_exact_c_api",
            "reference": "independent_numpy_binary64_closed_form_then_bfloat16_outputs",
            "relative_scale_floor": 0.0078125,
            "outputs": [{"role": role, "metrics": metrics} for role in roles],
            "qualification_status": "unsealed_runner_metrics_require_contract_validator",
        },
        "provenance": {
            "marin_revision": None,
            "adapter_sha256": None,
            "transformer_engine": {
                "version": "2.17.0",
                "source_tag": "v2.17",
                "source_commit": "2e559f062497bef768dfbe9d7e45548fadeca80a",
                "resolved_library_path": None,
                "library_sha256": None,
                "elf_build_id": None,
                "resolved_shared_library_dependencies": None,
            },
            "toolchain": {
                "compiler": None,
                "build_flags": None,
                "target_architectures": None,
            },
            "cuda": {
                "toolkit_version": None,
                "nvcc_version": None,
                "driver_version": 12080,
                "runtime_version": 12080,
            },
            "device": {
                "ordinal": 0,
                "model": None,
                "uuid": None,
                "compute_capability": None,
                "physical_sm_count": 132,
            },
        },
    }


def _run() -> dict:
    return {
        "position": 0,
        "shape": [2048, 4096],
        "boundary": "composed",
        "forward_backend": "transformer_engine",
        "backward_backend": "cudnn",
    }


def _replace(document: dict, path: tuple[str | int, ...], value: object) -> dict:
    result = copy.deepcopy(document)
    target = result
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), True, "schema_version"),
        (("shape",), [7, 13], "invocation identity"),
        (("tensor_contract", "dtype"), "float32", "tensor contract"),
        (("backends", "backward"), "transformer_engine", "backend identity"),
        (("counterbalance", "position"), 1, "counterbalance identity"),
        (
            ("workspace_queries", "forward", "dtype"),
            "float32",
            "workspace_queries.forward.dtype",
        ),
        (("timing", "warmup_invocations"), 9, "warmup_invocations"),
        (("timing", "raw_cuda_event_milliseconds"), [1.0] * 49, "raw sample count"),
        (("timing", "raw_cuda_event_milliseconds", 0), -1.0, "finite nonnegative"),
        (("timing", "median_cuda_event_milliseconds"), 25.0, "median drifted"),
        (("repeatability", "post_timing_invocations"), 2, "repeatability evidence"),
        (("repeatability", "comparison"), "numeric", "repeatability evidence"),
        (
            ("repeatability", "all_outputs_bitwise_equal"),
            False,
            "repeatability evidence",
        ),
        (("comparison", "contract", "id"), "other", "contract identity"),
        (("comparison", "contract", "sha256"), "0" * 64, "contract identity"),
        (
            ("comparison", "subject_id"),
            "ordinary_jax_disabled_shuttle",
            "subject identity",
        ),
        (("comparison", "outputs", 0, "role"), "dx", "output roles"),
        (
            ("comparison", "outputs", 0, "metrics", "max_absolute_error"),
            False,
            "finite nonnegative",
        ),
        (
            ("comparison", "outputs", 0, "metrics", "mean_absolute_error"),
            1.0,
            "mean_absolute_error exceeds max_absolute_error",
        ),
        (
            ("comparison", "outputs", 0, "metrics", "max_bfloat16_ulp_error"),
            1.0,
            "ulp_error",
        ),
        (
            ("comparison", "outputs", 0, "metrics", "max_bfloat16_ulp_error"),
            8,
            "predeclared reference limit",
        ),
        (
            ("provenance", "transformer_engine", "source_commit"),
            "0" * 40,
            "source identity",
        ),
        (("provenance", "toolchain", "compiler"), "nvcc", "toolchain provenance"),
        (("provenance", "device", "model"), "H100", "device.model"),
        (("provenance", "device", "physical_sm_count"), True, "physical_sm_count"),
    ],
)
def test_result_schema_rejects_unreviewed_or_mistyped_evidence(
    tmp_path: Path, path: tuple[str | int, ...], value: object, message: str
) -> None:
    run = _run()
    mutated = _replace(_valid_result(run), path, value)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(mutated))

    with pytest.raises(ValueError, match=message):
        validate_result(result, run)


def test_result_schema_accepts_blocked_unsealed_observation(tmp_path: Path) -> None:
    run = _run()
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_valid_result(run)))

    validated = validate_result(result, run)

    assert validated["status"] == "unsealed_hardware_observation"
    assert validated["comparison"]["qualification_status"] == "unsealed_runner_metrics_require_contract_validator"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", [], "shape drifted"),
        ("shape", [[1024]], "shape drifted"),
        ("shape", [0], "shape drifted"),
        ("shape", [-1], "shape drifted"),
        ("shape", [False], "shape drifted"),
        ("shape", [1024.0], "shape drifted"),
        ("shape", ["1024"], "shape drifted"),
        ("byte_count", 0, "byte_count drifted"),
        ("byte_count", -1, "byte_count drifted"),
        ("byte_count", False, "byte_count drifted"),
        ("byte_count", 1024.0, "byte_count drifted"),
        ("byte_count", "1024", "byte_count drifted"),
        ("byte_count", 1023, "does not match byte workspace shape"),
    ],
)
def test_result_schema_rejects_invalid_byte_workspace_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run = _run()
    mutated = _replace(_valid_result(run), ("workspace_queries", "forward", field), value)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(mutated))

    with pytest.raises(ValueError, match=message):
        validate_result(result, run)


def test_result_schema_accepts_equal_mean_and_max_absolute_error(
    tmp_path: Path,
) -> None:
    run = _run()
    result_document = _valid_result(run)
    for output in result_document["comparison"]["outputs"]:
        output["metrics"]["max_absolute_error"] = 0.001
        output["metrics"]["mean_absolute_error"] = 0.001
    result = tmp_path / "result.json"
    result.write_text(json.dumps(result_document))

    validate_result(result, run)
