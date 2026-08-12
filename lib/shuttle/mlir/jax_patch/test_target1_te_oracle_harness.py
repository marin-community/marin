# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the non-executing Target 1 TE oracle harness."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from target1_te_oracle_harness import (
    BACKEND_PAIRS,
    BOUNDARIES,
    COMPARISON_CONTRACT_ID,
    COMPARISON_CONTRACT_SHA256,
    PLAN_ID,
    SHAPES,
    _validate_raw_result,
    prepare_run_plan,
    seal_result,
    seal_run_plan,
    validate_build_manifest,
    validate_hardware_matrix,
    validate_result,
)


def test_build_manifest_keeps_link_execution_and_thresholds_blocked() -> None:
    manifest = validate_build_manifest()

    assert manifest["static_abi_gate"]["status"] == "passed_on_clean_official_checkout"
    assert manifest["build"]["cuda_link_status"] == "blocked_not_attempted_on_macos"
    assert manifest["build"]["runner_binary_sha256"] is None
    assert manifest["execution"] == {
        "status": "blocked_not_executed",
        "sealing_schema": "target1_te_execution_identity_v1",
        "required_provenance": [
            "marin_revision",
            "runner_binary_sha256",
            "harness_sha256",
            "build_manifest_sha256",
            "transformer_engine_library_elf_dependencies",
            "toolchain",
            "cuda",
            "cudnn",
            "device",
            "input_and_reference_digests",
        ],
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
        run["argv"][-22:]
        == [
            "--counterbalance-id",
            PLAN_ID,
            "--counterbalance-position",
            str(run["position"]),
            "--comparison-contract-id",
            COMPARISON_CONTRACT_ID,
            "--comparison-contract-sha256",
            COMPARISON_CONTRACT_SHA256,
            "--input-digest-set",
            run["input_digest_set"],
            "--input-x-sha256",
            run["input_digests"]["x"],
            "--input-gamma-sha256",
            run["input_digests"]["gamma"],
            "--input-dy-sha256",
            run["input_digests"].get("dy", "none"),
            "--reference-y-sha256",
            run["reference_digests"].get("y", "none"),
            "--reference-dx-sha256",
            run["reference_digests"].get("dx", "none"),
            "--reference-dgamma-sha256",
            run["reference_digests"].get("dgamma", "none"),
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
            "input_digests": run["input_digests"],
            "input_digest_set": run["input_digest_set"],
            "reference_digests": run["reference_digests"],
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
        "input_digest_set": "1" * 64,
        "input_digests": {"x": "5" * 64, "gamma": "6" * 64, "dy": "7" * 64},
        "reference_digests": {"y": "2" * 64, "dx": "3" * 64, "dgamma": "4" * 64},
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
        _validate_raw_result(result, run)


def test_result_schema_accepts_blocked_unsealed_observation(tmp_path: Path) -> None:
    run = _run()
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_valid_result(run)))

    validated = _validate_raw_result(result, run)

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
        _validate_raw_result(result, run)


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

    _validate_raw_result(result, run)


def _execution_identity(runner: Path) -> dict:
    harness = Path(__file__).with_name("target1_te_oracle_harness.py")
    build = Path(__file__).with_name("target1-te-oracle-runner-build-v1.json")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    te_library = runner.parent / "libtransformer_engine.so"
    cuda_library = runner.parent / "libcudart.so"
    cudnn_library = runner.parent / "libcudnn.so"
    te_library.write_bytes(b"te-library")
    cuda_library.write_bytes(b"cuda-library")
    cudnn_library.write_bytes(b"cudnn-library")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[4], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "hardware_class": "h100",
        "marin_revision": revision,
        "harness_sha256": digest(harness),
        "build_manifest_sha256": digest(build),
        "runner_binary_sha256": digest(runner),
        "transformer_engine": {
            "expert_oracle_contract_id": "target1_rowwise_bf16_te_2_17_expert_oracle_v1",
            "version": "2.17.0",
            "source_tag": "v2.17",
            "source_commit": "2e559f062497bef768dfbe9d7e45548fadeca80a",
            "resolved_library_path": str(te_library),
            "library_sha256": digest(te_library),
            "elf_build_id": "te-build-id",
            "resolved_shared_library_dependencies": [
                {"path": str(cuda_library), "sha256": digest(cuda_library), "elf_build_id": "cuda-build-id"}
            ],
        },
        "toolchain": {
            "compiler": "nvcc 12.8",
            "build_flags": ["-O3"],
            "target_architectures": ["sm_90"],
        },
        "cuda": {
            "toolkit_version": "12.8",
            "nvcc_version": "12.8",
            "driver_version": 12080,
            "runtime_version": 12080,
        },
        "cudnn": {
            "compile_time_version": "9.8",
            "runtime_version": "9.8",
            "resolved_library_path": str(cudnn_library),
            "library_sha256": digest(cudnn_library),
            "elf_build_id": "cudnn-build-id",
        },
        "device": {
            "ordinal": 0,
            "model": "H100",
            "uuid": "GPU-test",
            "compute_capability": "9.0",
            "physical_sm_count": 132,
        },
    }


def _sealed_matrix(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    runner = tmp_path / "runner"
    runner.write_bytes(b"reviewed-runner")
    plan_directory = tmp_path / "plan"
    plan = prepare_run_plan(plan_directory, runner)
    identity_path = tmp_path / "execution-identity.json"
    identity_path.write_text(json.dumps(_execution_identity(runner)))
    plan_path = plan_directory / "run-plan.json"
    plan = seal_run_plan(plan_path, runner, identity_path)
    results = tmp_path / "results"
    results.mkdir()
    identity = json.loads(identity_path.read_text())
    for run in plan["runs"]:
        sealed = _valid_result(run)
        sealed["status"] = "sealed_hardware_observation"
        sealed["comparison"]["qualification_status"] = "qualified_against_predeclared_reference"
        sealed["provenance"] = identity
        (results / run["result"]).write_text(json.dumps(sealed))
    return plan_path, identity_path, results, plan["runs"]


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _candidate_matrix(runs: list[dict], marin_revision: str) -> dict:
    records = []
    identities = {
        "ordinary_jax": [
            "marin_revision",
            "jax_version",
            "jaxlib_version",
            "jax_revision",
            "xla_revision",
            "cuda_plugin_identity",
            "pjrt_identity",
            "xla_build_identity",
        ],
        "source_ordered": [
            "marin_revision",
            "pipeline_abi_version",
            "compiler_options_sha256",
            "generated_executable_sha256",
            "invocation_abi_sha256",
            "persistent_cache_key",
        ],
        "fast_identity": [
            "marin_revision",
            "pipeline_abi_version",
            "compiler_options_sha256",
            "generated_executable_sha256",
            "invocation_abi_sha256",
            "persistent_cache_key",
        ],
    }
    subjects = {
        "ordinary_jax": "ordinary_jax_disabled_shuttle",
        "source_ordered": "ordinary_jax_through_shuttle_source_ordered",
        "fast_identity": "ordinary_jax_through_shuttle_fast",
    }
    subject_identities = {}
    for policy, fields in identities.items():
        subject_identity = {}
        for field in fields:
            if field == "pipeline_abi_version":
                subject_identity[field] = 5
            elif field in ("marin_revision", "jax_revision", "xla_revision"):
                subject_identity[field] = marin_revision if field == "marin_revision" else "a" * 40
            elif field.endswith("_sha256"):
                subject_identity[field] = "b" * 64
            else:
                subject_identity[field] = f"pinned-{field}"
        subject_identities[policy] = subject_identity
    subject_artifact_digests = {}
    seen_coordinates = set()
    for run in runs:
        coordinate = (tuple(run["shape"]), run["boundary"])
        if coordinate in seen_coordinates:
            continue
        seen_coordinates.add(coordinate)
        for role in {"forward": ["y"], "backward_recompute": ["dx", "dgamma"], "composed": ["y", "dx", "dgamma"]}[
            run["boundary"]
        ]:
            digest = hashlib.sha256(f"{run['shape']}-{run['boundary']}-{role}".encode()).hexdigest()
            for policy, subject in subjects.items():
                artifact = {
                    "policy": policy,
                    "subject_id": subject,
                    "subject_identity": subject_identities[policy],
                    "hardware_class": "h100",
                    "shape": run["shape"],
                    "boundary": run["boundary"],
                    "output_role": role,
                    "comparison_contract_sha256": COMPARISON_CONTRACT_SHA256,
                    "input_digest_set": run["input_digest_set"],
                    "output_digest": digest,
                }
                artifact_sha256 = _canonical_digest(artifact)
                artifact_key = f"{policy}:{run['shape'][0]}x{run['shape'][1]}:{run['boundary']}:{role}"
                subject_artifact_digests[artifact_key] = artifact_sha256
                identity_lowering_proof = None
                if policy == "fast_identity":
                    identity_lowering_proof = _canonical_digest(
                        {
                            "schema": "target1_identity_lowering_proof_v1",
                            "subject_artifact_sha256": artifact_sha256,
                            "coordinate": {
                                "shape": run["shape"],
                                "boundary": run["boundary"],
                                "output_role": role,
                            },
                        }
                    )
                records.append(
                    {
                        "policy": policy,
                        "subject_id": subject,
                        "hardware_class": "h100",
                        "shape": run["shape"],
                        "boundary": run["boundary"],
                        "output_role": role,
                        "comparison_contract_sha256": COMPARISON_CONTRACT_SHA256,
                        "input_digest_set": run["input_digest_set"],
                        "metrics": {
                            "max_absolute_error": 0.0,
                            "mean_absolute_error": 0.0,
                            "relative_linf_error": 0.0,
                            "max_bfloat16_ulp_error": 0,
                        },
                        "output_digest": digest,
                        "repeatability": {"post_timing_invocations": 3, "output_digests": [digest] * 3},
                        "subject_artifact_sha256": artifact_sha256,
                        "identity_lowering_proof": identity_lowering_proof,
                    }
                )
    return {
        "schema_version": 1,
        "status": "complete_pre_scorecard_matrix",
        "hardware_class": "h100",
        "comparison_contract_sha256": COMPARISON_CONTRACT_SHA256,
        "subject_identities": subject_identities,
        "subject_artifact_digests": subject_artifact_digests,
        "records": records,
        "scorecard_status_changed": False,
    }


def test_complete_matrix_uses_plan_coordinates_and_all_four_te_pairs(tmp_path: Path) -> None:
    plan_path, identity_path, results, runs = _sealed_matrix(tmp_path)
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(_candidate_matrix(runs, json.loads(identity_path.read_text())["marin_revision"]))
    )

    validated = validate_hardware_matrix(plan_path, results, identity_path, candidates_path)

    assert len(validated["records"]) == 36


@pytest.mark.parametrize("mutation", ["missing_te", "extra_te", "missing_candidate", "duplicate_candidate"])
def test_complete_matrix_rejects_incomplete_duplicate_or_extra_coordinates(tmp_path: Path, mutation: str) -> None:
    plan_path, identity_path, results, runs = _sealed_matrix(tmp_path)
    candidates = _candidate_matrix(runs, json.loads(identity_path.read_text())["marin_revision"])
    if mutation == "missing_te":
        (results / "result-23.json").unlink()
    elif mutation == "extra_te":
        (results / "extra.json").write_text("{}")
    elif mutation == "missing_candidate":
        candidates["records"].pop()
    else:
        candidates["records"].append(copy.deepcopy(candidates["records"][0]))
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))

    with pytest.raises(ValueError, match=r"24-run plan|36 records|incomplete|duplicate"):
        validate_hardware_matrix(plan_path, results, identity_path, candidates_path)


def test_complete_matrix_rejects_candidate_that_only_passes_one_backend_pair(tmp_path: Path) -> None:
    plan_path, identity_path, results, runs = _sealed_matrix(tmp_path)
    weak_te = json.loads((results / "result-08.json").read_text())
    weak_te["comparison"]["outputs"][1]["metrics"]["max_bfloat16_ulp_error"] = 2
    (results / "result-08.json").write_text(json.dumps(weak_te))
    candidates = _candidate_matrix(runs, json.loads(identity_path.read_text())["marin_revision"])
    target = next(
        record
        for record in candidates["records"]
        if record["policy"] == "source_ordered"
        and record["shape"] == [2048, 4096]
        and record["boundary"] == "composed"
        and record["output_role"] == "dx"
    )
    target["metrics"]["max_bfloat16_ulp_error"] = 2
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))

    with pytest.raises(AssertionError, match="matched expert-or-dtype-floor"):
        validate_hardware_matrix(plan_path, results, identity_path, candidates_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), True, "schema_version"),
        (("hardware_class",), "a100", "hardware_class"),
        (("runner_binary_sha256",), False, "nonempty string"),
        (("transformer_engine", "library_sha256"), "0" * 63, "SHA-256"),
        (("transformer_engine", "resolved_shared_library_dependencies"), [], "dependencies"),
        (("cuda", "driver_version"), 12080.0, "driver_version"),
        (("device", "physical_sm_count"), True, "physical_sm_count"),
        (("cudnn", "resolved_library_path"), None, "resolved_library_path"),
    ],
)
def test_plan_sealing_rejects_incomplete_or_mistyped_provenance(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    runner = tmp_path / "runner"
    runner.write_bytes(b"reviewed-runner")
    plan_path = tmp_path / "plan/run-plan.json"
    prepare_run_plan(plan_path.parent, runner)
    identity = _execution_identity(runner)
    target = identity
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))

    with pytest.raises(ValueError, match=message):
        seal_run_plan(plan_path, runner, identity_path)


def test_sealed_result_rejects_filename_swap_and_provenance_drift(tmp_path: Path) -> None:
    plan_path, identity_path, results, _runs = _sealed_matrix(tmp_path)
    swapped = results / "result-00.json"
    swapped.write_bytes((results / "result-01.json").read_bytes())

    with pytest.raises(ValueError, match=r"identity|backend|counterbalance"):
        validate_result(swapped, plan_path, identity_path)

    second = tmp_path / "second"
    second.mkdir()
    plan_path, identity_path, results, _runs = _sealed_matrix(second)
    result = json.loads((results / "result-00.json").read_text())
    result["provenance"]["cudnn"]["library_sha256"] = "0" * 64
    (results / "result-00.json").write_text(json.dumps(result))

    with pytest.raises(ValueError, match="provenance"):
        validate_result(results / "result-00.json", plan_path, identity_path)


def test_sealed_plan_rejects_runner_binary_mutation(tmp_path: Path) -> None:
    plan_path, identity_path, results, _runs = _sealed_matrix(tmp_path)
    runner = Path(json.loads(plan_path.read_text())["runner_binary"])
    runner.write_bytes(b"mutated-runner")

    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="runner binary bytes"):
        validate_hardware_matrix(plan_path, results, identity_path, candidates_path)


def test_result_sealing_binds_raw_runtime_to_reviewed_identity(tmp_path: Path) -> None:
    runner = tmp_path / "runner"
    runner.write_bytes(b"reviewed-runner")
    plan_directory = tmp_path / "plan"
    prepare_run_plan(plan_directory, runner)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(_execution_identity(runner)))
    plan_path = plan_directory / "run-plan.json"
    plan = seal_run_plan(plan_path, runner, identity_path)
    raw = tmp_path / "result-00.json"
    raw.write_text(json.dumps(_valid_result(plan["runs"][0])))
    sealed = tmp_path / "sealed/result-00.json"
    sealed.parent.mkdir()

    result = seal_result(raw, sealed, plan_path, identity_path)

    assert result["status"] == "sealed_hardware_observation"
    assert result["provenance"] == json.loads(identity_path.read_text())


@pytest.mark.parametrize("mutation", ["candidate_role", "candidate_identity", "te_backend", "input_join"])
def test_matrix_rejects_role_backend_input_and_subject_provenance_drift(tmp_path: Path, mutation: str) -> None:
    plan_path, identity_path, results, runs = _sealed_matrix(tmp_path)
    candidates = _candidate_matrix(runs, json.loads(identity_path.read_text())["marin_revision"])
    if mutation == "candidate_role":
        candidates["records"][0]["output_role"] = "dx"
    elif mutation == "candidate_identity":
        candidates["subject_identities"]["ordinary_jax"]["jax_version"] = "changed"
    elif mutation == "input_join":
        candidates["records"][0]["input_digest_set"] = "0" * 64
    else:
        result = json.loads((results / "result-00.json").read_text())
        result["backends"]["forward"] = "cudnn"
        (results / "result-00.json").write_text(json.dumps(result))
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))

    with pytest.raises(ValueError, match=r"coordinate|identity|input digest|backend identity"):
        validate_hardware_matrix(plan_path, results, identity_path, candidates_path)


@pytest.mark.parametrize("mutation", ["revision", "artifact", "fast_proof"])
def test_matrix_rejects_unjoined_candidate_subjects(tmp_path: Path, mutation: str) -> None:
    plan_path, identity_path, results, runs = _sealed_matrix(tmp_path)
    candidates = _candidate_matrix(runs, json.loads(identity_path.read_text())["marin_revision"])
    if mutation == "revision":
        candidates["subject_identities"]["source_ordered"]["marin_revision"] = "a" * 40
    elif mutation == "artifact":
        candidates["records"][0]["subject_artifact_sha256"] = "0" * 64
    else:
        fast = next(record for record in candidates["records"] if record["policy"] == "fast_identity")
        fast["identity_lowering_proof"] = "0" * 64
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))

    with pytest.raises(ValueError, match=r"revision|artifact|lowering proof"):
        validate_hardware_matrix(plan_path, results, identity_path, candidates_path)


def test_sealed_plan_rejects_noncanonical_case_directory_even_with_rewritten_plan(tmp_path: Path) -> None:
    plan_path, _identity_path, _results, _runs = _sealed_matrix(tmp_path)
    plan = json.loads(plan_path.read_text())
    case_flag = plan["runs"][0]["argv"].index("--case-directory") + 1
    plan["runs"][0]["argv"][case_flag] = "copied-cases/2048x4096"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="planned argv"):
        validate_result(_results / "result-00.json", plan_path, _identity_path)
