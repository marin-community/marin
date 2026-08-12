# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate non-scorecard Transformer Engine oracle runs."""

import argparse
import copy
import functools
import hashlib
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Mapping
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
    require_matched_oracle,
    require_reference_qualified,
)

RUNNER = Path(__file__).with_name("target1_te_oracle_runner.cpp")
BUILD_MANIFEST = Path(__file__).with_name("target1-te-oracle-runner-build-v1.json")
EXPERT_CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-te-2.17-expert-oracle-v1.json")
COMPARISON_CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-prerun-comparison-v1.json")
COMPARISON_CONTRACT_SHA256 = "af27e9c7d1e4f6fcdf3eacc9d950459d1f627e58d9c9f0d1133b0e3dae6b1504"
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
EXPECTED_RUNNER_SHA256 = "48f2ccabcbee63e04bae99ae2b0d693eb44cc963ff5a4686ac0d7a36f9208047"
EXPECTED_BUILD_MANIFEST_SHA256 = "6d8f092dabf566e4d856a83dac61532566c50a6adb75f5ed7c803159d183423b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


@functools.lru_cache(maxsize=1)
def _comparison_contract() -> Mapping[str, Any]:
    return load_comparison_contract(COMPARISON_CONTRACT)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _planned_run(plan: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    matches = [run for run in plan["runs"] if run["result"] == result_path.name]
    if len(matches) != 1:
        raise ValueError("result filename does not identify exactly one planned run")
    return matches[0]


def seal_result(raw_path: Path, sealed_path: Path, plan_path: Path, identity_path: Path) -> dict[str, Any]:
    """Attach reviewed build identity to one raw result after checking runtime fields."""
    if sealed_path.exists():
        raise ValueError("sealed result output must not exist")
    plan = _validate_plan(plan_path, require_sealed=True)
    identity = validate_execution_identity(_load_json(identity_path))
    if _sha256(identity_path) != plan["execution_identity_sha256"]:
        raise ValueError("execution identity digest does not match sealed plan")
    if identity["hardware_class"] != plan["hardware_class"]:
        raise ValueError("execution identity hardware class does not match sealed plan")
    expected_run = _planned_run(plan, raw_path)
    raw = _validate_raw_result(raw_path, expected_run)
    if raw["provenance"]["cuda"]["driver_version"] != identity["cuda"]["driver_version"]:
        raise ValueError("runtime CUDA driver does not match sealed identity")
    if raw["provenance"]["cuda"]["runtime_version"] != identity["cuda"]["runtime_version"]:
        raise ValueError("runtime CUDA version does not match sealed identity")
    if raw["provenance"]["device"]["ordinal"] != identity["device"]["ordinal"]:
        raise ValueError("runtime device ordinal does not match sealed identity")
    if raw["provenance"]["device"]["physical_sm_count"] != identity["device"]["physical_sm_count"]:
        raise ValueError("runtime physical SM count does not match sealed identity")
    sealed = copy.deepcopy(raw)
    sealed["status"] = "sealed_hardware_observation"
    sealed["comparison"]["qualification_status"] = "qualified_against_predeclared_reference"
    sealed["provenance"] = identity
    sealed_path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return validate_result(sealed_path, plan_path, identity_path)


def validate_result(path: Path, plan_path: Path, identity_path: Path) -> dict[str, Any]:
    """Validate one sealed observation, deriving its coordinate from the pinned plan."""
    plan = _validate_plan(plan_path, require_sealed=True)
    identity = validate_execution_identity(_load_json(identity_path))
    if _sha256(identity_path) != plan["execution_identity_sha256"]:
        raise ValueError("execution identity digest does not match sealed plan")
    return _validate_sealed_result(path, plan, identity)


def _validate_sealed_result(path: Path, plan: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    expected_run = _planned_run(plan, path)
    sealed = _load_json(path)
    if sealed.get("status") != "sealed_hardware_observation":
        raise ValueError("result is not sealed")
    if sealed.get("provenance") != identity:
        raise ValueError("result provenance does not match sealed execution identity")
    if sealed.get("comparison", {}).get("qualification_status") != "qualified_against_predeclared_reference":
        raise ValueError("result qualification status drifted")
    raw = copy.deepcopy(sealed)
    raw["status"] = "unsealed_hardware_observation"
    raw["comparison"]["qualification_status"] = "unsealed_runner_metrics_require_contract_validator"
    raw["provenance"] = {
        "marin_revision": None,
        "adapter_sha256": None,
        "transformer_engine": {
            "version": "2.17.0",
            "source_tag": "v2.17",
            "source_commit": SOURCE_COMMIT,
            "resolved_library_path": None,
            "library_sha256": None,
            "elf_build_id": None,
            "resolved_shared_library_dependencies": None,
        },
        "toolchain": {"compiler": None, "build_flags": None, "target_architectures": None},
        "cuda": {
            "toolkit_version": None,
            "nvcc_version": None,
            "driver_version": identity["cuda"]["driver_version"],
            "runtime_version": identity["cuda"]["runtime_version"],
        },
        "device": {
            "ordinal": identity["device"]["ordinal"],
            "model": None,
            "uuid": None,
            "compute_capability": None,
            "physical_sm_count": identity["device"]["physical_sm_count"],
        },
    }
    _validate_raw_result(path, expected_run, document=raw)
    return sealed


CANDIDATE_SUBJECTS = {
    "ordinary_jax": "ordinary_jax_disabled_shuttle",
    "source_ordered": "ordinary_jax_through_shuttle_source_ordered",
    "fast_identity": "ordinary_jax_through_shuttle_fast",
}


def _validate_candidate_identity(
    policy: str, value: object, marin_revision: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    identity_key = {
        "ordinary_jax": "ordinary_jax",
        "source_ordered": "shuttle_source_ordered",
        "fast_identity": "shuttle_fast",
    }[policy]
    required = set(contract["subjects"][identity_key]["required_identity"])
    required.discard("identity_lowering_proof")
    identity = _closed(value, required, f"{policy} subject identity")
    for field, item in identity.items():
        name = f"{policy} subject identity.{field}"
        if field == "pipeline_abi_version":
            _exact_int(item, 5, name)
        elif field in ("marin_revision", "jax_revision", "xla_revision"):
            _revision(item, name)
        elif field.endswith("_sha256"):
            _sha(item, name)
        else:
            _string(item, name)
    if identity["marin_revision"] != marin_revision:
        raise ValueError(f"{policy} subject Marin revision does not match sealed execution identity")
    return identity


def validate_hardware_matrix(
    plan_path: Path, results_directory: Path, identity_path: Path, candidates_path: Path
) -> dict[str, Any]:
    """Validate one complete 24-run TE matrix and all matched candidate outputs."""
    plan = _validate_plan(plan_path, require_sealed=True)
    identity = validate_execution_identity(_load_json(identity_path))
    if _sha256(identity_path) != plan["execution_identity_sha256"]:
        raise ValueError("execution identity digest does not match sealed plan")
    expected_names = {run["result"] for run in plan["runs"]}
    observed_names = {path.name for path in results_directory.glob("*.json")}
    if observed_names != expected_names:
        raise ValueError("TE result path set must equal the complete 24-run plan")
    te_metrics: dict[tuple[tuple[int, int], str, str, str, str], Mapping[str, object]] = {}
    for run in plan["runs"]:
        result = _validate_sealed_result(results_directory / run["result"], plan, identity)
        for output in result["comparison"]["outputs"]:
            key = (
                tuple(run["shape"]),
                run["boundary"],
                output["role"],
                run["forward_backend"],
                run["backward_backend"],
            )
            if key in te_metrics:
                raise ValueError("duplicate TE output coordinate")
            te_metrics[key] = output["metrics"]
    if len(te_metrics) != 48:
        raise ValueError("TE result matrix must contain exactly 48 output records")

    contract = _comparison_contract()
    candidates = _closed(
        _load_json(candidates_path),
        {
            "schema_version",
            "status",
            "hardware_class",
            "comparison_contract_sha256",
            "subject_identities",
            "subject_artifact_digests",
            "records",
            "scorecard_status_changed",
        },
        "candidate matrix",
    )
    _exact_int(candidates["schema_version"], 1, "candidate matrix.schema_version")
    if candidates["status"] != "complete_pre_scorecard_matrix" or candidates["scorecard_status_changed"] is not False:
        raise ValueError("candidate matrix status drifted")
    if (
        candidates["hardware_class"] != plan["hardware_class"]
        or candidates["comparison_contract_sha256"] != COMPARISON_CONTRACT_SHA256
    ):
        raise ValueError("candidate matrix join identity drifted")
    expected_keys = {
        (policy, tuple(run["shape"]), run["boundary"], role)
        for policy in CANDIDATE_SUBJECTS
        for run in plan["runs"]
        for role in OUTPUT_ROLES[run["boundary"]]
    }
    identities_value = candidates["subject_identities"]
    if not isinstance(identities_value, dict) or set(identities_value) != set(CANDIDATE_SUBJECTS):
        raise ValueError("candidate subject identities drifted")
    subject_identities = {
        policy: _validate_candidate_identity(policy, identities_value[policy], identity["marin_revision"], contract)
        for policy in CANDIDATE_SUBJECTS
    }
    artifact_digests = candidates["subject_artifact_digests"]
    if not isinstance(artifact_digests, dict) or set(artifact_digests) != {
        f"{policy}:{shape[0]}x{shape[1]}:{boundary}:{role}" for policy, shape, boundary, role in expected_keys
    }:
        raise ValueError("candidate subject artifact coordinate set drifted")
    for value in artifact_digests.values():
        _sha(value, "candidate subject artifact digest")
    records = candidates["records"]
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError("candidate matrix must contain exactly 36 records")
    observed: dict[tuple[str, tuple[int, int], str, str], dict[str, Any]] = {}
    runs_by_coordinate = {(tuple(run["shape"]), run["boundary"]): run for run in plan["runs"]}
    for value in records:
        record = _closed(
            value,
            {
                "policy",
                "subject_id",
                "hardware_class",
                "shape",
                "boundary",
                "output_role",
                "comparison_contract_sha256",
                "input_digest_set",
                "metrics",
                "output_digest",
                "repeatability",
                "subject_artifact_sha256",
                "identity_lowering_proof",
            },
            "candidate record",
        )
        if record["policy"] not in CANDIDATE_SUBJECTS or record["subject_id"] != CANDIDATE_SUBJECTS[record["policy"]]:
            raise ValueError("candidate subject or policy drifted")
        if record["hardware_class"] != plan["hardware_class"]:
            raise ValueError("candidate hardware class drifted")
        if (
            not isinstance(record["shape"], list)
            or len(record["shape"]) != 2
            or any(type(extent) is not int or extent <= 0 for extent in record["shape"])
        ):
            raise ValueError("candidate shape drifted")
        key = (record["policy"], tuple(record["shape"]), record["boundary"], record["output_role"])
        if key not in expected_keys or key in observed:
            raise ValueError("candidate coordinate is duplicate, extra, or invalid")
        if record["comparison_contract_sha256"] != COMPARISON_CONTRACT_SHA256:
            raise ValueError("candidate comparison contract drifted")
        run = runs_by_coordinate[(tuple(record["shape"]), record["boundary"])]
        if record["input_digest_set"] != run["input_digest_set"]:
            raise ValueError("candidate input digest set drifted")
        _sha(record["output_digest"], "candidate output digest")
        repeatability = _closed(
            record["repeatability"], {"post_timing_invocations", "output_digests"}, "candidate repeatability"
        )
        _exact_int(repeatability["post_timing_invocations"], 3, "candidate repeatability count")
        if repeatability["output_digests"] != [record["output_digest"]] * 3:
            raise ValueError("candidate repeatability digests drifted")
        artifact = {
            "policy": record["policy"],
            "subject_id": record["subject_id"],
            "subject_identity": subject_identities[record["policy"]],
            "hardware_class": record["hardware_class"],
            "shape": record["shape"],
            "boundary": record["boundary"],
            "output_role": record["output_role"],
            "comparison_contract_sha256": record["comparison_contract_sha256"],
            "input_digest_set": record["input_digest_set"],
            "output_digest": record["output_digest"],
        }
        expected_artifact_sha = _canonical_sha256(artifact)
        artifact_key = (
            f"{record['policy']}:{record['shape'][0]}x{record['shape'][1]}:"
            f"{record['boundary']}:{record['output_role']}"
        )
        if (
            record["subject_artifact_sha256"] != expected_artifact_sha
            or artifact_digests[artifact_key] != expected_artifact_sha
        ):
            raise ValueError("candidate subject artifact digest does not bind identity and coordinate")
        expected_proof = None
        if record["policy"] == "fast_identity":
            expected_proof = _canonical_sha256(
                {
                    "schema": "target1_identity_lowering_proof_v1",
                    "subject_artifact_sha256": expected_artifact_sha,
                    "coordinate": {
                        "shape": record["shape"],
                        "boundary": record["boundary"],
                        "output_role": record["output_role"],
                    },
                }
            )
        if record["identity_lowering_proof"] != expected_proof:
            raise ValueError("candidate identity lowering proof does not bind artifact and coordinate")
        require_reference_qualified(
            record["metrics"],
            shape=f"{record['shape'][0]}x{record['shape'][1]}",
            role=record["output_role"],
            contract=contract,
        )
        observed[key] = record
    if set(observed) != expected_keys:
        raise ValueError("candidate matrix is incomplete")

    for (_policy, shape, boundary, role), ordinary in [
        (key, record) for key, record in observed.items() if key[0] == "ordinary_jax"
    ]:
        for policy in ("source_ordered", "fast_identity"):
            candidate = observed[(policy, shape, boundary, role)]
            if candidate["output_digest"] != ordinary["output_digest"]:
                raise ValueError(f"{policy} output is not bitwise equal to ordinary JAX")
            for forward_backend, backward_backend in BACKEND_PAIRS:
                require_matched_oracle(
                    candidate["metrics"],
                    te_metrics[(shape, boundary, role, forward_backend, backward_backend)],
                    shape=f"{shape[0]}x{shape[1]}",
                    role=role,
                    contract=contract,
                )
    return candidates


def _load_json(path: Path) -> Any:
    payload = path.read_bytes()
    if len(payload) > 1024 * 1024:
        raise ValueError("JSON evidence exceeds the 1 MiB limit")
    return json.loads(
        payload,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )


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


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    text = _string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _revision(value: object, name: str) -> str:
    text = _string(value, name)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{name} must be a lowercase Git revision")
    return text


def validate_execution_identity(value: object) -> dict[str, Any]:
    """Validate the closed build, library, CUDA, cuDNN, and device identity."""
    identity = _closed(
        value,
        {
            "schema_version",
            "hardware_class",
            "marin_revision",
            "harness_sha256",
            "build_manifest_sha256",
            "runner_binary_sha256",
            "transformer_engine",
            "toolchain",
            "cuda",
            "cudnn",
            "device",
        },
        "execution identity",
    )
    _exact_int(identity["schema_version"], 1, "execution identity.schema_version")
    if identity["hardware_class"] not in ("h100", "gb200_or_b200"):
        raise ValueError("execution identity.hardware_class drifted")
    _revision(identity["marin_revision"], "execution identity.marin_revision")
    _sha(identity["harness_sha256"], "execution identity.harness_sha256")
    _sha(identity["build_manifest_sha256"], "execution identity.build_manifest_sha256")
    _sha(identity["runner_binary_sha256"], "execution identity.runner_binary_sha256")
    provider = _closed(
        identity["transformer_engine"],
        {
            "expert_oracle_contract_id",
            "version",
            "source_tag",
            "source_commit",
            "resolved_library_path",
            "library_sha256",
            "elf_build_id",
            "resolved_shared_library_dependencies",
        },
        "execution identity.transformer_engine",
    )
    if (provider["version"], provider["source_tag"], provider["source_commit"]) != (
        "2.17.0",
        "v2.17",
        SOURCE_COMMIT,
    ):
        raise ValueError("execution identity Transformer Engine source drifted")
    if provider["expert_oracle_contract_id"] != "target1_rowwise_bf16_te_2_17_expert_oracle_v1":
        raise ValueError("execution identity expert oracle contract drifted")
    _string(provider["resolved_library_path"], "execution identity Transformer Engine library path")
    _sha(provider["library_sha256"], "execution identity Transformer Engine library SHA")
    _string(provider["elf_build_id"], "execution identity Transformer Engine ELF build ID")
    dependencies = provider["resolved_shared_library_dependencies"]
    if not isinstance(dependencies, list) or not dependencies or len(dependencies) > 256:
        raise ValueError("execution identity Transformer Engine dependencies drifted")
    dependency_paths = [dependency.get("path") for dependency in dependencies if isinstance(dependency, dict)]
    if len(dependency_paths) != len(dependencies) or len(set(dependency_paths)) != len(dependency_paths):
        raise ValueError("execution identity Transformer Engine dependency paths must be unique")
    for dependency in dependencies:
        record = _closed(dependency, {"path", "sha256", "elf_build_id"}, "resolved dependency")
        _string(record["path"], "resolved dependency.path")
        _sha(record["sha256"], "resolved dependency.sha256")
        _string(record["elf_build_id"], "resolved dependency.elf_build_id")
    toolchain = _closed(
        identity["toolchain"], {"compiler", "build_flags", "target_architectures"}, "execution identity.toolchain"
    )
    _string(toolchain["compiler"], "execution identity.toolchain.compiler")
    for field in ("build_flags", "target_architectures"):
        values = toolchain[field]
        if not isinstance(values, list) or not values or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"execution identity.toolchain.{field} drifted")
    cuda = _closed(
        identity["cuda"],
        {"toolkit_version", "nvcc_version", "driver_version", "runtime_version"},
        "execution identity.cuda",
    )
    _string(cuda["toolkit_version"], "execution identity.cuda.toolkit_version")
    _string(cuda["nvcc_version"], "execution identity.cuda.nvcc_version")
    for field in ("driver_version", "runtime_version"):
        if type(cuda[field]) is not int or cuda[field] <= 0:
            raise ValueError(f"execution identity.cuda.{field} drifted")
    cudnn = _closed(
        identity["cudnn"],
        {"compile_time_version", "runtime_version", "resolved_library_path", "library_sha256", "elf_build_id"},
        "execution identity.cudnn",
    )
    for field in ("compile_time_version", "runtime_version", "resolved_library_path", "elf_build_id"):
        _string(cudnn[field], f"execution identity.cudnn.{field}")
    _sha(cudnn["library_sha256"], "execution identity.cudnn.library_sha256")
    device = _closed(
        identity["device"],
        {"ordinal", "model", "uuid", "compute_capability", "physical_sm_count"},
        "execution identity.device",
    )
    if type(device["ordinal"]) is not int or device["ordinal"] < 0:
        raise ValueError("execution identity.device.ordinal drifted")
    if type(device["physical_sm_count"]) is not int or device["physical_sm_count"] <= 0:
        raise ValueError("execution identity.device.physical_sm_count drifted")
    for field in ("model", "uuid", "compute_capability"):
        _string(device[field], f"execution identity.device.{field}")
    return identity


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
            "scope": "c++20_syntax_exact_public_headers_and_executable_sha256_input_probe_no_cuda_link",
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
        },
    }
    if document != expected:
        raise ValueError("runner build manifest drifted")
    return document


def _write_bfloat16(path: Path, value: Any) -> None:
    path.write_bytes(value.view("uint16").tobytes(order="C"))


def _expected_run_argv(runner_binary: str, run: Mapping[str, Any]) -> list[str]:
    rows, features = run["shape"]
    return [
        runner_binary,
        "--boundary",
        run["boundary"],
        "--forward-backend",
        run["forward_backend"],
        "--backward-backend",
        run["backward_backend"],
        "--rows",
        str(rows),
        "--features",
        str(features),
        "--case-directory",
        f"cases/{rows}x{features}",
        "--output",
        run["result"],
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


def prepare_run_plan(output: Path, runner_binary: Path) -> dict[str, Any]:
    """Write pinned BF16 inputs/references and a deterministic 24-run plan."""
    validate_build_manifest()
    load_contract(EXPERT_CONTRACT)
    _comparison_contract()
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
            case = output / f"cases/{rows}x{features}"
            input_names = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
            input_digests = {name: _sha256(case / f"{name}.bf16") for name in input_names}
            reference_digests = {role: _sha256(case / f"reference_{role}.bf16") for role in OUTPUT_ROLES[boundary]}
            input_digest_set = _canonical_sha256(input_digests)
            run = {
                "position": position,
                "shape": [rows, features],
                "boundary": boundary,
                "forward_backend": forward_backend,
                "backward_backend": backward_backend,
                "input_digests": input_digests,
                "input_digest_set": input_digest_set,
                "reference_digests": reference_digests,
                "result": result,
            }
            run["argv"] = _expected_run_argv(str(runner_binary), run)
            runs.append(run)
            position += 1
    plan = {
        "schema_version": 1,
        "status": "prepared_not_executed",
        "hardware_class": None,
        "counterbalance_plan_id": PLAN_ID,
        "runner_binary": str(runner_binary),
        "runner_binary_sha256": None,
        "execution_identity_sha256": None,
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


def _validate_plan(path: Path, *, require_sealed: bool) -> dict[str, Any]:
    plan = _closed(
        _load_json(path),
        {
            "schema_version",
            "status",
            "hardware_class",
            "counterbalance_plan_id",
            "runner_binary",
            "runner_binary_sha256",
            "execution_identity_sha256",
            "comparison_contract",
            "runs",
            "scorecard_status_changed",
        },
        "run plan",
    )
    _exact_int(plan["schema_version"], 1, "run plan.schema_version")
    if plan["counterbalance_plan_id"] != PLAN_ID or plan["scorecard_status_changed"] is not False:
        raise ValueError("run plan policy drifted")
    if plan["comparison_contract"] != {
        "id": COMPARISON_CONTRACT_ID,
        "path": COMPARISON_CONTRACT.name,
        "sha256": COMPARISON_CONTRACT_SHA256,
    }:
        raise ValueError("run plan comparison contract drifted")
    runs = plan["runs"]
    if not isinstance(runs, list) or len(runs) != 24:
        raise ValueError("run plan must contain exactly 24 runs")
    expected_coordinates = []
    position = 0
    for group, ((rows, features), boundary) in enumerate(
        shape_boundary for shape in SHAPES for shape_boundary in ((shape, item) for item in BOUNDARIES)
    ):
        ordered_pairs = BACKEND_PAIRS if group % 2 == 0 else tuple(reversed(BACKEND_PAIRS))
        for forward_backend, backward_backend in ordered_pairs:
            expected_coordinates.append((position, [rows, features], boundary, forward_backend, backward_backend))
            position += 1
    file_digests: dict[Path, str] = {}

    def file_digest(file: Path) -> str:
        if file not in file_digests:
            file_digests[file] = _sha256(file)
        return file_digests[file]

    for run, coordinate in zip(runs, expected_coordinates, strict=True):
        record = _closed(
            run,
            {
                "position",
                "shape",
                "boundary",
                "forward_backend",
                "backward_backend",
                "input_digests",
                "input_digest_set",
                "reference_digests",
                "argv",
                "result",
            },
            "planned run",
        )
        if (
            record["position"],
            record["shape"],
            record["boundary"],
            record["forward_backend"],
            record["backward_backend"],
        ) != coordinate:
            raise ValueError("planned run coordinate drifted")
        if record["result"] != f"result-{record['position']:02d}.json":
            raise ValueError("planned result filename drifted")
        _sha(record["input_digest_set"], "planned input digest set")
        if _canonical_sha256(record["input_digests"]) != record["input_digest_set"]:
            raise ValueError("planned input digest set does not match inputs")
        expected_inputs = {"x", "gamma"} if record["boundary"] == "forward" else {"x", "gamma", "dy"}
        if not isinstance(record["input_digests"], dict) or set(record["input_digests"]) != expected_inputs:
            raise ValueError("planned input roles drifted")
        for digest in record["input_digests"].values():
            _sha(digest, "planned input digest")
        if not isinstance(record["reference_digests"], dict) or set(record["reference_digests"]) != set(
            OUTPUT_ROLES[record["boundary"]]
        ):
            raise ValueError("planned reference roles drifted")
        for digest in record["reference_digests"].values():
            _sha(digest, "planned reference digest")
        case = path.parent / f"cases/{record['shape'][0]}x{record['shape'][1]}"
        for role, digest in record["input_digests"].items():
            if file_digest(case / f"{role}.bf16") != digest:
                raise ValueError("planned input bytes drifted")
        for role, digest in record["reference_digests"].items():
            if file_digest(case / f"reference_{role}.bf16") != digest:
                raise ValueError("planned reference bytes drifted")
        if record["argv"] != _expected_run_argv(plan["runner_binary"], record):
            raise ValueError("planned argv drifted")
    if require_sealed:
        if plan["status"] != "sealed_for_execution" or plan["hardware_class"] not in ("h100", "gb200_or_b200"):
            raise ValueError("run plan is not sealed for a hardware class")
        _sha(plan["runner_binary_sha256"], "run plan runner binary SHA")
        _sha(plan["execution_identity_sha256"], "run plan execution identity SHA")
        runner_binary = Path(plan["runner_binary"])
        if not runner_binary.is_file() or _sha256(runner_binary) != plan["runner_binary_sha256"]:
            raise ValueError("run plan runner binary bytes drifted")
    elif (
        plan["status"] != "prepared_not_executed"
        or plan["hardware_class"] is not None
        or plan["runner_binary_sha256"] is not None
        or plan["execution_identity_sha256"] is not None
    ):
        raise ValueError("prepared run plan state drifted")
    return plan


def seal_run_plan(plan_path: Path, runner_binary: Path, identity_path: Path) -> dict[str, Any]:
    """Bind a prepared plan to one exact binary and reviewed execution identity."""
    plan = _validate_plan(plan_path, require_sealed=False)
    identity = validate_execution_identity(_load_json(identity_path))
    runner_sha = _sha256(runner_binary)
    if identity["runner_binary_sha256"] != runner_sha:
        raise ValueError("execution identity runner binary SHA does not match bytes")
    if identity["harness_sha256"] != _sha256(Path(__file__)):
        raise ValueError("execution identity harness SHA does not match bytes")
    if identity["build_manifest_sha256"] != _sha256(BUILD_MANIFEST):
        raise ValueError("execution identity build manifest SHA does not match bytes")
    repository_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[4], check=True, capture_output=True, text=True
    ).stdout.strip()
    if identity["marin_revision"] != repository_revision:
        raise ValueError("execution identity Marin revision does not match checkout")
    provider = identity["transformer_engine"]
    if _sha256(Path(provider["resolved_library_path"])) != provider["library_sha256"]:
        raise ValueError("Transformer Engine library SHA does not match resolved bytes")
    for dependency in provider["resolved_shared_library_dependencies"]:
        if _sha256(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError("Transformer Engine dependency SHA does not match resolved bytes")
    cudnn = identity["cudnn"]
    if _sha256(Path(cudnn["resolved_library_path"])) != cudnn["library_sha256"]:
        raise ValueError("cuDNN library SHA does not match resolved bytes")
    model = identity["device"]["model"].lower()
    if (identity["hardware_class"] == "h100" and "h100" not in model) or (
        identity["hardware_class"] == "gb200_or_b200" and not any(name in model for name in ("gb200", "b200"))
    ):
        raise ValueError("execution identity device model does not match hardware class")
    plan["status"] = "sealed_for_execution"
    plan["hardware_class"] = identity["hardware_class"]
    plan["runner_binary"] = str(runner_binary)
    plan["runner_binary_sha256"] = runner_sha
    plan["execution_identity_sha256"] = _sha256(identity_path)
    for run in plan["runs"]:
        run["argv"] = _expected_run_argv(str(runner_binary), run)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return _validate_plan(plan_path, require_sealed=True)


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


def _validate_raw_result(
    path: Path, expected_run: dict[str, Any], *, document: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one raw runner observation before provenance sealing."""
    comparison_contract = _comparison_contract()
    result = _closed(
        _load_json(path) if document is None else document,
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
            "input_digests",
            "input_digest_set",
            "reference_digests",
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
    if comparison["input_digest_set"] != expected_run["input_digest_set"]:
        raise ValueError("comparison input digest set drifted")
    if comparison["input_digests"] != expected_run["input_digests"]:
        raise ValueError("comparison input digests drifted")
    if comparison["reference_digests"] != expected_run["reference_digests"]:
        raise ValueError("comparison reference digests drifted")
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
        probe = stub / "target1-sha256-probe"
        subprocess.run(
            [
                resolved_compiler,
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DTARGET1_SHA256_PROBE",
                f"-I{stub}",
                "-isystem",
                str(te_source / "transformer_engine/common/include"),
                str(RUNNER),
                "-o",
                str(probe),
            ],
            check=True,
        )
        expected = _sha256(RUNNER)
        observed = subprocess.run([probe, RUNNER, expected], check=True, capture_output=True, text=True).stdout.strip()
        if observed != _sha256(RUNNER):
            raise ValueError("runner executable SHA-256 input probe disagrees with Python SHA-256")
        mismatch = subprocess.run([probe, RUNNER, "0" * 64], capture_output=True, text=True)
        if mismatch.returncode == 0:
            raise ValueError("runner executable SHA-256 input probe accepted mismatched bytes")


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
                    "status": "passed_syntax_and_cpu_sha256_probe_no_cuda_link",
                    "host": f"{platform.system().lower()}_{platform.machine().lower()}",
                    "te_commit": SOURCE_COMMIT,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
