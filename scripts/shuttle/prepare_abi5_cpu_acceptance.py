# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a deterministic, config-free ABI 5 CPU acceptance capsule locally."""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SOURCE = Path(__file__).with_name("abi5_cpu_acceptance_manifest.json")
DEPENDENCY_INPUT_SOURCE = Path(__file__).with_name("abi5_cpu_linux_dependency_inputs.json")
RUNNER_SOURCE = Path(__file__).with_name("run_abi5_cpu_acceptance_preflight.sh")
RECEIPT_VERIFIER_SOURCE = Path(__file__).with_name("verify_abi5_cpu_post_submit_receipt.py")
EXPECTED_BASE_COMMIT = "0ac70a0a21bd7935980827bbf39d95e378335f99"
EXPECTED_JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
EXPECTED_XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
SEALED_ARTIFACT = "lib/shuttle/mlir/artifacts/native-preflight-20260810-jaxacceptance6"
FORBIDDEN_COMPONENTS = frozenset(
    {
        ".git",
        ".marin.yaml",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "coreweave.yaml",
    }
)
REQUIRED_EXTERNAL_IDENTITIES = frozenset(
    {
        "bundle_init_pinning_implementation_review",
        "bundle_content_sha256",
        "exact_bundle_blob_submission_review",
        "init_image_oci_ref",
        "iris_config_sha256",
        "iris_revision",
        "linux_dependency_lock_sha256",
        "linux_python_identity",
        "minimal_execution_environment_policy_review",
        "runner_implementation_review",
        "task_image_oci_ref",
    }
)
EXPECTED_CAPSULE_PATH_COUNT = 143
EXPECTED_CAPSULE_PATH_SET_SHA256 = "cea5fce42a01abb1c691a38591a7cc42ac1d6fdbf0913aad80c4f64ba8a10bc3"
EXPECTED_DEPENDENCY_INPUT_SHA256 = "29674654100e474eb7e5d8ff1ffca4e5fe5a9a26cd07dce3bfa5d2bc7a671a73"
EXPECTED_MANIFEST_FIELDS = frozenset(
    {
        "capsule_allowlist",
        "destination",
        "execution_identity",
        "launch_ready",
        "patch_sha256",
        "pipeline_abi_version",
        "preparation_base_commit",
        "resource_request",
        "retry_limits",
        "schema_version",
        "scorecard_status_changed",
        "sealed_artifact_prohibition",
        "target1_contract",
        "toolchain",
        "unresolved_external_identities",
    }
)
EXPECTED_TOOLCHAIN = {
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "jax_revision": EXPECTED_JAX_REVISION,
    "xla_revision": EXPECTED_XLA_REVISION,
    "stablehlo_revision": "806a6844dfd92cca1ce5391c86dca0ef9e952550",
    "llvm_revision": "9a4faee1068c09efbf837cfb7b0f5693b24635f4",
    "nanobind_revision": "30f12ae6650ecec86042053d522d9af585f269b0",
}
EXPECTED_EXECUTION_IDENTITY = {
    "schema_version": 2,
    "platform": {"architecture": "x86_64", "operating_system": "linux", "python_abi": "cp312"},
    "python": {
        "implementation": "CPython",
        "version": "3.12.11",
        "build_identity": None,
        "executable_sha256": None,
    },
    "bazel": {
        "version": "7.7.0",
        "binary_sha256": "953f1235a590546a4a9a83d757c075ecf7c7d8dbc30221fd086959a20d8c7a69",
        "verification": "sha256_before_first_execution",
    },
    "dependency_inputs": {
        "path": "linux-dependency-inputs.json",
        "sha256": EXPECTED_DEPENDENCY_INPUT_SHA256,
        "lock_ready": False,
        "unresolved": ["uv-build"],
    },
    "images": {"task_ref": None, "init_ref": None},
    "environment": {
        "status": "closed_empty_submitted_environment",
        "allowed_names": [],
        "inherit_host_environment": False,
        "runtime_receipt_required_names": [
            "IRIS_BUNDLE_ID",
            "IRIS_BUNDLE_INIT_IMAGE",
            "IRIS_NUM_TASKS",
            "IRIS_TASK_ID",
        ],
        "runtime_value_rules": {
            "IRIS_BUNDLE_ID": "equals_reviewed_bundle_sha256",
            "IRIS_BUNDLE_INIT_IMAGE": "equals_reviewed_init_image_oci_ref",
            "IRIS_NUM_TASKS": "decimal_integer_equals_1",
            "IRIS_TASK_ID": "iris_task_attempt_wire",
        },
        "forbidden_files": [".marin.yaml", "coreweave.yaml"],
    },
    "iris": {
        "minimum_contract_commit": "e0689926329548e0b0c987b1e197c67c189c4523",
        "controller_revision": None,
        "config_sha256": None,
        "checked_in_config_path": "lib/iris/config/cw-us-east-02a.yaml",
        "checked_in_config_sha256": "7c90860fd8a45f03aa8d7ff3fde0200edc19ab9b0ae6cbaa11ae74b723515507",
        "required_capabilities": ["exact_workspace_bundle_bytes", "per_job_bundle_init_image"],
    },
    "post_submit_bundle_proof": {
        "schema_version": 2,
        "status": "required_after_submission",
        "fields": {
            "bundle_manifest_sha256": "lowerhex_sha256",
            "expected_extraction_manifest_sha256": "lowerhex_sha256",
            "launch_response_job_id": "iris_job_wire",
            "persisted_bundle_id": "lowerhex_sha256",
            "persisted_bundle_init_image": "immutable_oci_ref",
            "reviewed_bundle_sha256": "lowerhex_sha256",
            "status_response_job_id": "iris_job_wire",
            "task_iris_bundle_id": "lowerhex_sha256",
            "task_iris_bundle_init_image": "immutable_oci_ref",
            "task_iris_num_tasks": "decimal_integer_equals_1",
            "task_iris_task_id": "iris_task_attempt_wire",
        },
        "identity_rule": (
            "launch and status job IDs match; persisted and task bundle IDs match reviewed bytes; "
            "persisted and task init images match the resolved immutable image"
        ),
    },
}
EXPECTED_PATCHES = {
    "lib/shuttle/mlir/jax_patch/0001-link-shuttle-xla-registry-adapter.patch": (
        "1e8b1400ee05bf8c8037277046113e1f1b1fae5ef56077899b7e853855e3424e"
    ),
    "lib/shuttle/mlir/jax_patch/0002-add-acceptance-observer-bridge.patch": (
        "c4d5bc4aaa4b72ee7e2ecb44d28719be99da8defcf01ebd107308df41cb71942"
    ),
    "lib/shuttle/mlir/xla_patch/0001-add-stablehlo-module-transform-hook.patch": (
        "b4e9f7cf2a49c42957cf24f16b43a1e0ead0a7e25d664629a841a5dbb0c7dbf9"
    ),
    "lib/shuttle/mlir/xla_patch/0002-anchor-lit-labels-to-xla-repository.patch": (
        "e4f3121f3123d7e2ee781cc5ee92f1ddeb0df662b6702d37615bcae821ecbc99"
    ),
}
EXPECTED_UV_BUILD_RESOLUTION = {
    "requirement": "uv-build>=0.7.19,<0.10.0",
    "repository_lock_package": None,
    "checked_in_wheel_path": None,
    "checked_in_wheel_sha256": None,
    "completion_steps": [
        (
            "Resolve one Linux x86_64 CPython 3.12 uv-build wheel satisfying the declared requirement in an "
            "explicitly authorized networked lock update."
        ),
        "Record the exact version, files.pythonhosted.org wheel URL, and SHA-256 in uv.lock and this contract.",
        (
            "Verify the downloaded wheel against both recorded hashes before any build and install with "
            "--no-build-isolation."
        ),
        "Independently review the completed lock before setting lock_ready true.",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_strict_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_strict_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path.name}") from error


def _exact_int(value: object, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must be the integer {expected}")


def _exact_bool(value: object, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} must be the boolean {str(expected).lower()}")


def _exact_string(value: object, expected: str, name: str) -> None:
    if type(value) is not str or value != expected:
        raise ValueError(f"{name} must be the declared string")


def _exact_none(value: object, name: str) -> None:
    if value is not None:
        raise ValueError(f"{name} must remain null")


def _unresolved_string(value: object, pattern: str, name: str) -> None:
    if value is None:
        return
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        raise ValueError(f"{name} must be null or match its declared string pattern")
    raise ValueError(f"{name} must remain null in the non-launch-ready manifest")


def _closed_mapping(value: object, expected_fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{name} fields changed")
    return value


def _validate_execution_identity(value: object) -> dict[str, Any]:
    identity = _closed_mapping(value, set(EXPECTED_EXECUTION_IDENTITY), "execution_identity")
    _exact_int(identity["schema_version"], 2, "execution_identity.schema_version")

    python = _closed_mapping(identity["python"], set(EXPECTED_EXECUTION_IDENTITY["python"]), "execution_identity.python")
    _unresolved_string(python["build_identity"], r"\S(?:.*\S)?", "execution_identity.python.build_identity")
    _unresolved_string(python["executable_sha256"], r"[0-9a-f]{64}", "execution_identity.python.executable_sha256")

    dependency_inputs = _closed_mapping(
        identity["dependency_inputs"],
        set(EXPECTED_EXECUTION_IDENTITY["dependency_inputs"]),
        "execution_identity.dependency_inputs",
    )
    _exact_bool(dependency_inputs["lock_ready"], False, "execution_identity.dependency_inputs.lock_ready")

    images = _closed_mapping(identity["images"], {"task_ref", "init_ref"}, "execution_identity.images")
    immutable_image = r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
    _unresolved_string(images["task_ref"], immutable_image, "execution_identity.images.task_ref")
    _unresolved_string(images["init_ref"], immutable_image, "execution_identity.images.init_ref")

    environment = _closed_mapping(
        identity["environment"], set(EXPECTED_EXECUTION_IDENTITY["environment"]), "execution_identity.environment"
    )
    _exact_bool(
        environment["inherit_host_environment"],
        False,
        "execution_identity.environment.inherit_host_environment",
    )
    if environment["allowed_names"] != []:
        raise ValueError("execution_identity.environment.allowed_names must be the empty list")

    iris = _closed_mapping(identity["iris"], set(EXPECTED_EXECUTION_IDENTITY["iris"]), "execution_identity.iris")
    _unresolved_string(iris["controller_revision"], r"[0-9a-f]{40}", "execution_identity.iris.controller_revision")
    _unresolved_string(iris["config_sha256"], r"[0-9a-f]{64}", "execution_identity.iris.config_sha256")
    _exact_string(
        iris["checked_in_config_path"],
        EXPECTED_EXECUTION_IDENTITY["iris"]["checked_in_config_path"],
        "execution_identity.iris.checked_in_config_path",
    )
    _exact_string(
        iris["checked_in_config_sha256"],
        EXPECTED_EXECUTION_IDENTITY["iris"]["checked_in_config_sha256"],
        "execution_identity.iris.checked_in_config_sha256",
    )
    if _sha256(REPOSITORY_ROOT / "lib/iris/config/cw-us-east-02a.yaml") != iris["checked_in_config_sha256"]:
        raise ValueError("execution_identity.iris checked-in config digest changed")

    proof = _closed_mapping(
        identity["post_submit_bundle_proof"],
        set(EXPECTED_EXECUTION_IDENTITY["post_submit_bundle_proof"]),
        "execution_identity.post_submit_bundle_proof",
    )
    _exact_int(proof["schema_version"], 2, "execution_identity.post_submit_bundle_proof.schema_version")
    _exact_string(
        proof["status"],
        EXPECTED_EXECUTION_IDENTITY["post_submit_bundle_proof"]["status"],
        "execution_identity.post_submit_bundle_proof.status",
    )
    expected_proof_fields = EXPECTED_EXECUTION_IDENTITY["post_submit_bundle_proof"]["fields"]
    proof_fields = _closed_mapping(
        proof["fields"], set(expected_proof_fields), "execution_identity.post_submit_bundle_proof.fields"
    )
    for field, expected in expected_proof_fields.items():
        _exact_string(proof_fields[field], expected, f"execution_identity.post_submit_bundle_proof.fields.{field}")
    _exact_string(
        proof["identity_rule"],
        EXPECTED_EXECUTION_IDENTITY["post_submit_bundle_proof"]["identity_rule"],
        "execution_identity.post_submit_bundle_proof.identity_rule",
    )

    if identity != EXPECTED_EXECUTION_IDENTITY:
        raise ValueError("execution_identity contract changed")
    return identity


def _validate_dependency_inputs(path: Path) -> dict[str, Any]:
    payload = _load_strict_json(path)
    if not isinstance(payload, dict):
        raise ValueError("dependency input contract must be a JSON object")
    expected_fields = {
        "schema_version",
        "target",
        "build_isolation",
        "install_mode",
        "lock_ready",
        "unresolved",
        "uv_build_resolution",
        "packages",
    }
    if set(payload) != expected_fields:
        raise ValueError("dependency input contract fields changed")
    _exact_int(payload["schema_version"], 2, "dependency_inputs.schema_version")
    if payload.get("target") != {
        "architecture": "x86_64",
        "operating_system": "linux",
        "python_abi": "cp312",
    }:
        raise ValueError("dependency input contract target changed")
    _exact_bool(payload["build_isolation"], False, "dependency_inputs.build_isolation")
    _exact_string(payload["install_mode"], "only_binary_require_hashes", "dependency_inputs.install_mode")
    _exact_bool(payload["lock_ready"], False, "dependency_inputs.lock_ready")
    if payload.get("unresolved") != ["uv-build"]:
        raise ValueError("dependency input contract must remain incomplete until uv-build is pinned")
    resolution = _closed_mapping(
        payload["uv_build_resolution"], set(EXPECTED_UV_BUILD_RESOLUTION), "dependency_inputs.uv_build_resolution"
    )
    for field in ("repository_lock_package", "checked_in_wheel_path", "checked_in_wheel_sha256"):
        _exact_none(resolution[field], f"dependency_inputs.uv_build_resolution.{field}")
    if resolution != EXPECTED_UV_BUILD_RESOLUTION:
        raise ValueError("dependency input uv-build resolution workflow changed")
    packages = payload.get("packages")
    expected_names = [
        "iniconfig",
        "ml-dtypes",
        "numpy",
        "opt-einsum",
        "packaging",
        "pluggy",
        "pygments",
        "pytest",
        "pytest-timeout",
        "scipy",
        "setuptools",
        "wheel",
        "uv-build",
    ]
    if (
        not isinstance(packages, list)
        or [package.get("name") for package in packages if isinstance(package, dict)] != expected_names
    ):
        raise ValueError("dependency input contract package set changed")
    if any(
        not isinstance(package, dict) or set(package) != {"name", "version", "url", "sha256"} for package in packages
    ):
        raise ValueError("dependency input contract package fields changed")
    for package in packages[:-1]:
        if not isinstance(package["version"], str) or not package["version"]:
            raise ValueError("dependency input contract has an unpinned known version")
        if not isinstance(package["url"], str) or not package["url"].startswith("https://files.pythonhosted.org/"):
            raise ValueError("dependency input contract has an unpinned known wheel URL")
        if not isinstance(package["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", package["sha256"]) is None:
            raise ValueError("dependency input contract has an invalid known wheel digest")
    if packages[-1] != {"name": "uv-build", "version": None, "url": None, "sha256": None}:
        raise ValueError("dependency input contract must leave only uv-build unresolved")
    _exact_none(packages[-1]["version"], "dependency_inputs.packages.uv-build.version")
    _exact_none(packages[-1]["url"], "dependency_inputs.packages.uv-build.url")
    _exact_none(packages[-1]["sha256"], "dependency_inputs.packages.uv-build.sha256")
    if _sha256(path) != EXPECTED_DEPENDENCY_INPUT_SHA256:
        raise ValueError("dependency input contract digest changed")
    locked_packages = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())["package"]
    locked_by_name = {package["name"]: package for package in locked_packages}
    for package in packages[:-1]:
        locked = locked_by_name.get(package["name"])
        expected_wheel = {"url": package["url"], "hash": f"sha256:{package['sha256']}"}
        if locked is None or locked.get("version") != package["version"]:
            raise ValueError("dependency input contract differs from the repository lock")
        if not any(
            wheel.get("url") == expected_wheel["url"] and wheel.get("hash") == expected_wheel["hash"]
            for wheel in locked.get("wheels", [])
        ):
            raise ValueError("dependency input contract wheel differs from the repository lock")
    if "uv-build" in locked_by_name:
        raise ValueError(
            "dependency input contract must be completed now that uv-build is present in the repository lock"
        )
    return payload


def validate_submitted_environment(value: object) -> None:
    """Reject every caller-supplied environment variable for this sealed run."""
    if type(value) is not dict or value:
        raise ValueError("submitted environment must be an empty JSON object")


def validate_post_submit_receipt(
    preparation_report: object,
    receipt: object,
    *,
    expected_init_image: str,
) -> None:
    """Validate public Iris launch/status/task observations against reviewed bytes."""
    if not isinstance(preparation_report, dict):
        raise ValueError("preparation report must be a JSON object")
    expected_fields = {
        "schema_version",
        "bundle_manifest_sha256",
        "expected_extraction_manifest_sha256",
        "launch_response_job_id",
        "persisted_bundle_id",
        "persisted_bundle_init_image",
        "reviewed_bundle_sha256",
        "status_response_job_id",
        "task_iris_bundle_id",
        "task_iris_bundle_init_image",
        "task_iris_num_tasks",
        "task_iris_task_id",
    }
    value = _closed_mapping(receipt, expected_fields, "post_submit_receipt")
    _exact_int(value["schema_version"], 1, "post_submit_receipt.schema_version")
    sha_fields = (
        "bundle_manifest_sha256",
        "expected_extraction_manifest_sha256",
        "persisted_bundle_id",
        "reviewed_bundle_sha256",
        "task_iris_bundle_id",
    )
    for field in sha_fields:
        if type(value[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            raise ValueError(f"post_submit_receipt.{field} must be a lower-case SHA-256")
    for field in ("launch_response_job_id", "status_response_job_id"):
        if type(value[field]) is not str or re.fullmatch(r"/[^/\s]+/[^/\s]+", value[field]) is None:
            raise ValueError(f"post_submit_receipt.{field} must be a root Iris job wire ID")
    _exact_string(value["task_iris_num_tasks"], "1", "post_submit_receipt.task_iris_num_tasks")
    immutable_image = r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
    if type(expected_init_image) is not str or re.fullmatch(immutable_image, expected_init_image) is None:
        raise ValueError("expected_init_image must be an immutable OCI reference")
    for field in ("persisted_bundle_init_image", "task_iris_bundle_init_image"):
        _exact_string(value[field], expected_init_image, f"post_submit_receipt.{field}")
    preparation_fields = {
        "bundle_manifest_sha256",
        "expected_extraction_manifest_sha256",
        "bundle_sha256",
    }
    expected: dict[str, str] = {}
    for field in preparation_fields:
        item = preparation_report.get(field)
        if type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None:
            raise ValueError(f"preparation_report.{field} must be a lower-case SHA-256")
        expected[field] = item
    _exact_string(
        value["bundle_manifest_sha256"],
        expected["bundle_manifest_sha256"],
        "post_submit_receipt.bundle_manifest_sha256",
    )
    _exact_string(
        value["expected_extraction_manifest_sha256"],
        expected["expected_extraction_manifest_sha256"],
        "post_submit_receipt.expected_extraction_manifest_sha256",
    )
    reviewed = expected["bundle_sha256"]
    _exact_string(value["reviewed_bundle_sha256"], reviewed, "post_submit_receipt.reviewed_bundle_sha256")
    if value["launch_response_job_id"] != value["status_response_job_id"]:
        raise ValueError("post-submit launch and status job IDs differ")
    expected_task_id = f"{value['launch_response_job_id']}/0:0"
    _exact_string(value["task_iris_task_id"], expected_task_id, "post_submit_receipt.task_iris_task_id")
    if value["persisted_bundle_id"] != reviewed or value["task_iris_bundle_id"] != reviewed:
        raise ValueError("post-submit bundle IDs differ from the reviewed bundle bytes")


def load_and_validate_post_submit_receipt(
    preparation_report_path: Path,
    receipt_path: Path,
    *,
    expected_init_image: str,
) -> None:
    """Strictly load and validate a post-submit receipt and preparation report."""
    validate_post_submit_receipt(
        _load_strict_json(preparation_report_path),
        _load_strict_json(receipt_path),
        expected_init_image=expected_init_image,
    )


def load_and_validate_manifest(path: Path) -> dict[str, Any]:
    payload = _load_strict_json(path)
    if not isinstance(payload, dict):
        raise ValueError("preparation manifest must be a JSON object")
    if set(payload) != EXPECTED_MANIFEST_FIELDS:
        raise ValueError("preparation manifest fields changed")
    _exact_int(payload["schema_version"], 2, "schema_version")
    if payload.get("preparation_base_commit") != EXPECTED_BASE_COMMIT:
        raise ValueError("preparation base commit changed")
    _exact_int(payload["pipeline_abi_version"], 5, "pipeline_abi_version")
    retry_limits = _closed_mapping(payload["retry_limits"], {"failure", "preemption", "task_failure"}, "retry_limits")
    for field in ("failure", "preemption", "task_failure"):
        _exact_int(retry_limits[field], 0, f"retry_limits.{field}")
    unresolved = payload.get("unresolved_external_identities")
    if (
        not isinstance(unresolved, list)
        or set(unresolved) != REQUIRED_EXTERNAL_IDENTITIES
        or len(unresolved) != len(REQUIRED_EXTERNAL_IDENTITIES)
    ):
        raise ValueError("unresolved external identity set changed")
    _exact_bool(payload["launch_ready"], False, "launch_ready")
    _exact_bool(payload["scorecard_status_changed"], False, "scorecard_status_changed")
    if payload.get("sealed_artifact_prohibition") != SEALED_ARTIFACT:
        raise ValueError("sealed jaxacceptance6 artifact path changed")
    toolchain = payload.get("toolchain", {})
    if toolchain != EXPECTED_TOOLCHAIN:
        raise ValueError("pinned toolchain identity changed")
    _validate_execution_identity(payload["execution_identity"])
    _validate_dependency_inputs(DEPENDENCY_INPUT_SOURCE)
    if payload.get("patch_sha256") != EXPECTED_PATCHES:
        raise ValueError("pinned patch identity changed")
    contract = payload.get("target1_contract", {})
    if isinstance(contract, dict):
        _exact_int(contract.get("wrapper_count"), 12, "target1_contract.wrapper_count")
        shapes = contract.get("shapes")
        if not isinstance(shapes, list) or len(shapes) != 2 or any(not isinstance(shape, list) for shape in shapes):
            raise ValueError("target1_contract.shapes changed")
        for shape_index, (shape, expected_shape) in enumerate(zip(shapes, ((2048, 4096), (7, 13)), strict=True)):
            if len(shape) != 2:
                raise ValueError(f"target1_contract.shapes.{shape_index} changed")
            for dimension_index, expected in enumerate(expected_shape):
                _exact_int(shape[dimension_index], expected, f"target1_contract.shapes.{shape_index}.{dimension_index}")
    if contract != {
        "boundaries": ["forward", "backward", "composed"],
        "dtype": "bfloat16",
        "oracle_status": "not_pinned",
        "policies": ["source_ordered", "fast"],
        "shapes": [[2048, 4096], [7, 13]],
        "wrapper_count": 12,
        "future_fast_rewrite_rule": (
            "Revise and independently review numerical tolerances before any non-bitwise FAST rewrite or timing run."
        ),
    }:
        raise ValueError("Target 1 installed-wheel contract changed")
    capsule_allowlist = payload.get("capsule_allowlist")
    if isinstance(capsule_allowlist, dict):
        _exact_int(
            capsule_allowlist.get("tracked_path_count"),
            EXPECTED_CAPSULE_PATH_COUNT,
            "capsule_allowlist.tracked_path_count",
        )
    if capsule_allowlist != {
        "root": "lib/shuttle",
        "tracked_path_count": EXPECTED_CAPSULE_PATH_COUNT,
        "tracked_path_set_sha256": EXPECTED_CAPSULE_PATH_SET_SHA256,
    }:
        raise ValueError("capsule allowlist changed")
    resource_request = _closed_mapping(
        payload["resource_request"], {"cpu", "disk_gb", "gpu", "memory_gib", "timeout_seconds"}, "resource_request"
    )
    expected_resources = {"cpu": 24, "disk_gb": 250, "gpu": 0, "memory_gib": 96, "timeout_seconds": 14400}
    for field, expected in expected_resources.items():
        _exact_int(resource_request[field], expected, f"resource_request.{field}")
    if resource_request != {
        "cpu": 24,
        "disk_gb": 250,
        "gpu": 0,
        "memory_gib": 96,
        "timeout_seconds": 14400,
    }:
        raise ValueError("resource request changed")
    if payload.get("destination") != "s3://marin-us-east-02a/iris/cw-us-east-02a/state/bundles":
        raise ValueError("external destination changed")
    return payload


def _tracked_shuttle_files(repository_root: Path) -> tuple[tuple[Path, int], ...]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "lib/shuttle"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        metadata, relative = line.split("\t", 1)
        git_mode = metadata.split(" ", 1)[0]
        path = Path(relative)
        if not relative or any(component in FORBIDDEN_COMPONENTS for component in path.parts):
            continue
        source = repository_root / path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"capsule source must be a regular file: {relative}")
        mode = 0o755 if git_mode == "100755" else 0o644
        files.append((path, mode))
    required = {
        Path("lib/shuttle/mlir/jax_patch/shuttle_jaxlib_target1_acceptance.py"),
        Path("lib/shuttle/mlir/jax_patch/target1_acceptance_contract.py"),
        Path("lib/shuttle/mlir/jax_patch/test_shuttle_jaxlib_target1_acceptance.py"),
        Path("lib/shuttle/mlir/jax_patch/test_target1_acceptance_contract.py"),
    }
    paths = {path for path, _ in files}
    if not required.issubset(paths):
        raise ValueError("Target 1 installed-wheel driver sources are missing")
    files = tuple(sorted(files))
    path_set_digest = hashlib.sha256(("\n".join(path.as_posix() for path, _ in files) + "\n").encode()).hexdigest()
    if len(files) != EXPECTED_CAPSULE_PATH_COUNT or path_set_digest != EXPECTED_CAPSULE_PATH_SET_SHA256:
        raise ValueError("tracked lib/shuttle capsule path set changed")
    return files


def _copy_capsule_sources(repository_root: Path, capsule: Path) -> None:
    for relative, mode in _tracked_shuttle_files(repository_root):
        destination = capsule / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative, destination)
        os.chmod(destination, mode)
    shutil.copyfile(MANIFEST_SOURCE, capsule / "acceptance-manifest.json")
    os.chmod(capsule / "acceptance-manifest.json", 0o644)
    shutil.copyfile(DEPENDENCY_INPUT_SOURCE, capsule / "linux-dependency-inputs.json")
    os.chmod(capsule / "linux-dependency-inputs.json", 0o644)
    shutil.copyfile(RUNNER_SOURCE, capsule / "run_abi5_cpu_acceptance_preflight.sh")
    os.chmod(capsule / "run_abi5_cpu_acceptance_preflight.sh", 0o755)
    shutil.copyfile(RECEIPT_VERIFIER_SOURCE, capsule / "verify_abi5_cpu_post_submit_receipt.py")
    os.chmod(capsule / "verify_abi5_cpu_post_submit_receipt.py", 0o644)


def _validate_repository_inputs(repository_root: Path, manifest: dict[str, Any]) -> str:
    options = (repository_root / "lib/shuttle/src/shuttle/options.py").read_text()
    registration = (repository_root / "lib/shuttle/mlir/lib/Transforms/XlaRegistration.cc").read_text()
    if "PIPELINE_ABI_VERSION = 5" not in options or "kPipelineAbiVersion = 5" not in registration:
        raise ValueError("Python and C++ pipeline ABI pins must both be 5")
    for relative, expected in manifest["patch_sha256"].items():
        actual = _sha256(repository_root / relative)
        if actual != expected:
            raise ValueError(f"pinned patch digest changed: {relative}")
    sealed = repository_root / manifest["sealed_artifact_prohibition"]
    if not sealed.is_dir():
        raise ValueError("sealed jaxacceptance6 artifact is missing")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, source_commit],
            cwd=repository_root,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("capsule source is not descended from the reviewed preparation base")
    return source_commit


def _source_inventory(capsule: Path, *, include_mode: bool) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in capsule.rglob("*") if item.is_file()):
        relative = path.relative_to(capsule).as_posix()
        if any(component in FORBIDDEN_COMPONENTS for component in Path(relative).parts):
            raise ValueError(f"forbidden capsule member: {relative}")
        item: dict[str, Any] = {
            "path": relative,
            "type": "file",
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if include_mode:
            item["mode"] = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        inventory.append(item)
    return inventory


def _zip_inventory(bundle: Path) -> list[dict[str, Any]]:
    inventory = []
    observed_paths = set()
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if (
                info.filename.startswith("/")
                or "\\" in info.filename
                or any(component in ("", ".", "..") for component in path.parts)
                or path.as_posix() != info.filename
            ):
                raise ValueError(f"unsafe ZIP member path: {info.filename}")
            if info.filename in observed_paths:
                raise ValueError(f"duplicate ZIP member path: {info.filename}")
            observed_paths.add(info.filename)
            payload = archive.read(info)
            inventory.append(
                {
                    "path": info.filename,
                    "type": "file",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "mode": f"{(info.external_attr >> 16) & 0o7777:04o}",
                }
            )
    return inventory


def _create_deterministic_zip(capsule: Path, bundle: Path) -> None:
    """Write content-addressed ZIP bytes without filesystem-time dependencies."""
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(item for item in capsule.rglob("*") if item.is_file()):
            relative = path.relative_to(capsule).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(path.stat().st_mode)) << 16
            archive.writestr(info, path.read_bytes())


def prepare_capsule(repository_root: Path, output: Path) -> dict[str, Any]:
    manifest = load_and_validate_manifest(MANIFEST_SOURCE)
    source_commit = _validate_repository_inputs(repository_root, manifest)
    if output.exists():
        raise ValueError("preparation output must not already exist")
    output.mkdir(parents=True)
    capsule = output / "capsule"
    capsule.mkdir()
    _copy_capsule_sources(repository_root, capsule)
    source_manifest = _source_inventory(capsule, include_mode=True)
    extraction_manifest = _source_inventory(capsule, include_mode=False)
    bundle = output / "bundle.zip"
    _create_deterministic_zip(capsule, bundle)
    zip_manifest = _zip_inventory(bundle)
    if zip_manifest != source_manifest:
        raise AssertionError("ZIP central-directory inventory differs from reviewed capsule source")
    if [{key: item[key] for key in ("path", "type", "size", "sha256")} for item in zip_manifest] != extraction_manifest:
        raise AssertionError("expected extraction inventory differs from ZIP payload")
    (output / "bundle-members.json").write_text(json.dumps(zip_manifest, indent=2, sort_keys=True) + "\n")
    (output / "expected-extraction.json").write_text(json.dumps(extraction_manifest, indent=2, sort_keys=True) + "\n")
    report = {
        "bundle_content_id": _sha256(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "bundle_manifest_sha256": _canonical_sha256(zip_manifest),
        "dependency_inputs_sha256": _sha256(capsule / "linux-dependency-inputs.json"),
        "execution_identity_schema_version": manifest["execution_identity"]["schema_version"],
        "expected_extraction_manifest_sha256": _canonical_sha256(extraction_manifest),
        "launch_ready": manifest["launch_ready"],
        "member_count": len(zip_manifest),
        "runner_sha256": _sha256(capsule / "run_abi5_cpu_acceptance_preflight.sh"),
        "source_commit": source_commit,
        "unresolved_external_identities": manifest["unresolved_external_identities"],
    }
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _require_clean_exact_source(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("capsule preparation requires a clean reviewed worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("capsule source commit is not an exact SHA")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, head],
            cwd=repository_root,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("capsule source commit is not descended from the reviewed preparation base")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if not arguments.allow_dirty_for_tests:
        _require_clean_exact_source(arguments.repository_root)
    report = prepare_capsule(arguments.repository_root, arguments.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
