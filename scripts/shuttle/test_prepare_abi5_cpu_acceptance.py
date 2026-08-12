# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for local-only ABI 5 CPU capsule preparation."""

import json
import os
import re
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest
from iris.cluster.client.bundle import create_workspace_zip
from prepare_abi5_cpu_acceptance import (
    DEPENDENCY_INPUT_SOURCE,
    MANIFEST_SOURCE,
    _validate_dependency_inputs,
    _zip_inventory,
    load_and_validate_manifest,
    load_and_validate_post_submit_receipt,
    prepare_capsule,
    validate_post_submit_receipt,
    validate_submitted_environment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_manifest_fails_closed_until_external_identities_exist() -> None:
    manifest = load_and_validate_manifest(MANIFEST_SOURCE)
    assert manifest["schema_version"] == 2
    assert manifest["pipeline_abi_version"] == 5
    assert manifest["launch_ready"] is False
    assert manifest["retry_limits"] == {
        "failure": 0,
        "preemption": 0,
        "task_failure": 0,
    }
    assert set(manifest["unresolved_external_identities"]) == {
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
    identity = manifest["execution_identity"]
    assert identity["schema_version"] == 2
    assert identity["platform"] == {
        "architecture": "x86_64",
        "operating_system": "linux",
        "python_abi": "cp312",
    }
    assert identity["python"] == {
        "build_identity": None,
        "executable_sha256": None,
        "implementation": "CPython",
        "version": "3.12.11",
    }
    assert identity["images"] == {"init_ref": None, "task_ref": None}
    assert identity["environment"]["allowed_names"] == []
    assert identity["environment"]["inherit_host_environment"] is False
    assert identity["iris"]["minimum_contract_commit"] == "e0689926329548e0b0c987b1e197c67c189c4523"
    assert identity["post_submit_bundle_proof"]["status"] == "required_after_submission"


def test_checked_in_dependency_inputs_pin_every_locally_known_wheel() -> None:
    contract = _validate_dependency_inputs(DEPENDENCY_INPUT_SOURCE)
    packages = contract["packages"]
    assert len(packages) == 13
    assert all(package["url"].startswith("https://files.pythonhosted.org/") for package in packages[:-1])
    assert all(package["sha256"] is not None for package in packages[:-1])
    assert packages[-1] == {"name": "uv-build", "sha256": None, "url": None, "version": None}
    assert contract["lock_ready"] is False
    assert contract["build_isolation"] is False
    assert contract["uv_build_resolution"]["repository_lock_package"] is None
    assert contract["uv_build_resolution"]["checked_in_wheel_sha256"] is None


def test_submitted_environment_is_closed_and_empty() -> None:
    validate_submitted_environment({})
    for environment in ({"HF_TOKEN": "secret"}, {"PATH": "/usr/bin"}, [], None):
        with pytest.raises(ValueError, match="submitted environment must be an empty JSON object"):
            validate_submitted_environment(environment)


def _valid_post_submit_receipt() -> tuple[dict[str, object], dict[str, object], str]:
    reviewed = "1" * 64
    manifest = "2" * 64
    extraction = "3" * 64
    image = "registry.example/iris/bundle-init@sha256:" + "4" * 64
    preparation = {
        "bundle_sha256": reviewed,
        "bundle_manifest_sha256": manifest,
        "expected_extraction_manifest_sha256": extraction,
    }
    receipt = {
        "schema_version": 1,
        "bundle_manifest_sha256": manifest,
        "expected_extraction_manifest_sha256": extraction,
        "launch_response_job_id": "/user/acceptance",
        "persisted_bundle_id": reviewed,
        "persisted_bundle_init_image": image,
        "reviewed_bundle_sha256": reviewed,
        "status_response_job_id": "/user/acceptance",
        "task_iris_bundle_id": reviewed,
        "task_iris_bundle_init_image": image,
        "task_iris_num_tasks": "1",
        "task_iris_task_id": "/user/acceptance/0:0",
    }
    return preparation, receipt, image


def test_post_submit_receipt_binds_public_iris_observations_to_reviewed_bytes() -> None:
    preparation, receipt, image = _valid_post_submit_receipt()
    validate_post_submit_receipt(preparation, receipt, expected_init_image=image)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", True, "schema_version"),
        ("launch_response_job_id", "/user/other", "job IDs differ"),
        ("persisted_bundle_id", "5" * 64, "bundle IDs differ"),
        ("task_iris_bundle_id", "5" * 64, "bundle IDs differ"),
        ("persisted_bundle_init_image", "registry.example/image:latest", "persisted_bundle_init_image"),
        ("task_iris_bundle_init_image", None, "task_iris_bundle_init_image"),
        ("task_iris_num_tasks", 1, "task_iris_num_tasks"),
        ("task_iris_task_id", "/user/acceptance/1:0", "task_iris_task_id"),
        ("bundle_manifest_sha256", "6" * 64, "bundle_manifest_sha256"),
    ),
)
def test_post_submit_receipt_mutations_fail_closed(field: str, value: object, message: str) -> None:
    preparation, receipt, image = _valid_post_submit_receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match=message):
        validate_post_submit_receipt(preparation, receipt, expected_init_image=image)


def test_post_submit_receipt_rejects_unknown_fields() -> None:
    preparation, receipt, image = _valid_post_submit_receipt()
    receipt["controller_claim"] = "trusted"
    with pytest.raises(ValueError, match="fields changed"):
        validate_post_submit_receipt(preparation, receipt, expected_init_image=image)


def test_post_submit_receipt_file_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    preparation, receipt, image = _valid_post_submit_receipt()
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    receipt_path = tmp_path / "receipt.json"
    encoded = json.dumps(receipt)
    receipt_path.write_text(encoded[:-1] + ',"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        load_and_validate_post_submit_receipt(preparation_path, receipt_path, expected_init_image=image)


def test_post_submit_receipt_verifier_accepts_public_iris_receipt(tmp_path: Path) -> None:
    preparation, receipt, image = _valid_post_submit_receipt()
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/shuttle/verify_abi5_cpu_post_submit_receipt.py"),
            "--preparation-report",
            str(preparation_path),
            "--receipt",
            str(receipt_path),
            "--expected-init-image",
            image,
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("pipeline_abi_version",), 4, "pipeline_abi_version"),
        (("retry_limits", "preemption"), 1, "retry_limits.preemption"),
        (("launch_ready",), True, "launch_ready"),
        (("toolchain", "jax_revision"), "0" * 40, "pinned toolchain identity"),
        (("toolchain", "stablehlo_revision"), "0" * 40, "pinned toolchain identity"),
        (("execution_identity", "images", "task_ref"), "image:latest", "execution_identity.images.task_ref"),
        (("execution_identity", "python", "version"), "3.12.12", "execution_identity"),
        (
            ("execution_identity", "environment", "allowed_names"),
            ["PATH", "PATH"],
            "execution_identity.environment.allowed_names",
        ),
        (
            ("execution_identity", "environment", "inherit_host_environment"),
            True,
            "execution_identity.environment.inherit_host_environment",
        ),
        (
            ("execution_identity", "iris", "checked_in_config_sha256"),
            "0" * 64,
            "execution_identity.iris.checked_in_config_sha256",
        ),
        (
            ("execution_identity", "iris", "controller_revision"),
            "0" * 40,
            "execution_identity.iris.controller_revision",
        ),
        (("sealed_artifact_prohibition",), "lib/shuttle/mlir/artifacts/other", "sealed jaxacceptance6"),
    ),
)
def test_manifest_mutations_fail_closed(tmp_path: Path, path: tuple[str | int, ...], value, message: str) -> None:
    payload = json.loads(MANIFEST_SOURCE.read_text())
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    mutated = tmp_path / "manifest.json"
    mutated.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_and_validate_manifest(mutated)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("schema_version",), True, "schema_version"),
        (("pipeline_abi_version",), 5.0, "pipeline_abi_version"),
        (("launch_ready",), 0, "launch_ready"),
        (("scorecard_status_changed",), None, "scorecard_status_changed"),
        (("retry_limits", "failure"), False, "retry_limits.failure"),
        (("retry_limits", "preemption"), "0", "retry_limits.preemption"),
        (("resource_request", "cpu"), 24.0, "resource_request.cpu"),
        (("resource_request", "gpu"), False, "resource_request.gpu"),
        (("capsule_allowlist", "tracked_path_count"), 140.0, "capsule_allowlist.tracked_path_count"),
        (("target1_contract", "wrapper_count"), True, "target1_contract.wrapper_count"),
        (("target1_contract", "shapes", 0, 0), 2048.0, "target1_contract.shapes.0.0"),
        (("execution_identity", "schema_version"), True, "execution_identity.schema_version"),
        (
            ("execution_identity", "dependency_inputs", "lock_ready"),
            0,
            "execution_identity.dependency_inputs.lock_ready",
        ),
        (("execution_identity", "python", "build_identity"), 1, "execution_identity.python.build_identity"),
        (
            ("execution_identity", "python", "executable_sha256"),
            False,
            "execution_identity.python.executable_sha256",
        ),
        (("execution_identity", "images", "task_ref"), 1.0, "execution_identity.images.task_ref"),
        (
            ("execution_identity", "environment", "allowed_names"),
            "PATH",
            "execution_identity.environment.allowed_names",
        ),
        (
            ("execution_identity", "iris", "controller_revision"),
            False,
            "execution_identity.iris.controller_revision",
        ),
        (
            ("execution_identity", "post_submit_bundle_proof", "schema_version"),
            1.0,
            "execution_identity.post_submit_bundle_proof.schema_version",
        ),
        (
            ("execution_identity", "post_submit_bundle_proof", "fields", "persisted_bundle_id"),
            None,
            "execution_identity.post_submit_bundle_proof.fields.persisted_bundle_id",
        ),
        (
            ("execution_identity", "post_submit_bundle_proof", "identity_rule"),
            None,
            "execution_identity.post_submit_bundle_proof.identity_rule",
        ),
    ),
)
def test_manifest_rejects_cross_type_scalar_substitutions(
    tmp_path: Path, path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = json.loads(MANIFEST_SOURCE.read_text())
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    mutated = tmp_path / "manifest.json"
    mutated.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=re.escape(message)):
        load_and_validate_manifest(mutated)


def test_preparation_builds_config_free_capsule_with_iris_equivalent_inventory(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    report = prepare_capsule(REPOSITORY_ROOT, output)
    capsule = output / "capsule"
    assert report["launch_ready"] is False
    assert report["bundle_sha256"] == report["bundle_content_id"]
    assert (
        report["dependency_inputs_sha256"]
        == load_and_validate_manifest(MANIFEST_SOURCE)["execution_identity"]["dependency_inputs"]["sha256"]
    )
    assert report["execution_identity_schema_version"] == 2
    assert (output / "bundle.zip").stat().st_size == report["bundle_size"]
    assert (capsule / "run_abi5_cpu_acceptance_preflight.sh").is_file()
    assert (capsule / "acceptance-manifest.json").is_file()
    assert (capsule / "linux-dependency-inputs.json").is_file()
    assert (capsule / "verify_abi5_cpu_post_submit_receipt.py").is_file()
    assert (capsule / "lib/shuttle/mlir/jax_patch/shuttle_jaxlib_target1_acceptance.py").is_file()
    iris_zip = tmp_path / "iris-client.zip"
    iris_zip.write_bytes(create_workspace_zip(capsule))
    assert _zip_inventory(iris_zip) == _zip_inventory(output / "bundle.zip")
    forbidden = (".git", ".venv", "artifacts", "coreweave.yaml", ".marin.yaml")
    members = json.loads((output / "bundle-members.json").read_text())
    assert members
    assert all(not any(component in forbidden for component in Path(item["path"]).parts) for item in members)
    extraction = json.loads((output / "expected-extraction.json").read_text())
    assert all("mode" not in item for item in extraction)
    assert all("mode" in item for item in members)


def test_dependency_input_mutations_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(DEPENDENCY_INPUT_SOURCE.read_text())
    payload["packages"][0]["sha256"] = "0" * 64
    mutated = tmp_path / "dependencies.json"
    mutated.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dependency input contract"):
        _validate_dependency_inputs(mutated)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("schema_version",), True, "schema_version"),
        (("schema_version",), 1.0, "schema_version"),
        (("schema_version",), "1", "schema_version"),
        (("schema_version",), None, "schema_version"),
        (("build_isolation",), 0, "build_isolation"),
        (("lock_ready",), "false", "lock_ready"),
        (("uv_build_resolution", "repository_lock_package"), "uv-build", "repository_lock_package"),
        (("uv_build_resolution", "checked_in_wheel_sha256"), False, "checked_in_wheel_sha256"),
    ),
)
def test_dependency_contract_rejects_cross_type_scalar_substitutions(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    payload = json.loads(DEPENDENCY_INPUT_SOURCE.read_text())
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    mutated = tmp_path / "dependencies.json"
    mutated.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _validate_dependency_inputs(mutated)


def test_two_fresh_preparations_have_identical_content_addressed_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = prepare_capsule(REPOSITORY_ROOT, first)
    second_report = prepare_capsule(REPOSITORY_ROOT, second)
    assert (first / "bundle.zip").read_bytes() == (second / "bundle.zip").read_bytes()
    assert first_report["bundle_sha256"] == second_report["bundle_sha256"]


def test_duplicate_json_keys_and_zip_paths_fail_closed(tmp_path: Path) -> None:
    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_and_validate_manifest(duplicate_json)

    duplicate_zip = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate_zip, "w") as archive:
            archive.writestr("same", b"first")
            archive.writestr("same", b"second")
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        _zip_inventory(duplicate_zip)

    alias_zip = tmp_path / "alias.zip"
    with zipfile.ZipFile(alias_zip, "w") as archive:
        archive.writestr("dir/../escape", b"payload")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        _zip_inventory(alias_zip)


def test_runner_rejects_unresolved_manifest_before_external_work(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    prepare_capsule(REPOSITORY_ROOT, output)
    result = subprocess.run(
        ["bash", "run_abi5_cpu_acceptance_preflight.sh"],
        cwd=output / "capsule",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not launch-ready" in result.stderr


def _resolved_launch(bundle_sha256: str) -> dict[str, str]:
    return {
        "bundle_init_pinning_implementation_review": "7" * 40,
        "bundle_content_sha256": bundle_sha256,
        "exact_bundle_blob_submission_review": "9" * 40,
        "init_image_oci_ref": "registry.example/init@sha256:" + "1" * 64,
        "iris_config_sha256": "2" * 64,
        "iris_revision": "3" * 40,
        "linux_dependency_lock_sha256": "4" * 64,
        "linux_python_identity": "CPython 3.12.11 (linux-x86_64-glibc) sha256:" + "a" * 64,
        "minimal_execution_environment_policy_review": "8" * 40,
        "runner_implementation_review": "5" * 40,
        "task_image_oci_ref": "registry.example/task@sha256:" + "6" * 64,
    }


def test_runner_validates_closed_resolved_schema_and_exact_bundle_id(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    report = prepare_capsule(REPOSITORY_ROOT, output)
    capsule = output / "capsule"
    resolved = _resolved_launch(report["bundle_sha256"])
    (capsule / "resolved-launch.json").write_text(json.dumps(resolved))
    environment = {
        "PATH": os.environ["PATH"],
        "IRIS_BUNDLE_ID": report["bundle_sha256"],
    }
    result = subprocess.run(
        ["bash", "run_abi5_cpu_acceptance_preflight.sh"],
        cwd=capsule,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "external execution is intentionally not implemented" in result.stderr

    environment["IRIS_BUNDLE_ID"] = "0" * 64
    mismatch = subprocess.run(
        ["bash", "run_abi5_cpu_acceptance_preflight.sh"],
        cwd=capsule,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "IRIS_BUNDLE_ID differs" in mismatch.stderr

    resolved["task_image_oci_ref"] = "registry.example/task:latest"
    (capsule / "resolved-launch.json").write_text(json.dumps(resolved))
    mutable_image = subprocess.run(
        ["bash", "run_abi5_cpu_acceptance_preflight.sh"],
        cwd=capsule,
        env={**environment, "IRIS_BUNDLE_ID": report["bundle_sha256"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert mutable_image.returncode != 0
    assert "complete immutable OCI reference" in mutable_image.stderr

    resolved["task_image_oci_ref"] = "registry.example/task@sha256:" + "6" * 64
    resolved["extra"] = "not reviewed"
    (capsule / "resolved-launch.json").write_text(json.dumps(resolved))
    extra = subprocess.run(
        ["bash", "run_abi5_cpu_acceptance_preflight.sh"],
        cwd=capsule,
        env={**environment, "IRIS_BUNDLE_ID": report["bundle_sha256"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert extra.returncode != 0
    assert "exactly close" in extra.stderr
