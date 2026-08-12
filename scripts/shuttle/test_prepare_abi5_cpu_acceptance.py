# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for local-only ABI 5 CPU capsule preparation."""

import json
import os
import subprocess
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
    prepare_capsule,
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
    assert identity["schema_version"] == 1
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


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("pipeline_abi_version",), 4, "pipeline ABI"),
        (("retry_limits", "preemption"), 1, "retry limits"),
        (("launch_ready",), True, "unresolved external"),
        (("toolchain", "jax_revision"), "0" * 40, "pinned toolchain identity"),
        (("toolchain", "stablehlo_revision"), "0" * 40, "pinned toolchain identity"),
        (("execution_identity", "images", "task_ref"), "image:latest", "image references"),
        (("execution_identity", "python", "version"), "3.12.12", "execution identity"),
        (("execution_identity", "environment", "allowed_names"), ["PATH", "PATH"], "execution identity"),
        (("execution_identity", "iris", "controller_revision"), "0" * 40, "execution identity"),
        (("sealed_artifact_prohibition",), "lib/shuttle/mlir/artifacts/other", "sealed jaxacceptance6"),
    ),
)
def test_manifest_mutations_fail_closed(tmp_path: Path, path: tuple[str, ...], value, message: str) -> None:
    payload = json.loads(MANIFEST_SOURCE.read_text())
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    mutated = tmp_path / "manifest.json"
    mutated.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
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
    assert report["execution_identity_schema_version"] == 1
    assert (output / "bundle.zip").stat().st_size == report["bundle_size"]
    assert (capsule / "run_abi5_cpu_acceptance_preflight.sh").is_file()
    assert (capsule / "acceptance-manifest.json").is_file()
    assert (capsule / "linux-dependency-inputs.json").is_file()
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
