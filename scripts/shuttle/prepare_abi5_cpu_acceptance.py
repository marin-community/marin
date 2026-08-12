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
EXPECTED_DEPENDENCY_INPUT_SHA256 = "f872f3aa324f11a7b833ce4f1374ed2cc0118b5d030ae472c6d12a3b113328f7"
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
    "schema_version": 1,
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
        "status": "unresolved_closed_allowlist_required",
        "allowed_names": None,
        "required_names": [
            "HOME",
            "IRIS_BUNDLE_ID",
            "IRIS_BUNDLE_INIT_IMAGE",
            "IRIS_NUM_TASKS",
            "IRIS_TASK_ID",
            "IRIS_WORKDIR",
            "PATH",
            "PWD",
            "TMPDIR",
        ],
        "forbidden_names": [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GCS_RESOLVE_REFRESH_SECS",
            "GITHUB_TOKEN",
            "HF_TOKEN",
            "MARIN_PROVENANCE",
            "WANDB_API_KEY",
        ],
        "forbidden_files": [".marin.yaml", "coreweave.yaml"],
    },
    "iris": {
        "minimum_contract_commit": "e0689926329548e0b0c987b1e197c67c189c4523",
        "controller_revision": None,
        "config_sha256": None,
        "required_capabilities": ["exact_workspace_bundle_bytes", "per_job_bundle_init_image"],
    },
    "post_submit_bundle_proof": {
        "schema_version": 1,
        "status": "required_after_submission",
        "fields": {
            "bundle_manifest_sha256": "lowerhex_sha256",
            "controller_bundle_id": "lowerhex_sha256",
            "expected_extraction_manifest_sha256": "lowerhex_sha256",
            "reviewed_bundle_sha256": "lowerhex_sha256",
            "task_iris_bundle_id": "lowerhex_sha256",
        },
        "identity_rule": "controller_bundle_id == task_iris_bundle_id == reviewed_bundle_sha256",
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
        "packages",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise ValueError("dependency input contract schema changed")
    if payload.get("target") != {
        "architecture": "x86_64",
        "operating_system": "linux",
        "python_abi": "cp312",
    }:
        raise ValueError("dependency input contract target changed")
    if payload.get("build_isolation") is not False or payload.get("install_mode") != "only_binary_require_hashes":
        raise ValueError("dependency input contract installation mode changed")
    if payload.get("lock_ready") is not False or payload.get("unresolved") != ["uv-build"]:
        raise ValueError("dependency input contract must remain incomplete until uv-build is pinned")
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


def load_and_validate_manifest(path: Path) -> dict[str, Any]:
    payload = _load_strict_json(path)
    if not isinstance(payload, dict):
        raise ValueError("preparation manifest must be a JSON object")
    if set(payload) != EXPECTED_MANIFEST_FIELDS:
        raise ValueError("preparation manifest fields changed")
    if payload.get("schema_version") != 2:
        raise ValueError("unknown preparation manifest schema")
    if payload.get("preparation_base_commit") != EXPECTED_BASE_COMMIT:
        raise ValueError("preparation base commit changed")
    if payload.get("pipeline_abi_version") != 5:
        raise ValueError("pipeline ABI must be 5")
    if payload.get("retry_limits") != {"failure": 0, "preemption": 0, "task_failure": 0}:
        raise ValueError("all retry limits must be explicit zero")
    unresolved = payload.get("unresolved_external_identities")
    if (
        not isinstance(unresolved, list)
        or set(unresolved) != REQUIRED_EXTERNAL_IDENTITIES
        or len(unresolved) != len(REQUIRED_EXTERNAL_IDENTITIES)
    ):
        raise ValueError("unresolved external identity set changed")
    if payload.get("launch_ready") is not False:
        raise ValueError("unresolved external identities prohibit a launch-ready manifest")
    if payload.get("scorecard_status_changed") is not False:
        raise ValueError("local preparation must not change scorecard status")
    if payload.get("sealed_artifact_prohibition") != SEALED_ARTIFACT:
        raise ValueError("sealed jaxacceptance6 artifact path changed")
    toolchain = payload.get("toolchain", {})
    if toolchain != EXPECTED_TOOLCHAIN:
        raise ValueError("pinned toolchain identity changed")
    execution_identity = payload.get("execution_identity")
    if not isinstance(execution_identity, dict):
        raise ValueError("execution identity must be an object")
    if execution_identity.get("images") != {"task_ref": None, "init_ref": None}:
        raise ValueError("image references must remain unresolved until immutable OCI references are reviewed")
    if execution_identity != EXPECTED_EXECUTION_IDENTITY:
        raise ValueError("execution identity contract changed")
    _validate_dependency_inputs(DEPENDENCY_INPUT_SOURCE)
    if payload.get("patch_sha256") != EXPECTED_PATCHES:
        raise ValueError("pinned patch identity changed")
    contract = payload.get("target1_contract", {})
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
    if payload.get("capsule_allowlist") != {
        "root": "lib/shuttle",
        "tracked_path_count": EXPECTED_CAPSULE_PATH_COUNT,
        "tracked_path_set_sha256": EXPECTED_CAPSULE_PATH_SET_SHA256,
    }:
        raise ValueError("capsule allowlist changed")
    if payload.get("resource_request") != {
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
