#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the GrugMoE GPU real-checkpoint validation inside an Iris H100 task.

This is intentionally a remote-side script. Submit it with an Iris job that
clones a Marin branch, then runs this script from that checkout. The script
installs vLLM from a coherent source checkout using precompiled native
extensions instead of copying selected PR files into an installed wheel.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VLLM_REPO_URL = "https://github.com/marin-community/vllm.git"
VLLM_VALIDATION_BRANCH = "grugmoe-gpu-validation"
RUN_ID_ENV = "MARIN_GRUGMOE_GPU_E2E_RUN_ID"
OUTPUT_DIR_ENV = "MARIN_GRUGMOE_GPU_E2E_OUTPUT_DIR"
INSTALL_REPORT_PATH_ENV = "MARIN_GRUGMOE_GPU_E2E_INSTALL_REPORT_PATH"
ATTENTION_BACKEND_ENV = "MARIN_GRUGMOE_VLLM_ATTENTION_BACKEND"
COREWEAVE_REGION = "cw-us-east-02a"
VLLM_TARGET_DEVICE = "cuda"
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/grugmoe-gpu-real-checkpoint-e2e"
NATIVE_SENSITIVE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".hip",
    ".rs",
    ".pyx",
)
NATIVE_SENSITIVE_PREFIXES = (
    "csrc/",
    "cmake/",
    "rocm/",
    "rust/",
    "vllm/csrc/",
    "vllm_flash_attn/",
)
NATIVE_SENSITIVE_FILENAMES = {
    "CMakeLists.txt",
    "Cargo.lock",
    "Cargo.toml",
    "setup.py",
    "pyproject.toml",
    "requirements/build.txt",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _default_run_id(attention_backend: str) -> str:
    suffix = attention_backend.lower().removesuffix("_attn")
    return f"{_utc_stamp()}-coherent-{suffix}-{uuid.uuid4().hex[:8]}"


def _join_path(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *(part.strip("/") for part in parts)])


def _venv_env(venv_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = os.pathsep.join([str(venv_dir / "bin"), env.get("PATH", "")])
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def _base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {shlex.join(command)}", flush=True)
    with log_path.open("w") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()
    elapsed = time.time() - started
    report = {
        "command": command,
        "command_text": shlex.join(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "returncode": returncode,
        "elapsed_seconds": elapsed,
    }
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return report


def _check_output(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, stderr=subprocess.STDOUT, text=True).strip()


def _prepend_env_path(env: dict[str, str], name: str, values: list[str]) -> None:
    if not values:
        return
    existing = [value for value in env.get(name, "").split(os.pathsep) if value]
    env[name] = os.pathsep.join(list(dict.fromkeys([*values, *existing])))


def _python_library_dirs(*, venv_python: Path, cwd: Path) -> list[str]:
    code = """
import json
import site
import sysconfig
from pathlib import Path

roots = {
    Path(path)
    for path in [
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
        *site.getsitepackages(),
    ]
    if path
}

library_dirs = []
for root in sorted(roots):
    nvidia_root = root / "nvidia"
    if nvidia_root.is_dir():
        for pattern in ("*/lib", "*/lib64"):
            for lib_dir in sorted(nvidia_root.glob(pattern)):
                if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
                    library_dirs.append(str(lib_dir))

    torch_lib = root / "torch" / "lib"
    if torch_lib.is_dir() and any(torch_lib.glob("*.so*")):
        library_dirs.append(str(torch_lib))

print(json.dumps(list(dict.fromkeys(library_dirs))))
"""
    return json.loads(_check_output([str(venv_python), "-c", code], cwd=cwd).splitlines()[-1])


def _relative_glob(root: Path, patterns: tuple[str, ...], *, limit: int = 200) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                matches.append(str(path.relative_to(root)))
    return sorted(set(matches))[:limit]


def _clone_or_update_vllm(vllm_dir: Path, ref: str) -> None:
    if not vllm_dir.exists():
        _run(
            ["git", "clone", VLLM_REPO_URL, str(vllm_dir)],
            cwd=vllm_dir.parent,
            log_path=vllm_dir.parent / "logs" / "git-clone-vllm.log",
        )
    _run(["git", "fetch", "origin", "--prune"], cwd=vllm_dir, log_path=vllm_dir.parent / "logs" / "git-fetch-vllm.log")
    # vLLM's precompiled-wheel path asks git for the current branch before
    # computing the base commit. A detached checkout falls back to a nightly
    # wheel, so pin the requested ref through a throwaway local branch.
    _run(
        ["git", "checkout", "-B", VLLM_VALIDATION_BRANCH, ref],
        cwd=vllm_dir,
        log_path=vllm_dir.parent / "logs" / "git-checkout-vllm.log",
    )


def _changed_vllm_files(vllm_dir: Path) -> list[str]:
    _run(["git", "fetch", "origin", "main"], cwd=vllm_dir, log_path=vllm_dir.parent / "logs" / "git-fetch-vllm-main.log")
    output = _check_output(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=vllm_dir)
    return [line for line in output.splitlines() if line]


def _native_sensitive_files(files: list[str]) -> list[str]:
    sensitive: list[str] = []
    for path in files:
        if path in NATIVE_SENSITIVE_FILENAMES:
            sensitive.append(path)
            continue
        if path.endswith(NATIVE_SENSITIVE_SUFFIXES):
            sensitive.append(path)
            continue
        if path.startswith(NATIVE_SENSITIVE_PREFIXES):
            sensitive.append(path)
    return sensitive


def _target_python_snapshot(*, venv_python: Path, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    code = """
import importlib
import importlib.metadata as md
import json
import os
import sys
import time

repo_root = os.getcwd()
worker_extension_dir = os.path.join(repo_root, "tests", "vllm")
sys.path.insert(0, worker_extension_dir)
sys.path.insert(0, repo_root)

def direct_url(package):
    try:
        value = md.distribution(package).read_text("direct_url.json")
    except md.PackageNotFoundError:
        return "not-installed"
    return value.strip() if value else ""

def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not-installed"

def import_check(module_name):
    started = time.time()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        }
    return {
        "ok": True,
        "module": module_name,
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
        "elapsed_seconds": time.time() - started,
    }

def worker_extension_check():
    result = import_check("grugmoe_gpu_real_checkpoint_backend")
    result["class"] = "GrugMoeDiagnosticsWorkerExtension"
    result["worker_extension_dir"] = worker_extension_dir
    if result["ok"]:
        module = importlib.import_module(result["module"])
        result["class_present"] = hasattr(module, result["class"])
        result["ok"] = result["class_present"]
    return result

print(json.dumps({
    "packages": {
        package: {"version": version(package), "direct_url": direct_url(package)}
        for package in ("marin-core", "vllm", "tpu-inference", "jax", "torch", "torchvision", "torchaudio")
    },
    "imports": {
        "vllm": import_check("vllm"),
        "vllm._C": import_check("vllm._C"),
        "grugmoe": import_check("vllm.model_executor.models.grugmoe"),
        "marin_worker_extension": worker_extension_check(),
    },
}, sort_keys=True))
"""
    output = _check_output([str(venv_python), "-c", code], cwd=cwd, env=env)
    return json.loads(output.splitlines()[-1])


def _native_artifact_snapshot(*, venv_python: Path, vllm_dir: Path, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    code = """
import json
import site
import sysconfig
from pathlib import Path

roots = {
    path
    for path in [
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
        *site.getsitepackages(),
    ]
    if path
}

def rel_matches(root, patterns):
    root_path = Path(root)
    matches = []
    for pattern in patterns:
        matches.extend(
            str(path.relative_to(root_path))
            for path in root_path.glob(pattern)
            if path.exists()
        )
    return sorted(set(matches))[:200]

print(json.dumps({
    str(root): rel_matches(root, ("vllm*.pth", "__editable__*", "vllm/*.so", "vllm/**/*.so"))
    for root in sorted(roots)
}, sort_keys=True))
"""
    site_packages = json.loads(_check_output([str(venv_python), "-c", code], cwd=cwd, env=env).splitlines()[-1])
    return {
        "source_tree": _relative_glob(vllm_dir, ("vllm/*.so", "vllm/**/*.so")),
        "editable_link_tree": _relative_glob(
            vllm_dir,
            ("build/__editable__*/vllm/*.so", "build/__editable__*/vllm/**/*.so"),
        ),
        "site_packages": site_packages,
    }


def _build_install_report(
    *,
    run_id: str,
    output_dir: str,
    install_report_path: str,
    venv_python: Path,
    vllm_dir: Path,
    changed_files: list[str],
    python_only_check: dict[str, Any],
    command_reports: list[dict[str, Any]],
    runtime_env: dict[str, str],
    cuda_library_dirs: list[str],
) -> dict[str, Any]:
    target_python = _target_python_snapshot(venv_python=venv_python, cwd=_repo_root(), env=runtime_env)
    native_artifacts = _native_artifact_snapshot(
        venv_python=venv_python, vllm_dir=vllm_dir, cwd=_repo_root(), env=runtime_env
    )
    ld_library_path_entries = [value for value in runtime_env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if value]
    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "install_report_path": install_report_path,
        "marin_sha": _check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root()),
        "vllm_sha": _check_output(["git", "rev-parse", "HEAD"], cwd=vllm_dir),
        "vllm_source_dir": str(vllm_dir),
        "vllm_changed_files": changed_files,
        "python_only_check": python_only_check,
        "commands": command_reports,
        "python_executable": str(venv_python),
        "packages": target_python["packages"],
        "imports": target_python["imports"],
        "native_artifacts": native_artifacts,
        "cuda_library_path": {
            "added_library_dirs": cuda_library_dirs,
            "ld_library_path_entry_count": len(ld_library_path_entries),
        },
        "environment": {
            "VLLM_TARGET_DEVICE": os.environ.get("VLLM_TARGET_DEVICE"),
            "VLLM_USE_PRECOMPILED": os.environ.get("VLLM_USE_PRECOMPILED"),
            "VERBOSE": os.environ.get("VERBOSE"),
            "LD_LIBRARY_PATH_ENTRY_COUNT": len(ld_library_path_entries),
        },
        "created_at": _utc_stamp(),
    }


def _failed_required_imports(install_report: dict[str, Any]) -> dict[str, Any]:
    required_imports = ("vllm", "vllm._C", "grugmoe", "marin_worker_extension")
    imports = install_report.get("imports", {})
    return {
        name: imports.get(name, {"ok": False, "error": "missing import check"})
        for name in required_imports
        if imports.get(name, {}).get("ok") is not True
    }


def _upload_validation_artifacts(
    *,
    output_dir: str,
    install_report_path: str,
    install_report: dict[str, Any],
    local_report_path: Path,
    log_dir: Path,
    venv_python: Path,
    repo_root: Path,
) -> None:
    local_report_path.write_text(json.dumps(install_report, indent=2, sort_keys=True) + "\n")
    code = """
import importlib.util
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.getcwd())
backend_path = repo_root / "tests" / "vllm" / "grugmoe_gpu_real_checkpoint_backend.py"
spec = importlib.util.spec_from_file_location("grugmoe_gpu_real_checkpoint_backend_upload", backend_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load backend from {backend_path}")
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
output_dir = os.environ["GRUGMOE_GPU_OUTPUT_DIR"]
install_report_path = os.environ["GRUGMOE_GPU_INSTALL_REPORT_PATH"]
local_report_path = os.environ["GRUGMOE_GPU_LOCAL_REPORT_PATH"]
log_dir = Path(os.environ["GRUGMOE_GPU_LOG_DIR"])

with open(local_report_path) as f:
    install_report = json.load(f)
backend._write_json(install_report_path, install_report)
backend._copy_local_file(local_report_path, backend._join_path(output_dir, "install-report.local.json"))
for log_path in sorted(log_dir.glob("*.log")):
    backend._copy_local_file(str(log_path), backend._join_path(output_dir, "setup-logs", log_path.name))
"""
    upload_env = _venv_env(
        venv_python.parent.parent,
        {
            "PYTHONPATH": os.pathsep.join([str(repo_root), os.environ.get("PYTHONPATH", "")]),
            "GRUGMOE_GPU_OUTPUT_DIR": output_dir,
            "GRUGMOE_GPU_INSTALL_REPORT_PATH": install_report_path,
            "GRUGMOE_GPU_LOCAL_REPORT_PATH": str(local_report_path),
            "GRUGMOE_GPU_LOG_DIR": str(log_dir),
        },
    )
    subprocess.check_call([str(venv_python), "-c", code], cwd=repo_root, env=upload_env)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GrugMoE GPU validation from a coherent vLLM source checkout.")
    parser.add_argument("--vllm-ref", required=True, help="vLLM branch or SHA to validate.")
    parser.add_argument("--vllm-dir", default="/tmp/grugmoe-vllm-src", help="Remote vLLM checkout path.")
    parser.add_argument(
        "--attention-backend",
        default="TRITON_ATTN",
        choices=("TRITON_ATTN", "FLASH_ATTN"),
        help="vLLM attention backend to pass to the GPU e2e.",
    )
    parser.add_argument("--run-id", help=f"Stable result run id. Defaults to {RUN_ID_ENV} or a timestamp.")
    parser.add_argument("--output-dir", help=f"Full result prefix. Defaults to {OUTPUT_ROOT}/<run-id>.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = _repo_root()
    vllm_dir = Path(args.vllm_dir)
    log_dir = Path("/tmp/grugmoe-gpu-validation-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or os.environ.get(RUN_ID_ENV) or _default_run_id(args.attention_backend)
    output_dir = args.output_dir or os.environ.get(OUTPUT_DIR_ENV) or _join_path(OUTPUT_ROOT, run_id)
    install_report_path = _join_path(output_dir, "install-report.json")
    local_report_path = log_dir / "install-report.local.json"

    os.environ[RUN_ID_ENV] = run_id
    os.environ[OUTPUT_DIR_ENV] = output_dir
    os.environ[INSTALL_REPORT_PATH_ENV] = install_report_path
    os.environ[ATTENTION_BACKEND_ENV] = args.attention_backend
    os.environ.setdefault("IRIS_WORKER_REGION", COREWEAVE_REGION)
    os.environ.setdefault("COREWEAVE_REGION", COREWEAVE_REGION)
    os.environ["VLLM_TARGET_DEVICE"] = VLLM_TARGET_DEVICE
    os.environ["VLLM_USE_PRECOMPILED"] = "1"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    command_reports: list[dict[str, Any]] = []
    _clone_or_update_vllm(vllm_dir, args.vllm_ref)
    changed_files = _changed_vllm_files(vllm_dir)
    sensitive_files = _native_sensitive_files(changed_files)
    python_only_check = {
        "base": "origin/main",
        "changed_file_count": len(changed_files),
        "native_sensitive_files": sensitive_files,
        "passed": not sensitive_files,
    }
    if sensitive_files:
        print(json.dumps(python_only_check, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    venv_dir = repo_root / ".venv"
    venv_python = venv_dir / "bin" / "python"
    command_reports.append(
        _run(
            ["uv", "venv", str(venv_dir), "--python", "3.12", "--seed", "--clear"],
            cwd=repo_root,
            log_path=log_dir / "uv-venv.log",
        )
    )
    active_venv_env = _venv_env(venv_dir)
    command_reports.append(
        _run(
            [
                "uv",
                "sync",
                "--active",
                "--package",
                "marin-core",
                "--package",
                "marin-levanter",
                "--no-group",
                "dev",
                "--extra",
                "gpu",
            ],
            cwd=repo_root,
            env=active_venv_env,
            log_path=log_dir / "uv-sync-marin.log",
        )
    )

    uv_pip_env = _base_env()
    command_reports.append(
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "pytest",
                "pytest-timeout",
                "fsspec",
                "s3fs",
                "requests",
            ],
            cwd=repo_root,
            env=uv_pip_env,
            log_path=log_dir / "uv-pip-install-test-runtime.log",
        )
    )
    command_reports.append(
        _run(
            ["uv", "pip", "install", "--python", str(venv_python), "jax[cuda13]==0.10.1"],
            cwd=repo_root,
            env=uv_pip_env,
            log_path=log_dir / "uv-pip-install-jax-cuda.log",
        )
    )
    command_reports.append(
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--reinstall-package",
                "torch",
                "--reinstall-package",
                "torchvision",
                "--reinstall-package",
                "torchaudio",
                "--torch-backend=cu130",
                "torch==2.11.0",
                "torchvision==0.26.0",
                "torchaudio==2.11.0",
            ],
            cwd=vllm_dir,
            env=uv_pip_env,
            log_path=log_dir / "uv-pip-install-torch-cu130.log",
        )
    )
    vllm_install_env = _base_env({"VLLM_TARGET_DEVICE": VLLM_TARGET_DEVICE, "VLLM_USE_PRECOMPILED": "1", "VERBOSE": "1"})
    command_reports.append(
        _run(
            [
                "uv",
                "pip",
                "install",
                "-v",
                "--python",
                str(venv_python),
                "--editable",
                str(vllm_dir),
                "--torch-backend=auto",
            ],
            # Avoid Marin's root uv config, which intentionally builds the
            # normal pinned vLLM dependency with TPU-specific build variables.
            cwd=vllm_dir,
            env=vllm_install_env,
            log_path=log_dir / "uv-pip-install-editable-vllm.log",
        )
    )

    cuda_library_dirs = _python_library_dirs(venv_python=venv_python, cwd=repo_root)
    import_check_env = _venv_env(
        venv_dir,
        {
            "VLLM_TARGET_DEVICE": VLLM_TARGET_DEVICE,
            "VLLM_USE_PRECOMPILED": "1",
            "PYTHONPATH": os.pathsep.join([str(repo_root), os.environ.get("PYTHONPATH", "")]),
        },
    )
    _prepend_env_path(import_check_env, "LD_LIBRARY_PATH", cuda_library_dirs)

    install_report = _build_install_report(
        run_id=run_id,
        output_dir=output_dir,
        install_report_path=install_report_path,
        venv_python=venv_python,
        vllm_dir=vllm_dir,
        changed_files=changed_files,
        python_only_check=python_only_check,
        command_reports=command_reports,
        runtime_env=import_check_env,
        cuda_library_dirs=cuda_library_dirs,
    )
    _upload_validation_artifacts(
        output_dir=output_dir,
        install_report_path=install_report_path,
        install_report=install_report,
        local_report_path=local_report_path,
        log_dir=log_dir,
        venv_python=venv_python,
        repo_root=repo_root,
    )
    failed_imports = _failed_required_imports(install_report)
    if failed_imports:
        print(json.dumps({"failed_required_imports": failed_imports}, indent=2, sort_keys=True), file=sys.stderr)
        return 3

    pytest_env = dict(import_check_env)
    pytest_env.update(
        {
            RUN_ID_ENV: run_id,
            OUTPUT_DIR_ENV: output_dir,
            INSTALL_REPORT_PATH_ENV: install_report_path,
            ATTENTION_BACKEND_ENV: args.attention_backend,
            "IRIS_WORKER_REGION": COREWEAVE_REGION,
            "COREWEAVE_REGION": COREWEAVE_REGION,
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "MARIN_GIT_SHA": install_report["marin_sha"],
        }
    )
    pytest_report = _run(
        [
            str(venv_python),
            "-m",
            "pytest",
            "tests/vllm/test_grugmoe_gpu_real_checkpoint_e2e.py",
            "-q",
            "-s",
            "-o",
            "addopts=",
        ],
        cwd=repo_root,
        env=pytest_env,
        log_path=log_dir / "pytest-grugmoe-gpu-e2e.log",
        check=False,
    )
    command_reports.append(pytest_report)
    install_report["commands"] = command_reports
    _upload_validation_artifacts(
        output_dir=output_dir,
        install_report_path=install_report_path,
        install_report=install_report,
        local_report_path=local_report_path,
        log_dir=log_dir,
        venv_python=venv_python,
        repo_root=repo_root,
    )
    return int(pytest_report["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
