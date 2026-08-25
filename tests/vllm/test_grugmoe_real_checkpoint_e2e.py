# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-checkpoint GrugMoE e2e for TPU vLLM and Levanter/JAX.

Lower-level diagnostics for failures are linked from the PR body; this test keeps
Marin coverage focused on the trained checkpoint correctness path.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.test_utils import skip_if_no_tpu
from tests.vllm import grugmoe_real_checkpoint_backend as backend

TPU_LOCK_PATH = "/tmp/marin-tpu-e2e.lock"
BACKEND_MODULE = "tests.vllm.grugmoe_real_checkpoint_backend"


def _require_no_active_xdist(request: pytest.FixtureRequest) -> None:
    xdist_env = {key: value for key, value in os.environ.items() if key.startswith("PYTEST_XDIST_WORKER")}
    workerinput = getattr(request.config, "workerinput", None)
    numprocesses = getattr(request.config.option, "numprocesses", None)
    if xdist_env or workerinput is not None or numprocesses not in (None, 0, "0"):
        raise RuntimeError(
            "GrugMoE real-checkpoint e2e cannot run under active pytest-xdist parallelism. "
            f"Detected env={xdist_env!r}, workerinput={workerinput is not None}, numprocesses={numprocesses!r}. "
            "Run this test with -n 0 or without xdist."
        )


def _vfio_lsof_diagnostic() -> dict[str, Any]:
    vfio_dir = Path("/dev/vfio")
    diagnostic: dict[str, Any] = {"vfio_dir": str(vfio_dir), "checked": False, "holders": []}
    if not vfio_dir.exists():
        diagnostic["reason"] = "missing /dev/vfio"
        return diagnostic

    device_paths = [str(path) for path in vfio_dir.iterdir()]
    if not device_paths:
        diagnostic["reason"] = "empty /dev/vfio"
        return diagnostic

    lsof = shutil.which("lsof")
    if lsof is None:
        diagnostic["reason"] = "lsof not installed"
        return diagnostic

    result = subprocess.run([lsof, *device_paths], text=True, capture_output=True, check=False)
    diagnostic.update(
        {
            "checked": True,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    rows = [line for line in result.stdout.splitlines()[1:] if line.strip()]
    diagnostic["holders"] = rows
    if rows:
        raise RuntimeError(
            "Detected existing /dev/vfio users before a GrugMoE TPU-owning phase. "
            f"Release stale/manual TPU processes first. lsof rows: {rows[:20]!r}"
        )
    return diagnostic


@pytest.fixture(scope="module")
def no_active_xdist(request: pytest.FixtureRequest) -> None:
    _require_no_active_xdist(request)


@pytest.fixture(scope="module")
def tpu_lock(no_active_xdist: None):
    del no_active_xdist
    with open(TPU_LOCK_PATH, "w") as lock_file:
        try:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another Marin TPU e2e appears to hold {TPU_LOCK_PATH}") from exc
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(json.dumps({"pid": os.getpid(), "test": __name__, "started": time.time()}) + "\n")
            lock_file.flush()
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@pytest.fixture(scope="module")
def e2e_paths(tpu_lock: None) -> backend.E2EPaths:
    del tpu_lock
    backend._require_runtime_region()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_dir = backend._join_path(backend.OUTPUT_ROOT, run_id)
    paths = backend.E2EPaths(
        output_dir=output_dir,
        cache_dir=backend._join_path(backend.CACHE_ROOT, run_id),
        artifact_dir=backend._join_path(output_dir, "artifact"),
        export_result_path=backend._join_path(output_dir, "export-result.json"),
        vllm_result_path=backend._join_path(output_dir, "vllm-result.json"),
        levanter_result_path=backend._join_path(output_dir, "levanter-result.json"),
        summary_result_path=backend._join_path(output_dir, "result.json"),
    )
    backend._require_constants_are_europe_west4(paths)
    return paths


def _run_backend(phase: str, paths: backend.E2EPaths, result_path: str) -> dict[str, Any]:
    backend._require_europe_west4_path("result_path", result_path)
    command = [
        sys.executable,
        "-m",
        BACKEND_MODULE,
        "--backend",
        phase,
        "--checkpoint-path",
        backend.CHECKPOINT_PATH,
        "--tokenizer-path",
        backend.TOKENIZER_PATH,
        "--output-dir",
        paths.output_dir,
        "--artifact-dir",
        paths.artifact_dir,
        "--cache-dir",
        paths.cache_dir,
        "--result-path",
        result_path,
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MARIN_GIT_SHA", backend._git_sha())
    print("grugmoe_real_checkpoint_e2e_command=" + json.dumps(command), flush=True)
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode == 0:
        return backend._read_json(result_path)
    if backend._exists(result_path):
        result = backend._read_json(result_path)
        result["backend_returncode"] = completed.returncode
        return result
    completed.check_returncode()
    raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def export_result(e2e_paths: backend.E2EPaths) -> dict[str, Any]:
    diagnostic = _vfio_lsof_diagnostic()
    print("grugmoe_real_checkpoint_export_tpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("export", e2e_paths, e2e_paths.export_result_path)


@pytest.fixture(scope="module")
def vllm_result(e2e_paths: backend.E2EPaths, export_result: dict[str, Any]) -> dict[str, Any]:
    assert export_result["artifact_dir"] == e2e_paths.artifact_dir
    diagnostic = _vfio_lsof_diagnostic()
    print("grugmoe_real_checkpoint_vllm_tpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("vllm", e2e_paths, e2e_paths.vllm_result_path)


@pytest.fixture(scope="module")
def levanter_result(e2e_paths: backend.E2EPaths) -> dict[str, Any]:
    diagnostic = _vfio_lsof_diagnostic()
    print("grugmoe_real_checkpoint_levanter_tpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("levanter", e2e_paths, e2e_paths.levanter_result_path)


def _write_summary_update(
    e2e_paths: backend.E2EPaths,
    *,
    export_result: dict[str, Any] | None = None,
    backend_result: dict[str, Any] | None = None,
) -> None:
    if backend._exists(e2e_paths.summary_result_path):
        summary = backend._read_json(e2e_paths.summary_result_path)
    else:
        summary = {
            "checkpoint_path": backend.CHECKPOINT_PATH,
            "tokenizer_path": backend.TOKENIZER_PATH,
            "region": backend.REGION,
            "tpu_type": backend.TPU_TYPE,
            "prompt": backend.PROMPT,
            "expected_continuation": backend.EXPECTED_CONTINUATION,
            "result_paths": {
                "export": e2e_paths.export_result_path,
                "vllm": e2e_paths.vllm_result_path,
                "levanter": e2e_paths.levanter_result_path,
                "summary": e2e_paths.summary_result_path,
            },
            "runtime": backend._runtime_snapshot(include_grugmoe_spec=True),
            "backend_results": {},
            "caveat": (
                "This e2e validates real trained-checkpoint serving correctness through vLLM and Levanter/JAX. "
                "It does not validate TP, broad context windows, logprob parity, router replay, or performance."
            ),
        }

    if export_result is not None:
        summary["export_result"] = export_result
    if backend_result is not None:
        phase = str(backend_result["phase"])
        summary.setdefault("backend_results", {})[phase] = backend_result
        summary[f"actual_{phase}_output"] = backend_result.get("completion")

    backend_results = summary.get("backend_results", {})
    summary["completed_backend_phases"] = sorted(backend_results)
    summary["passed"] = all(backend_results.get(phase, {}).get("passed") is True for phase in ("vllm", "levanter"))
    backend._write_json(e2e_paths.summary_result_path, summary)
    print("grugmoe_real_checkpoint_e2e_result=" + json.dumps(summary, sort_keys=True), flush=True)


def test_grugmoe_real_checkpoint_e2e_static_preconditions() -> None:
    backend._require_constants_are_europe_west4()
    assert backend.REGION == "europe-west4"
    assert backend.TPU_TYPE == "v6e-4"
    assert backend.CHECKPOINT_PATH.startswith(backend.EUROPE_WEST4_GCS_PREFIX)
    assert backend.TOKENIZER_PATH.startswith(backend.EUROPE_WEST4_GCS_PREFIX)


@skip_if_no_tpu
@pytest.mark.slow
@pytest.mark.data_integration
def test_grugmoe_real_checkpoint_vllm_output(
    e2e_paths: backend.E2EPaths,
    export_result: dict[str, Any],
    vllm_result: dict[str, Any],
) -> None:
    _write_summary_update(e2e_paths, export_result=export_result, backend_result=vllm_result)
    assert vllm_result["phase"] == "vllm"
    assert vllm_result["completion"] == backend.EXPECTED_CONTINUATION
    assert vllm_result["passed"] is True


@skip_if_no_tpu
@pytest.mark.slow
@pytest.mark.data_integration
def test_grugmoe_real_checkpoint_levanter_output(
    e2e_paths: backend.E2EPaths,
    levanter_result: dict[str, Any],
) -> None:
    _write_summary_update(e2e_paths, backend_result=levanter_result)
    assert levanter_result["phase"] == "levanter"
    assert levanter_result["completion"] == backend.EXPECTED_CONTINUATION
    assert levanter_result["passed"] is True
