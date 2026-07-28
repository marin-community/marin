# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only evidence capture for a running distributed task."""

import json
from typing import Any

from rigging.timing import Timestamp

from iris.cluster.runtime.nccl_ras import NcclRasFormat, collective_count_skews, parse_json_response
from iris.cluster.runtime.profile import ExecResult, ProfileDispatch, capture_threads

MAX_DIAGNOSTIC_BYTES = 4 * 1024 * 1024
_SECTION_LIMIT = 256 * 1024
_ERROR_LIMIT = 2048
_PROCESS_WAITS_SCRIPT = r"""for p in /proc/"$1"/task/*; do
  t=${p##*/}
  printf '%s ' "$t"
  cat "$p/wchan"
  grep '^State:' "$p/status"
done"""
_RAS_SCRIPT = r"""import socket, sys
s = socket.create_connection(("127.0.0.1", 28028), timeout=float(sys.argv[2]))
s.settimeout(float(sys.argv[2]))
s.sendall(f"TIMEOUT {max(1, int(float(sys.argv[2]) + .999))}\nSET FORMAT {sys.argv[1]}\nVERBOSE STATUS\n".encode())
s.shutdown(socket.SHUT_WR)
out = []
while True:
    data = s.recv(65536)
    if not data: break
    out.append(data)
s.close()
sys.stdout.buffer.write(b"".join(out))
"""
_RUNTIME_VERSION_SCRIPT = r"""import importlib.metadata as metadata
names = ('jax', 'jaxlib', 'nvidia-nccl-cu12', 'nvidia-nccl-cu13', 'nvidia-cudnn-cu12', 'nvidia-cudnn-cu13')
for name in names:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
"""


def capture_distributed_diagnostic(
    dispatch: ProfileDispatch, *, pid: str, source: str, attempt_id: int | None, timeout: int = 5
) -> bytes:
    """Return a capped JSON evidence bundle, retaining successful partial probes."""
    if not 1 <= timeout <= 30:
        raise ValueError("distributed collector timeout must be between 1 and 30 seconds")
    errors: list[dict[str, str]] = []
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": Timestamp.now().as_naive_utc().isoformat(),
        "source": source,
        "attempt_id": attempt_id,
        "errors": errors,
    }
    bundle["nccl_ras"] = _capture_ras(dispatch, timeout, errors)
    bundle["threads"] = _capture_threads(dispatch, pid, errors)
    bundle["process_waits"] = _capture_command(
        dispatch,
        [
            "sh",
            "-c",
            _PROCESS_WAITS_SCRIPT,
            "sh",
            pid,
        ],
        timeout,
        "process_waits",
        errors,
    )
    bundle["gpu"] = _capture_command(
        dispatch,
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,driver_version,utilization.gpu,memory.used,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        timeout,
        "gpu",
        errors,
    )
    bundle["environment"] = _capture_command(
        dispatch,
        ["sh", "-c", "env | LC_ALL=C sort | grep -E '^(NCCL_|XLA_FLAGS=|CUDA_)' || true"],
        timeout,
        "environment",
        errors,
    )
    bundle["runtime"] = _capture_command(dispatch, ["python", "-c", _RUNTIME_VERSION_SCRIPT], timeout, "runtime", errors)
    return _encode_capped(bundle)


def _capture_ras(dispatch: ProfileDispatch, timeout: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    result = _run(dispatch, ["python", "-c", _RAS_SCRIPT, NcclRasFormat.JSON.value, str(timeout)], timeout)
    if result.returncode == 0:
        try:
            report = parse_json_response(result.stdout)
            return {
                "status": "ok",
                "response_format": "json",
                "raw_response": _text(result.stdout),
                "report": report,
                "collective_count_skews": [s.__dict__ for s in collective_count_skews(report)],
            }
        except ValueError:
            pass
    text = _run(dispatch, ["python", "-c", _RAS_SCRIPT, NcclRasFormat.TEXT.value, str(timeout)], timeout)
    if text.returncode == 0:
        return {
            "status": "ok",
            "response_format": "text",
            "raw_response": _text(text.stdout),
            "report": None,
            "collective_count_skews": [],
        }
    _error(errors, "nccl_ras", text.stderr or result.stderr)
    return {"status": "unavailable", "raw_response": _text(text.stdout or result.stdout)}


def _capture_threads(dispatch: ProfileDispatch, pid: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        return {"status": "ok", "format": "py-spy", "text": _text(capture_threads(dispatch, pid=pid))}
    except Exception as exc:
        _error(errors, "threads", str(exc))
        return {"status": "error"}


def _capture_command(
    dispatch: ProfileDispatch, command: list[str], timeout: int, name: str, errors: list[dict[str, str]]
) -> dict[str, Any]:
    result = _run(dispatch, command, timeout)
    if result.returncode == 0:
        return {"status": "ok", "text": _text(result.stdout)}
    _error(errors, name, result.stderr)
    return {"status": "unavailable", "text": _text(result.stdout)}


def _run(dispatch: ProfileDispatch, command: list[str], timeout: int) -> ExecResult:
    try:
        return dispatch.exec(command, timeout=timeout)
    except Exception as exc:
        return ExecResult(1, b"", str(exc))


def _text(value: bytes) -> str:
    return value.decode("utf-8", "replace")[:_SECTION_LIMIT]


def _error(errors: list[dict[str, str]], collector: str, message: str) -> None:
    errors.append({"collector": collector, "message": message[:_ERROR_LIMIT]})


def _encode_capped(bundle: dict[str, Any]) -> bytes:
    data = json.dumps(bundle, sort_keys=True, default=str).encode()
    if len(data) <= MAX_DIAGNOSTIC_BYTES:
        return data
    bundle["errors"].append({"collector": "bundle", "message": "payload truncated at 4 MiB"})
    for section in ("threads", "nccl_ras", "process_waits", "gpu", "environment", "runtime"):
        bundle[section] = {"status": "truncated"}
        data = json.dumps(bundle, sort_keys=True, default=str).encode()
        if len(data) <= MAX_DIAGNOSTIC_BYTES:
            return data
    raise ValueError("distributed diagnostic could not be capped")
