# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone, bounded distributed GPU diagnostic probe.

Iris copies this file into a running task container and executes it once. Keep
the module stdlib-only: the target may use a bring-your-own image or an older
Marin bundle that does not contain the controller's Iris revision.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
COLLECTOR_VERSION = "iris-distributed-diagnostic-v1"
MAX_BUNDLE_BYTES = 4 * 1024 * 1024 - 64 * 1024
DEFAULT_COLLECTOR_TIMEOUT_SECONDS = 10
MIN_COLLECTOR_TIMEOUT_SECONDS = 1
MAX_COLLECTOR_TIMEOUT_SECONDS = 30
DEFAULT_PY_SPY = "py-spy"
DEFAULT_NCCL_RAS_HOST = "127.0.0.1"
DEFAULT_NCCL_RAS_PORT = 28028

_MAX_ERROR_CHARS = 2048
_MAX_ERRORS = 32
_NCCL_RAW_LIMIT = 384 * 1024
_NCCL_REPORT_LIMIT = 256 * 1024
_THREAD_STDOUT_LIMIT = 768 * 1024
_COMMAND_STDERR_LIMIT = 32 * 1024
_PROCESS_LIMIT = 640 * 1024
_ENVIRONMENT_LIMIT = 128 * 1024
_RUNTIME_PATH_LIMIT = 512 * 1024
_GPU_STDOUT_LIMIT = 64 * 1024
_TRUNCATION_MARKER = "\n... [truncated by Iris diagnostic probe]"

_RUNTIME_PACKAGE_NAMES = frozenset({"jax", "jaxlib", "numpy", "cuda-python", "nvidia-ml-py"})
_RUNTIME_PACKAGE_PREFIXES = ("jax-cuda",)
_NVIDIA_PACKAGE_COMPONENTS = (
    "nccl",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
)
_LIBRARY_COMPONENTS = (
    "libnccl",
    "libcuda",
    "libcudnn",
    "libcublas",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libnvjitlink",
    "jaxlib",
)
_GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
)


class CollectorStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class _CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool = False


@dataclass(frozen=True)
class _RasQueryResult:
    raw: bytes
    complete: bool
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class _StringLocation:
    parent: Any
    key: str | int
    value: str


def _bounded_read(file, limit: int) -> tuple[bytes, bool]:
    file.seek(0)
    data = file.read(limit + 1)
    return data[:limit], len(data) > limit


def _run_command(command: list[str], *, timeout: float, stdout_limit: int) -> _CommandResult:
    if timeout <= 0:
        return _CommandResult(124, b"", b"collector deadline expired", False, False, timed_out=True)

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except OSError as exc:
            return _CommandResult(127, b"", str(exc).encode(), False, False)

        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()

        stdout, stdout_truncated = _bounded_read(stdout_file, stdout_limit)
        stderr, stderr_truncated = _bounded_read(stderr_file, _COMMAND_STDERR_LIMIT)
        return _CommandResult(
            returncode,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            timed_out=timed_out,
        )


def _text(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def _add_error(errors: list[dict[str, str]], collector: str, message: object) -> None:
    if len(errors) >= _MAX_ERRORS:
        return
    errors.append({"collector": collector, "message": str(message)[:_MAX_ERROR_CHARS]})


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _query_nccl_ras(
    *,
    host: str,
    port: int,
    response_format: str,
    deadline: float,
) -> _RasQueryResult:
    """Return one bounded NCCL RAS response."""
    remaining = _remaining(deadline)
    if remaining <= 0:
        return _RasQueryResult(b"", False, False, "collector deadline expired")

    request = "\n".join(
        (
            f"TIMEOUT {max(1, math.ceil(remaining))}",
            f"SET FORMAT {response_format}",
            "VERBOSE STATUS",
            "",
        )
    ).encode()
    chunks: list[bytes] = []
    size = 0
    complete = False
    truncated = False
    error: str | None = None

    try:
        with socket.create_connection((host, port), timeout=remaining) as connection:
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            while size < _NCCL_RAW_LIMIT:
                remaining = _remaining(deadline)
                if remaining <= 0:
                    error = "response deadline expired"
                    break
                connection.settimeout(remaining)
                try:
                    chunk = connection.recv(min(64 * 1024, _NCCL_RAW_LIMIT - size + 1))
                except TimeoutError:
                    error = "response deadline expired"
                    break
                if not chunk:
                    complete = True
                    break
                if size + len(chunk) > _NCCL_RAW_LIMIT:
                    chunks.append(chunk[: _NCCL_RAW_LIMIT - size])
                    size = _NCCL_RAW_LIMIT
                    truncated = True
                    break
                chunks.append(chunk)
                size += len(chunk)
    except OSError as exc:
        error = str(exc)

    return _RasQueryResult(b"".join(chunks), complete, truncated, error)


def _parse_ras_json(raw: bytes) -> dict[str, Any]:
    text = _text(raw)
    start = text.find("{")
    if start < 0:
        raise ValueError("NCCL RAS response did not contain a JSON object")
    report, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(report, dict):
        raise ValueError("NCCL RAS JSON response must be an object")
    return report


def _bounded_report(report: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    encoded = json.dumps(report, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    if len(encoded) <= _NCCL_REPORT_LIMIT:
        return report, False
    return None, True


def _ras_response(
    raw: bytes,
    *,
    complete: bool,
    truncated: bool,
    error: str | None,
) -> dict[str, Any]:
    return {
        "raw_response": _text(raw),
        "complete": complete,
        "truncated": truncated,
        "error": error,
    }


def _collect_nccl_ras(
    *,
    host: str,
    port: int,
    deadline: float,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    remaining = _remaining(deadline)
    json_deadline = time.monotonic() + remaining * 0.65
    json_result = _query_nccl_ras(
        host=host,
        port=port,
        response_format="json",
        deadline=json_deadline,
    )
    json_response = _ras_response(
        json_result.raw,
        complete=json_result.complete,
        truncated=json_result.truncated,
        error=json_result.error,
    )

    try:
        report = _parse_ras_json(json_result.raw)
    except ValueError as exc:
        report = None
        json_response["parse_error"] = str(exc)
    if report is not None:
        bounded_report, report_truncated = _bounded_report(report)
        json_response["report"] = bounded_report
        json_response["report_truncated"] = report_truncated
        if json_result.error:
            _add_error(errors, "nccl_ras_json", json_result.error)
        return {
            "status": (
                CollectorStatus.PARTIAL
                if json_result.error or json_result.truncated or report_truncated
                else CollectorStatus.OK
            ),
            "response_format": "json",
            "json": json_response,
            "text": None,
        }

    text_result = _query_nccl_ras(
        host=host,
        port=port,
        response_format="text",
        deadline=deadline,
    )
    text_response = _ras_response(
        text_result.raw,
        complete=text_result.complete,
        truncated=text_result.truncated,
        error=text_result.error,
    )
    if json_result.error:
        _add_error(errors, "nccl_ras_json", json_result.error)
    if text_result.error:
        _add_error(errors, "nccl_ras_text", text_result.error)

    has_evidence = bool(json_result.raw or text_result.raw)
    return {
        "status": CollectorStatus.PARTIAL if has_evidence else CollectorStatus.UNAVAILABLE,
        "response_format": "text" if text_result.raw else None,
        "json": json_response,
        "text": text_response,
    }


def _resume_after_pyspy(
    pid: int,
    *,
    isolated_pid_namespace: bool,
    errors: list[dict[str, str]],
) -> None:
    """Clear a group-stop left by a failed ptrace attachment."""
    if sys.platform != "linux":
        return
    if not isolated_pid_namespace:
        try:
            os.killpg(os.getpgid(pid), signal.SIGCONT)
        except PermissionError as exc:
            _add_error(errors, "threads_resume", exc)
        except ProcessLookupError:
            pass
        return
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        _add_error(errors, "threads_resume", exc)
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            os.kill(int(entry.name), signal.SIGCONT)
        except PermissionError as exc:
            _add_error(errors, "threads_resume", f"pid {entry.name}: {exc}")
        except ProcessLookupError:
            pass


def _collect_threads(
    *,
    pid: int,
    py_spy: str,
    deadline: float,
    errors: list[dict[str, str]],
    isolated_pid_namespace: bool,
) -> dict[str, Any]:
    try:
        result = _run_command(
            [py_spy, "dump", "--pid", str(pid), "--subprocesses"],
            timeout=_remaining(deadline),
            stdout_limit=_THREAD_STDOUT_LIMIT,
        )
    finally:
        _resume_after_pyspy(pid, isolated_pid_namespace=isolated_pid_namespace, errors=errors)

    stderr = _text(result.stderr)
    partial_non_python_child = bool(result.stdout) and "Failed to find python version from target process" in stderr
    if result.returncode == 0:
        status = CollectorStatus.PARTIAL if result.stdout_truncated or result.stderr_truncated else CollectorStatus.OK
    elif partial_non_python_child:
        status = CollectorStatus.PARTIAL
        _add_error(errors, "threads", stderr)
    else:
        status = CollectorStatus.UNAVAILABLE
        _add_error(errors, "threads", stderr or f"py-spy exited {result.returncode}")
    if result.timed_out:
        status = CollectorStatus.PARTIAL if result.stdout else CollectorStatus.UNAVAILABLE
        _add_error(errors, "threads", "py-spy exceeded its collector deadline")

    return {
        "status": status,
        "format": "py-spy",
        "text": _text(result.stdout),
        "stderr": stderr,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "truncated": result.stdout_truncated or result.stderr_truncated,
    }


def _read_path(path: Path, limit: int) -> tuple[bytes, bool]:
    with path.open("rb") as file:
        data = file.read(limit + 1)
    return data[:limit], len(data) > limit


def _collect_process(pid: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    result: dict[str, Any] = {
        "status": CollectorStatus.OK,
        "pid": pid,
        "status_text": "",
        "command": "",
        "threads": [],
        "truncated": False,
    }
    consumed = 0

    try:
        status, truncated = _read_path(proc / "status", min(64 * 1024, _PROCESS_LIMIT))
        result["status_text"] = _text(status)
        result["truncated"] = truncated
        consumed += len(status)
    except OSError as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "process_status", exc)

    try:
        command, truncated = _read_path(proc / "cmdline", min(64 * 1024, _PROCESS_LIMIT - consumed))
        result["command"] = _text(command).replace("\0", " ").strip()
        result["truncated"] = result["truncated"] or truncated
        consumed += len(command)
    except OSError as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "process_command", exc)

    try:
        task_paths = sorted((proc / "task").iterdir(), key=lambda path: int(path.name))
    except OSError as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "process_threads", exc)
        return result

    result["observed_thread_count"] = len(task_paths)
    for task_path in task_paths:
        if consumed >= _PROCESS_LIMIT:
            result["truncated"] = True
            break
        try:
            wchan, wchan_truncated = _read_path(task_path / "wchan", 4096)
            thread_status, status_truncated = _read_path(
                task_path / "status",
                min(32 * 1024, _PROCESS_LIMIT - consumed),
            )
        except OSError as exc:
            _add_error(errors, "process_threads", f"tid {task_path.name}: {exc}")
            result["status"] = CollectorStatus.PARTIAL
            continue
        consumed += len(wchan) + len(thread_status)
        result["threads"].append(
            {
                "tid": int(task_path.name),
                "wchan": _text(wchan).strip(),
                "status_text": _text(thread_status),
            }
        )
        result["truncated"] = result["truncated"] or wchan_truncated or status_truncated

    result["included_thread_count"] = len(result["threads"])
    if result["truncated"] and result["status"] == CollectorStatus.OK:
        result["status"] = CollectorStatus.PARTIAL
    return result


def _read_process_environment(pid: int) -> tuple[dict[str, str], bool]:
    raw, truncated = _read_path(Path("/proc") / str(pid) / "environ", _ENVIRONMENT_LIMIT)
    environment: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        raw_name, raw_value = item.split(b"=", 1)
        name = _text(raw_name)
        if name.startswith(("NCCL_", "XLA_", "CUDA_")) or name == "XLA_FLAGS":
            environment[name] = _text(raw_value)
    return dict(sorted(environment.items())), truncated


def _collect_environment(pid: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        variables, truncated = _read_process_environment(pid)
        return {
            "status": CollectorStatus.PARTIAL if truncated else CollectorStatus.OK,
            "variables": variables,
            "truncated": truncated,
        }
    except OSError as exc:
        _add_error(errors, "environment", exc)
        return {"status": CollectorStatus.UNAVAILABLE, "variables": {}, "truncated": False}


def _is_runtime_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    if normalized in _RUNTIME_PACKAGE_NAMES or normalized.startswith(_RUNTIME_PACKAGE_PREFIXES):
        return True
    return normalized.startswith("nvidia-") and any(component in normalized for component in _NVIDIA_PACKAGE_COMPONENTS)


def _is_runtime_library(path: str) -> bool:
    lowered = path.lower()
    return any(component in lowered for component in _LIBRARY_COMPONENTS)


def _is_shared_library(path: str) -> bool:
    name = Path(path).name.lower()
    return ".so" in name or name.endswith((".dylib", ".dll"))


def _file_identity(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "realpath": str(path.resolve(strict=False))}
    try:
        stat = path.stat()
    except OSError as exc:
        record["error"] = str(exc)
        return record
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
    )
    return record


def _loaded_runtime_libraries(pid: int) -> tuple[list[dict[str, Any]], bool]:
    raw, truncated = _read_path(Path("/proc") / str(pid) / "maps", _RUNTIME_PATH_LIMIT)
    paths: set[str] = set()
    for line in _text(raw).splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if path.startswith("/") and _is_runtime_library(path):
            paths.add(path.removesuffix(" (deleted)"))
    return [_file_identity(Path(path)) for path in sorted(paths)], truncated


def _installed_runtime_packages() -> tuple[list[dict[str, Any]], bool]:
    packages: list[dict[str, Any]] = []
    consumed = 0
    truncated = False
    for distribution in sorted(
        importlib.metadata.distributions(),
        key=lambda dist: (dist.metadata.get("Name") or "").lower(),
    ):
        name = distribution.metadata.get("Name") or ""
        if not _is_runtime_package(name):
            continue
        record: dict[str, Any] = {
            "name": name,
            "version": distribution.version,
            "location": str(distribution.locate_file("")),
            "libraries": [],
        }
        for relative_path in distribution.files or ():
            path_text = str(relative_path)
            if _is_runtime_library(path_text) and _is_shared_library(path_text):
                identity = _file_identity(Path(distribution.locate_file(relative_path)))
                encoded_size = len(json.dumps(identity, default=str).encode())
                if consumed + encoded_size > _RUNTIME_PATH_LIMIT:
                    truncated = True
                    break
                record["libraries"].append(identity)
                consumed += encoded_size
        packages.append(record)
        if truncated:
            break
    return packages, truncated


def _collect_runtime(pid: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": CollectorStatus.OK,
        "probe_python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "target_executable": None,
        "loaded_libraries": [],
        "packages": [],
        "truncated": False,
    }
    try:
        target_executable = Path(os.readlink(Path("/proc") / str(pid) / "exe"))
        result["target_executable"] = _file_identity(target_executable)
    except OSError as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "runtime_executable", exc)

    try:
        libraries, truncated = _loaded_runtime_libraries(pid)
        result["loaded_libraries"] = libraries
        result["truncated"] = truncated
    except OSError as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "runtime_libraries", exc)

    try:
        packages, truncated = _installed_runtime_packages()
        result["packages"] = packages
        result["truncated"] = result["truncated"] or truncated
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        result["status"] = CollectorStatus.PARTIAL
        _add_error(errors, "runtime_packages", exc)

    if result["truncated"] and result["status"] == CollectorStatus.OK:
        result["status"] = CollectorStatus.PARTIAL
    return result


def _collect_gpus(deadline: float, errors: list[dict[str, str]]) -> dict[str, Any]:
    query = ",".join(_GPU_FIELDS)
    result = _run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout=_remaining(deadline),
        stdout_limit=_GPU_STDOUT_LIMIT,
    )
    stdout = _text(result.stdout)
    stderr = _text(result.stderr)
    rows = [
        {field: value.strip() for field, value in zip(_GPU_FIELDS, row, strict=False)}
        for row in csv.reader(stdout.splitlines())
        if row
    ]
    if result.returncode == 0:
        status = CollectorStatus.PARTIAL if result.stdout_truncated or result.stderr_truncated else CollectorStatus.OK
    else:
        status = CollectorStatus.PARTIAL if stdout else CollectorStatus.UNAVAILABLE
        _add_error(errors, "gpus", stderr or f"nvidia-smi exited {result.returncode}")
    if result.timed_out:
        _add_error(errors, "gpus", "nvidia-smi exceeded its collector deadline")
    return {
        "status": status,
        "fields": list(_GPU_FIELDS),
        "gpus": rows,
        "raw_csv": stdout,
        "stderr": stderr,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "truncated": result.stdout_truncated or result.stderr_truncated,
    }


def collect_diagnostic(
    *,
    pid: int,
    source: str,
    attempt_id: int | None,
    captured_at: str,
    timeout: int,
    py_spy: str = DEFAULT_PY_SPY,
    ras_host: str = DEFAULT_NCCL_RAS_HOST,
    ras_port: int = DEFAULT_NCCL_RAS_PORT,
    isolated_pid_namespace: bool = False,
) -> dict[str, Any]:
    """Collect one partial-result diagnostic bundle for a running task process."""
    if not MIN_COLLECTOR_TIMEOUT_SECONDS <= timeout <= MAX_COLLECTOR_TIMEOUT_SECONDS:
        raise ValueError(
            f"collector timeout must be between {MIN_COLLECTOR_TIMEOUT_SECONDS} "
            f"and {MAX_COLLECTOR_TIMEOUT_SECONDS} seconds"
        )

    started = time.monotonic()
    collector_deadline = started + timeout
    errors: list[dict[str, str]] = []
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "captured_at": captured_at,
        "source": source,
        "attempt_id": attempt_id,
        "errors": errors,
    }

    for name, collector in (
        ("process", lambda: _collect_process(pid, errors)),
        ("environment", lambda: _collect_environment(pid, errors)),
        ("runtime", lambda: _collect_runtime(pid, errors)),
    ):
        try:
            bundle[name] = collector()
        except Exception as exc:
            _add_error(errors, name, exc)
            bundle[name] = {"status": CollectorStatus.UNAVAILABLE}

    remaining = _remaining(collector_deadline)
    ras_deadline = time.monotonic() + remaining * 0.45
    threads_deadline = time.monotonic() + remaining * 0.80
    for name, collector in (
        (
            "nccl_ras",
            lambda: _collect_nccl_ras(
                host=ras_host,
                port=ras_port,
                deadline=ras_deadline,
                errors=errors,
            ),
        ),
        (
            "threads",
            lambda: _collect_threads(
                pid=pid,
                py_spy=py_spy,
                deadline=threads_deadline,
                errors=errors,
                isolated_pid_namespace=isolated_pid_namespace,
            ),
        ),
        ("gpus", lambda: _collect_gpus(collector_deadline, errors)),
    ):
        try:
            bundle[name] = collector()
        except Exception as exc:
            _add_error(errors, name, exc)
            bundle[name] = {"status": CollectorStatus.UNAVAILABLE}

    return bundle


def _truncate_utf8(text: str, byte_limit: int) -> str:
    if byte_limit <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[-byte_limit:] if byte_limit > 0 else ""
    prefix = text.encode("utf-8")[: byte_limit - len(_TRUNCATION_MARKER)].decode("utf-8", "ignore")
    return prefix + _TRUNCATION_MARKER


def _string_locations(value: Any) -> list[_StringLocation]:
    locations: list[_StringLocation] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str):
                locations.append(_StringLocation(value, key, child))
            else:
                locations.extend(_string_locations(child))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                locations.append(_StringLocation(value, index, child))
            else:
                locations.extend(_string_locations(child))
    return locations


def _json_bytes(bundle: dict[str, Any]) -> bytes:
    return json.dumps(
        bundle,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode()


def encode_bundle(bundle: dict[str, Any]) -> bytes:
    """Encode a diagnostic bundle below the Iris profile transport limit."""
    data = _json_bytes(bundle)
    if len(data) <= MAX_BUNDLE_BYTES:
        return data

    bundle["truncated"] = True
    errors = bundle.setdefault("errors", [])
    _add_error(errors, "bundle", f"payload exceeded {MAX_BUNDLE_BYTES} bytes and was truncated")
    while True:
        data = _json_bytes(bundle)
        if len(data) <= MAX_BUNDLE_BYTES:
            return data
        strings = _string_locations(bundle)
        if not strings:
            raise ValueError("diagnostic bundle contains no trimmable text")
        location = max(strings, key=lambda item: len(item.value.encode("utf-8")))
        value_size = len(location.value.encode("utf-8"))
        if value_size == 0:
            raise ValueError("diagnostic bundle could not be capped")
        overage = len(data) - MAX_BUNDLE_BYTES
        location.parent[location.key] = _truncate_utf8(
            location.value,
            max(0, value_size - overage - 1024),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--attempt-id", type=int)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--py-spy", default=DEFAULT_PY_SPY)
    parser.add_argument("--ras-host", default=DEFAULT_NCCL_RAS_HOST)
    parser.add_argument("--ras-port", type=int, default=DEFAULT_NCCL_RAS_PORT)
    parser.add_argument("--isolated-pid-namespace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = collect_diagnostic(
            pid=args.pid,
            source=args.source,
            attempt_id=args.attempt_id,
            captured_at=args.captured_at,
            timeout=args.timeout,
            py_spy=args.py_spy,
            ras_host=args.ras_host,
            ras_port=args.ras_port,
            isolated_pid_namespace=args.isolated_pid_namespace,
        )
    except Exception as exc:
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "captured_at": args.captured_at,
            "source": args.source,
            "attempt_id": args.attempt_id,
            "process": {"status": CollectorStatus.UNAVAILABLE},
            "environment": {"status": CollectorStatus.UNAVAILABLE},
            "runtime": {"status": CollectorStatus.UNAVAILABLE},
            "nccl_ras": {"status": CollectorStatus.UNAVAILABLE},
            "threads": {"status": CollectorStatus.UNAVAILABLE},
            "gpus": {"status": CollectorStatus.UNAVAILABLE},
            "errors": [{"collector": "probe", "message": str(exc)[:_MAX_ERROR_CHARS]}],
        }
    sys.stdout.buffer.write(encode_bundle(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
