#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded, read-only distributed GPU diagnostics from one task."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

SCHEMA_VERSION = 1
MAX_BUNDLE_BYTES = 4 * 1024 * 1024 - 64 * 1024
DEFAULT_COLLECTOR_TIMEOUT_SECONDS = 10
MIN_COLLECTOR_TIMEOUT_SECONDS = 1
MAX_COLLECTOR_TIMEOUT_SECONDS = 30
DEFAULT_PY_SPY = "py-spy"
DEFAULT_NCCL_RAS = "ncclras"
DEFAULT_NCCL_RAS_HOST = "127.0.0.1"
DEFAULT_NCCL_RAS_PORT = 28028

_COMMAND_STDERR_BYTES = 16 * 1024
_ENVIRONMENT_BYTES = 128 * 1024
_ENVIRONMENT_SOURCE_BYTES = 1024 * 1024
_GPU_BYTES = 64 * 1024
_NCCL_RAS_BYTES = 384 * 1024
_PACKAGE_BYTES = 256 * 1024
_PROCESS_BYTES = 384 * 1024
_RUNTIME_MAP_BYTES = 512 * 1024
_THREAD_BYTES = 512 * 1024
_MAX_ERRORS = 32
_MAX_ERROR_BYTES = 2048
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
_RUNTIME_LIBRARY_COMPONENTS = (
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
_NVIDIA_SMI_FIELDS = (
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
class CollectorError:
    collector: str
    message: str


@dataclass(frozen=True)
class ThreadState:
    tid: int
    wchan: str
    status: str


@dataclass(frozen=True)
class ProcessEvidence:
    status: CollectorStatus
    pid: int
    command: str
    process_status: str
    threads: list[ThreadState]
    observed_thread_count: int
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class EnvironmentEvidence:
    status: CollectorStatus
    variables: dict[str, str]
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class FileIdentity:
    path: str
    realpath: str
    size_bytes: int | None
    mtime_ns: int | None
    error: str | None


@dataclass(frozen=True)
class PackageIdentity:
    name: str
    version: str


@dataclass(frozen=True)
class RuntimeEvidence:
    status: CollectorStatus
    target_executable: FileIdentity | None
    packages: list[PackageIdentity]
    loaded_libraries: list[FileIdentity]
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class NcclRasEvidence:
    status: CollectorStatus
    response_format: str | None
    json_text: str | None
    text: str | None
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class ThreadEvidence:
    status: CollectorStatus
    text: str
    stderr: str
    returncode: int
    timed_out: bool
    truncated: bool


@dataclass(frozen=True)
class GpuSnapshot:
    index: str
    uuid: str
    name: str
    driver_version: str
    utilization_gpu_percent: str
    utilization_memory_percent: str
    memory_used_mib: str
    memory_total_mib: str
    power_draw_watts: str
    power_limit_watts: str


@dataclass(frozen=True)
class GpuEvidence:
    status: CollectorStatus
    gpus: list[GpuSnapshot]
    stderr: str
    returncode: int
    timed_out: bool
    truncated: bool


@dataclass(frozen=True)
class DiagnosticBundle:
    schema_version: int
    captured_at: str
    source: str
    attempt_id: int | None
    process: ProcessEvidence
    environment: EnvironmentEvidence
    runtime: RuntimeEvidence
    nccl_ras: NcclRasEvidence
    threads: ThreadEvidence
    gpus: GpuEvidence
    errors: list[CollectorError]


@dataclass(frozen=True)
class _CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool


@dataclass(frozen=True)
class _ProcessEnvironment:
    variables: dict[str, str]
    virtual_env: str | None
    truncated: bool


@dataclass(frozen=True)
class _PackageCollection:
    packages: list[PackageIdentity]
    truncated: bool
    error: str | None


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _bounded_read(file, limit: int) -> tuple[bytes, bool]:
    file.seek(0)
    data = file.read(limit + 1)
    return data[:limit], len(data) > limit


def _run_command(command: list[str], *, deadline: float, stdout_limit: int) -> _CommandResult:
    remaining = _remaining(deadline)
    if remaining <= 0:
        return _CommandResult(124, b"", b"collector deadline expired", False, False, True)

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, start_new_session=True)
        except OSError as exc:
            return _CommandResult(127, b"", str(exc).encode(), False, False, False)

        timed_out = False
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()

        stdout, stdout_truncated = _bounded_read(stdout_file, stdout_limit)
        stderr, stderr_truncated = _bounded_read(stderr_file, _COMMAND_STDERR_BYTES)
        return _CommandResult(
            returncode,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            timed_out,
        )


def _truncate_text(text: str, byte_limit: int) -> tuple[str, bool]:
    encoded = text.encode()
    if len(encoded) <= byte_limit:
        return text, False
    marker = _TRUNCATION_MARKER.encode()
    prefix = encoded[: max(0, byte_limit - len(marker))].decode("utf-8", "ignore")
    return prefix + _TRUNCATION_MARKER, True


def _decode(data: bytes, *, truncated: bool = False) -> str:
    text = data.decode("utf-8", "replace")
    return text + _TRUNCATION_MARKER if truncated else text


def _error_text(result: _CommandResult) -> str:
    if result.timed_out:
        return "collector deadline expired"
    stderr = _decode(result.stderr, truncated=result.stderr_truncated).strip()
    return stderr or f"command exited {result.returncode}"


def _add_error(errors: list[CollectorError], collector: str, message: object) -> str:
    text, _ = _truncate_text(str(message), _MAX_ERROR_BYTES)
    if len(errors) < _MAX_ERRORS:
        errors.append(CollectorError(collector, text))
    return text


def _read_path(path: Path, limit: int) -> tuple[str, bool]:
    with path.open("rb") as file:
        data = file.read(limit + 1)
    return _decode(data[:limit]), len(data) > limit


def _collect_process(pid: int, errors: list[CollectorError]) -> ProcessEvidence:
    proc = Path("/proc") / str(pid)
    section_errors: list[str] = []
    truncated = False
    consumed = 0

    try:
        process_status, value_truncated = _read_path(proc / "status", min(64 * 1024, _PROCESS_BYTES))
        truncated |= value_truncated
        consumed += len(process_status.encode())
    except OSError as exc:
        process_status = ""
        section_errors.append(_add_error(errors, "process_status", exc))

    try:
        command, value_truncated = _read_path(proc / "cmdline", min(64 * 1024, _PROCESS_BYTES - consumed))
        command = command.replace("\0", " ").strip()
        truncated |= value_truncated
        consumed += len(command.encode())
    except OSError as exc:
        command = ""
        section_errors.append(_add_error(errors, "process_command", exc))

    try:
        task_paths = sorted((proc / "task").iterdir(), key=lambda path: int(path.name))
    except OSError as exc:
        message = _add_error(errors, "process_threads", exc)
        return ProcessEvidence(
            CollectorStatus.PARTIAL,
            pid,
            command,
            process_status,
            [],
            0,
            truncated,
            message,
        )

    threads: list[ThreadState] = []
    for task_path in task_paths:
        if consumed >= _PROCESS_BYTES:
            truncated = True
            break
        try:
            wchan, wchan_truncated = _read_path(task_path / "wchan", 4096)
            status_limit = min(32 * 1024, _PROCESS_BYTES - consumed)
            status, status_truncated = _read_path(task_path / "status", status_limit)
        except OSError as exc:
            section_errors.append(_add_error(errors, "process_threads", f"tid {task_path.name}: {exc}"))
            continue
        threads.append(ThreadState(int(task_path.name), wchan.strip(), status))
        consumed += len(wchan.encode()) + len(status.encode())
        truncated |= wchan_truncated or status_truncated

    status = CollectorStatus.PARTIAL if section_errors or truncated else CollectorStatus.OK
    return ProcessEvidence(
        status,
        pid,
        command,
        process_status,
        threads,
        len(task_paths),
        truncated,
        "; ".join(section_errors) or None,
    )


def _read_process_environment(pid: int) -> _ProcessEnvironment:
    path = Path("/proc") / str(pid) / "environ"
    with path.open("rb") as file:
        raw = file.read(_ENVIRONMENT_SOURCE_BYTES + 1)
    source_truncated = len(raw) > _ENVIRONMENT_SOURCE_BYTES
    raw = raw[:_ENVIRONMENT_SOURCE_BYTES]

    variables: dict[str, str] = {}
    virtual_env = None
    consumed = 0
    output_truncated = False
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        raw_name, raw_value = item.split(b"=", 1)
        name = raw_name.decode("utf-8", "replace")
        value = raw_value.decode("utf-8", "replace")
        if name == "VIRTUAL_ENV":
            virtual_env = value
        if not (name.startswith(("NCCL_", "XLA_", "CUDA_")) or name == "XLA_FLAGS"):
            continue
        remaining = _ENVIRONMENT_BYTES - consumed - len(name.encode())
        if remaining <= 0:
            output_truncated = True
            break
        value, value_truncated = _truncate_text(value, remaining)
        variables[name] = value
        consumed += len(name.encode()) + len(value.encode())
        output_truncated |= value_truncated

    return _ProcessEnvironment(dict(sorted(variables.items())), virtual_env, source_truncated or output_truncated)


def _collect_environment(environment: _ProcessEnvironment) -> EnvironmentEvidence:
    status = CollectorStatus.PARTIAL if environment.truncated else CollectorStatus.OK
    return EnvironmentEvidence(status, environment.variables, environment.truncated, None)


def _file_identity(path: Path, display_path: str | None = None) -> FileIdentity:
    try:
        stat = path.stat()
    except OSError as exc:
        return FileIdentity(
            display_path or str(path),
            str(path.resolve(strict=False)),
            None,
            None,
            str(exc),
        )
    return FileIdentity(
        display_path or str(path),
        str(path.resolve(strict=False)),
        stat.st_size,
        stat.st_mtime_ns,
        None,
    )


def _is_runtime_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    if normalized in _RUNTIME_PACKAGE_NAMES or normalized.startswith(_RUNTIME_PACKAGE_PREFIXES):
        return True
    return normalized.startswith("nvidia-") and any(component in normalized for component in _NVIDIA_PACKAGE_COMPONENTS)


def _target_python(pid: int, environment: _ProcessEnvironment) -> Path:
    if environment.virtual_env:
        virtual_env_python = Path("/proc") / str(pid) / "root" / environment.virtual_env.lstrip("/") / "bin" / "python"
        if virtual_env_python.exists():
            return virtual_env_python
    return Path("/proc") / str(pid) / "exe"


def _installed_runtime_packages(
    pid: int,
    environment: _ProcessEnvironment,
    deadline: float,
) -> _PackageCollection:
    result = _run_command(
        ["uv", "pip", "list", "--python", str(_target_python(pid, environment)), "--format", "json"],
        deadline=deadline,
        stdout_limit=_PACKAGE_BYTES,
    )
    if result.returncode != 0:
        return _PackageCollection(
            [],
            result.stdout_truncated or result.stderr_truncated,
            _error_text(result),
        )
    try:
        records = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _PackageCollection([], True, f"uv pip list returned invalid JSON: {exc}")

    packages = [
        PackageIdentity(str(record["name"]), str(record["version"]))
        for record in records
        if isinstance(record, dict)
        and "name" in record
        and "version" in record
        and _is_runtime_package(str(record["name"]))
    ]
    return _PackageCollection(
        packages,
        result.stdout_truncated or result.stderr_truncated,
        None,
    )


def _loaded_runtime_libraries(pid: int) -> tuple[list[FileIdentity], bool]:
    maps, truncated = _read_path(Path("/proc") / str(pid) / "maps", _RUNTIME_MAP_BYTES)
    paths: set[str] = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5].removesuffix(" (deleted)")
        if path.startswith("/") and any(component in path.lower() for component in _RUNTIME_LIBRARY_COMPONENTS):
            paths.add(path)
    libraries = [_file_identity(Path(path), path) for path in sorted(paths)]
    return libraries, truncated


def _collect_runtime(
    pid: int,
    environment: _ProcessEnvironment,
    deadline: float,
    errors: list[CollectorError],
) -> RuntimeEvidence:
    section_errors: list[str] = []
    executable_path = Path("/proc") / str(pid) / "exe"
    try:
        target_executable = _file_identity(executable_path, os.readlink(executable_path))
    except OSError as exc:
        target_executable = None
        section_errors.append(_add_error(errors, "runtime_executable", exc))

    packages = _installed_runtime_packages(pid, environment, deadline)
    if packages.error:
        section_errors.append(_add_error(errors, "runtime_packages", packages.error))

    try:
        loaded_libraries, libraries_truncated = _loaded_runtime_libraries(pid)
    except OSError as exc:
        loaded_libraries = []
        libraries_truncated = False
        section_errors.append(_add_error(errors, "runtime_libraries", exc))
    truncated = packages.truncated or libraries_truncated
    status = CollectorStatus.PARTIAL if section_errors or truncated else CollectorStatus.OK
    return RuntimeEvidence(
        status,
        target_executable,
        packages.packages,
        loaded_libraries,
        truncated,
        "; ".join(section_errors) or None,
    )


def _collect_nccl_ras(
    nccl_ras: str,
    host: str,
    port: int,
    deadline: float,
    errors: list[CollectorError],
) -> NcclRasEvidence:
    timeout = max(1, int(_remaining(deadline) + 0.999))
    command = [nccl_ras, "-h", host, "-p", str(port), "-v", "-t", str(timeout)]
    json_result = _run_command([*command, "-f", "json"], deadline=deadline, stdout_limit=_NCCL_RAS_BYTES)
    json_text = _decode(json_result.stdout, truncated=json_result.stdout_truncated) or None

    if json_result.returncode == 0 and json_text:
        try:
            json.loads(json_text)
        except json.JSONDecodeError as exc:
            json_error = f"ncclras returned invalid JSON: {exc}"
        else:
            truncated = json_result.stdout_truncated or json_result.stderr_truncated
            return NcclRasEvidence(
                CollectorStatus.PARTIAL if truncated else CollectorStatus.OK,
                "json",
                json_text,
                None,
                truncated,
                None,
            )
    else:
        json_error = _error_text(json_result)

    text_result = _run_command(command, deadline=deadline, stdout_limit=_NCCL_RAS_BYTES)
    text = _decode(text_result.stdout, truncated=text_result.stdout_truncated) or None
    errors_text = [json_error]
    if text_result.returncode != 0:
        errors_text.append(_error_text(text_result))
    message = _add_error(errors, "nccl_ras", "; ".join(errors_text))
    truncated = (
        json_result.stdout_truncated
        or json_result.stderr_truncated
        or text_result.stdout_truncated
        or text_result.stderr_truncated
    )
    if text_result.returncode == 0 and text:
        return NcclRasEvidence(CollectorStatus.PARTIAL, "text", json_text, text, truncated, message)
    status = CollectorStatus.PARTIAL if json_text or text else CollectorStatus.UNAVAILABLE
    return NcclRasEvidence(status, None, json_text, text, truncated, message)


def _collect_threads(pid: int, py_spy: str, deadline: float, errors: list[CollectorError]) -> ThreadEvidence:
    result = _run_command(
        [py_spy, "dump", "--pid", str(pid), "--subprocesses", "--nonblocking"],
        deadline=deadline,
        stdout_limit=_THREAD_BYTES,
    )
    text = _decode(result.stdout, truncated=result.stdout_truncated)
    stderr = _decode(result.stderr, truncated=result.stderr_truncated)
    truncated = result.stdout_truncated or result.stderr_truncated
    if result.returncode == 0:
        status = CollectorStatus.PARTIAL if truncated else CollectorStatus.OK
    elif text:
        status = CollectorStatus.PARTIAL
        _add_error(errors, "threads", _error_text(result))
    else:
        status = CollectorStatus.UNAVAILABLE
        _add_error(errors, "threads", _error_text(result))
    return ThreadEvidence(status, text, stderr, result.returncode, result.timed_out, truncated)


def _collect_gpus(deadline: float, errors: list[CollectorError]) -> GpuEvidence:
    query = ",".join(_NVIDIA_SMI_FIELDS)
    result = _run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        deadline=deadline,
        stdout_limit=_GPU_BYTES,
    )
    rows = [row for row in csv.reader(_decode(result.stdout).splitlines()) if row]
    gpus = [
        GpuSnapshot(
            index=row[0].strip(),
            uuid=row[1].strip(),
            name=row[2].strip(),
            driver_version=row[3].strip(),
            utilization_gpu_percent=row[4].strip(),
            utilization_memory_percent=row[5].strip(),
            memory_used_mib=row[6].strip(),
            memory_total_mib=row[7].strip(),
            power_draw_watts=row[8].strip(),
            power_limit_watts=row[9].strip(),
        )
        for row in rows
        if len(row) == len(_NVIDIA_SMI_FIELDS)
    ]
    truncated = result.stdout_truncated or result.stderr_truncated
    if result.returncode == 0:
        status = CollectorStatus.PARTIAL if truncated else CollectorStatus.OK
    elif gpus:
        status = CollectorStatus.PARTIAL
        _add_error(errors, "gpus", _error_text(result))
    else:
        status = CollectorStatus.UNAVAILABLE
        _add_error(errors, "gpus", _error_text(result))
    return GpuEvidence(
        status,
        gpus,
        _decode(result.stderr, truncated=result.stderr_truncated),
        result.returncode,
        result.timed_out,
        truncated,
    )


def collect_diagnostic(
    *,
    pid: int,
    source: str,
    attempt_id: int | None,
    captured_at: str,
    timeout: int,
    py_spy: str = DEFAULT_PY_SPY,
    nccl_ras: str = DEFAULT_NCCL_RAS,
    ras_host: str = DEFAULT_NCCL_RAS_HOST,
    ras_port: int = DEFAULT_NCCL_RAS_PORT,
) -> DiagnosticBundle:
    """Collect independent evidence from a running task without changing its state."""
    if not MIN_COLLECTOR_TIMEOUT_SECONDS <= timeout <= MAX_COLLECTOR_TIMEOUT_SECONDS:
        raise ValueError(
            f"collector timeout must be between {MIN_COLLECTOR_TIMEOUT_SECONDS} "
            f"and {MAX_COLLECTOR_TIMEOUT_SECONDS} seconds"
        )

    started = time.monotonic()
    deadline = started + timeout
    errors: list[CollectorError] = []
    process = _collect_process(pid, errors)
    try:
        environment_data = _read_process_environment(pid)
    except OSError as exc:
        environment_error = _add_error(errors, "environment", exc)
        environment_data = _ProcessEnvironment({}, None, False)
        environment = EnvironmentEvidence(CollectorStatus.UNAVAILABLE, {}, False, environment_error)
    else:
        environment = _collect_environment(environment_data)
    runtime = _collect_runtime(
        pid,
        environment_data,
        min(deadline, started + timeout * 0.25),
        errors,
    )
    nccl_ras_evidence = _collect_nccl_ras(
        nccl_ras,
        ras_host,
        ras_port,
        min(deadline, started + timeout * 0.55),
        errors,
    )
    threads = _collect_threads(
        pid,
        py_spy,
        min(deadline, started + timeout * 0.85),
        errors,
    )
    gpus = _collect_gpus(deadline, errors)
    return DiagnosticBundle(
        SCHEMA_VERSION,
        captured_at,
        source,
        attempt_id,
        process,
        environment,
        runtime,
        nccl_ras_evidence,
        threads,
        gpus,
        errors,
    )


def encode_bundle(bundle: DiagnosticBundle) -> bytes:
    """Encode a typed diagnostic bundle within the profile transport limit."""
    data = json.dumps(asdict(bundle), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError(f"diagnostic bundle exceeded {MAX_BUNDLE_BYTES} bytes")
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--attempt-id", type=int)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--py-spy", default=DEFAULT_PY_SPY)
    parser.add_argument("--nccl-ras", default=DEFAULT_NCCL_RAS)
    parser.add_argument("--ras-host", default=DEFAULT_NCCL_RAS_HOST)
    parser.add_argument("--ras-port", type=int, default=DEFAULT_NCCL_RAS_PORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = collect_diagnostic(
        pid=args.pid,
        source=args.source,
        attempt_id=args.attempt_id,
        captured_at=args.captured_at,
        timeout=args.timeout,
        py_spy=args.py_spy,
        nccl_ras=args.nccl_ras,
        ras_host=args.ras_host,
        ras_port=args.ras_port,
    )
    sys.stdout.buffer.write(encode_bundle(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
