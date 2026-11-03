"""Dependency-light process failure receipts and live stack capture."""

from __future__ import annotations

import faulthandler
import os
import signal
import socket
import time
import traceback
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, TextIO

from marinskyrl.environment_contract import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    DebugMode,
    PYTHONFAULTHANDLER_ENV,
    ensure_debug_artifact_directories,
    safe_artifact_component,
    write_atomic_json,
    write_process_manifest,
)


class ProcessOutcomeKind(StrEnum):
    EXIT = "exit"
    SIGNAL = "signal"


@dataclass(frozen=True)
class ProcessOutcome:
    """Normalized subprocess outcome that preserves signal termination."""

    kind: ProcessOutcomeKind
    raw_returncode: int
    public_exit_code: int
    signal: int | None = None
    signal_name: str | None = None

    @classmethod
    def from_returncode(cls, returncode: int) -> "ProcessOutcome":
        if returncode >= 0:
            return cls(kind=ProcessOutcomeKind.EXIT, raw_returncode=returncode, public_exit_code=returncode)
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        return cls(
            kind=ProcessOutcomeKind.SIGNAL,
            raw_returncode=returncode,
            public_exit_code=128 + signal_number,
            signal=signal_number,
            signal_name=signal_name,
        )


def _receipt_path_and_identity(
    role: str,
    suffix: str,
    *,
    pid: int,
    environment: Mapping[str, str],
) -> tuple[Path, dict[str, object]] | None:
    artifact_root = environment.get(DEBUG_ARTIFACT_DIR_ENV)
    if not artifact_root:
        return None
    ensure_debug_artifact_directories(artifact_root)
    timestamp_ns = time.time_ns()
    hostname = socket.gethostname()
    path = (
        Path(artifact_root)
        / "outcomes"
        / (f"{safe_artifact_component(role)}.{hostname}.{pid}.{timestamp_ns}{suffix}.json")
    )
    return path, {
        "schema_version": 1,
        "role": role,
        "hostname": hostname,
        "pid": pid,
        "observed_at_ns": timestamp_ns,
    }


def write_process_outcome(
    role: str,
    returncode: int,
    *,
    pid: int | None = None,
    environment: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ProcessOutcome, Path | None]:
    """Write an atomic, secret-free outcome receipt when diagnostics are active."""
    outcome = ProcessOutcome.from_returncode(returncode)
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    process_id = os.getpid() if pid is None else pid
    receipt = _receipt_path_and_identity(role, "", pid=process_id, environment=values)
    if receipt is None:
        return outcome, None
    path, identity = receipt
    payload = {
        **identity,
        **asdict(outcome),
        "metadata": dict(metadata or {}),
    }
    write_atomic_json(path, payload)
    return outcome, path


def write_exception_receipt(
    role: str,
    error: BaseException,
    *,
    environment: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist the original Python exception before native runtime teardown can mask it."""
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    receipt = _receipt_path_and_identity(role, ".exception", pid=os.getpid(), environment=values)
    if receipt is None:
        return None
    path, identity = receipt
    payload = {
        **identity,
        "kind": "exception",
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "metadata": dict(metadata or {}),
    }
    write_atomic_json(path, payload)
    return path


_live_stack_files: dict[Path, TextIO] = {}


def enable_fatal_stack_capture(*, environment: Mapping[str, str] | None = None) -> bool:
    """Enable fatal-signal Python tracebacks in an already-running interpreter."""
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    if values.get(PYTHONFAULTHANDLER_ENV) != "1":
        return False
    if not faulthandler.is_enabled():
        faulthandler.enable()
    return True


def install_live_stack_capture(
    role: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Register ``SIGUSR2`` to append an all-thread dump in distributed debug mode."""
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    artifact_root = values.get(DEBUG_ARTIFACT_DIR_ENV)
    if values.get(DEBUG_MODE_ENV) != DebugMode.DISTRIBUTED or artifact_root is None:
        return None
    ensure_debug_artifact_directories(artifact_root)
    path = Path(artifact_root) / "stacks" / f"{safe_artifact_component(role)}.{socket.gethostname()}.{os.getpid()}.log"
    if path in _live_stack_files:
        return path
    output = path.open("a")
    _live_stack_files[path] = output
    # Keep capture operator-triggered. Periodic faulthandler dumps can terminate
    # native-heavy Ray/vLLM actors while they are actively serving requests.
    faulthandler.register(signal.SIGUSR2, file=output, all_threads=True, chain=False)
    return path


def initialize_process_diagnostics(
    role: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path | None]:
    """Enable the manager-projected diagnostics and write this process's manifest."""
    enable_fatal_stack_capture(environment=environment)
    manifest = write_process_manifest(role, environment=environment)
    stack_path = install_live_stack_capture(role, environment=environment)
    return manifest, stack_path
