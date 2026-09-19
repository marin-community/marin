# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent, resource-bounded ShellSim sessions backed by the pinned Rust bridge."""

import base64
import json
import math
import os
import selectors
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

SHELLSIM_REVISION = "5674a9492c35ffe390a0d23b49c0a340b12beb30"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class ShellSimError(RuntimeError):
    """The bridge rejected an operation or its process/protocol failed."""


class ShellSimTimeout(ShellSimError):
    """A bridge operation exceeded its wall deadline and the session was killed."""


@dataclass(frozen=True)
class ShellSimLimits:
    """Cumulative fuel/output and peak memory/disk budgets for one trial."""

    cpu: int = 10_000_000
    memory: int = 64 * 1024 * 1024
    disk: int = 64 * 1024 * 1024
    output: int = 4 * 1024 * 1024


@dataclass(frozen=True)
class ShellSimUsage:
    cpu_used: int
    memory_current: int
    memory_peak: int
    disk_current: int
    disk_peak: int
    output_bytes: int


@dataclass(frozen=True)
class ShellSimResult:
    stdout: str
    stderr: str
    return_code: int
    stop_reason: str | None
    usage: ShellSimUsage


class ShellSimSession:
    """Own a ShellSim process and its VFS until close or a transport failure.

    Calls are serialized, including pipe writes, so callers may use this from
    asynchronous adapters through ``asyncio.to_thread``. Shell commands never
    execute on the host. Fuel and output budgets do not reset between calls.
    Explicit ``cwd`` and ``env`` updates persist, as do shell state changes.
    """

    def __init__(
        self,
        bridge_path: str = "taskcompendium-shellsim",
        *,
        limits: ShellSimLimits = ShellSimLimits(),
        timeout: float = 30,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if any(value <= 0 or value >= 2**64 for value in asdict(limits).values()):
            raise ValueError("ShellSim limits must be positive unsigned 64-bit integers")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._closed = False
        self._process = subprocess.Popen(
            [bridge_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        try:
            result = self._request("init", limits=asdict(limits))
            if result.get("revision") != SHELLSIM_REVISION:
                raise ShellSimError("bridge uses a different ShellSim revision")
        except BaseException:
            self._abort()
            raise

    def _abort(self) -> None:
        self._closed = True
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.close()
        self._process.stdout.close()

    def _request(self, operation: str, *, timeout: float | None = None, **arguments: Any) -> dict[str, Any]:
        timeout = self.timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        payload = json.dumps({"op": operation, **arguments}, separators=(",", ":")).encode() + b"\n"
        if len(payload) > MAX_REQUEST_BYTES:
            raise ShellSimError("request exceeds byte limit")
        with self._lock:
            if self._closed:
                raise ShellSimError("session is closed")
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            deadline = time.monotonic() + timeout
            sent = 0
            received = bytearray()
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(self._process.stdin, selectors.EVENT_WRITE)
                    selector.register(self._process.stdout, selectors.EVENT_READ)
                    while b"\n" not in received:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ShellSimTimeout(f"{operation} exceeded {timeout} seconds")
                        for key, _ in selector.select(remaining):
                            if key.fileobj is self._process.stdin:
                                sent += os.write(key.fd, payload[sent : sent + 65536])
                                if sent == len(payload):
                                    selector.unregister(key.fileobj)
                            else:
                                chunk = os.read(key.fd, 65536)
                                if not chunk:
                                    raise ShellSimError("bridge exited before completing its response")
                                received.extend(chunk)
                                if len(received) > MAX_RESPONSE_BYTES:
                                    raise ShellSimError("response exceeds byte limit")
                response = json.loads(received)
                if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
                    raise ShellSimError("invalid bridge response")
                if response["ok"] and not isinstance(response.get("result"), dict):
                    raise ShellSimError("invalid bridge result")
            except (OSError, ValueError, ShellSimError) as error:
                self._abort()
                if isinstance(error, ShellSimError):
                    raise
                raise ShellSimError(f"bridge transport failed: {error}") from error
            if not response["ok"]:
                raise ShellSimError(str(response.get("error", "bridge rejected request")))
            return response["result"]

    def run(
        self,
        command: str,
        stdin: bytes = b"",
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellSimResult:
        """Execute a simulated shell action in the trial's persistent session.

        An explicit timeout overrides the session deadline for this action.
        Expiration kills the bridge and makes the session unusable.
        """
        result = self._request(
            "exec", timeout=timeout, command=command, stdin=base64.b64encode(stdin).decode(), cwd=cwd, env=env or {}
        )
        return ShellSimResult(
            stdout=base64.b64decode(result["stdout"], validate=True).decode(errors="replace"),
            stderr=base64.b64decode(result["stderr"], validate=True).decode(errors="replace"),
            return_code=result["return_code"],
            stop_reason=result["stop_reason"],
            usage=ShellSimUsage(**result["usage"]),
        )

    def read_file(self, path: str) -> bytes:
        """Read bytes from the VFS, following VFS symlinks only."""
        return base64.b64decode(self._request("read", path=path)["data"], validate=True)

    def write_file(self, path: str, data: bytes) -> None:
        """Write bytes to an existing VFS directory; enforce its disk quota."""
        self._request("write", path=path, data=base64.b64encode(data).decode())

    def mkdir(self, path: str) -> None:
        """Create a VFS directory and its parents."""
        self._request("mkdir", path=path)

    def list_dir(self, path: str) -> tuple[str, ...]:
        """Return names of immediate directory entries."""
        return tuple(self._request("list", path=path)["paths"])

    def walk(self, path: str) -> tuple[str, ...]:
        """Return sorted absolute VFS paths, including the requested root."""
        return tuple(self._request("walk", path=path)["paths"])

    def list_files(self, path: str) -> list[str]:
        """Return recursive absolute file paths for result downloads."""
        return [entry for entry in self.walk(path) if self.is_file(entry)]

    def is_dir(self, path: str) -> bool:
        return self._request("stat", path=path)["is_dir"]

    def is_file(self, path: str) -> bool:
        return self._request("stat", path=path)["is_file"]

    def close(self) -> None:
        """Release the child and all simulated state. Repeated close is harmless."""
        with self._lock:
            if not self._closed:
                self._abort()

    def __enter__(self) -> "ShellSimSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
