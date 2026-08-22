# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe an Iris task health endpoint for a Kubernetes exec probe."""

import argparse
import http.client
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HEALTH_PATH = "/healthz"
HEALTH_PORT_FILE_ENV = "IRIS_HEALTH_PORT_FILE"
HEALTH_FAILURE_COUNT_FILE_ENV = "IRIS_HEALTH_FAILURE_COUNT_FILE"
HEALTH_TERMINATION_FILE_ENV = "IRIS_HEALTH_TERMINATION_FILE"
MAX_RESPONSE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class ProbeResult:
    healthy: bool
    detail: str


def _read_port() -> int:
    path = Path(os.environ[HEALTH_PORT_FILE_ENV])
    port = int(path.read_text(encoding="utf-8").strip())
    if not 1 <= port <= 65535:
        raise ValueError(f"published port {port} is outside the valid range")
    return port


def probe_health(timeout: float) -> ProbeResult:
    """Send one health request without following redirects."""
    try:
        port = _read_port()
    except (FileNotFoundError, OSError, ValueError) as error:
        return ProbeResult(False, f"health port is not available: {error}")

    return probe_http_health(port, timeout)


def probe_http_health(port: int, timeout: float) -> ProbeResult:
    """Send one health request to a known local port."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", HEALTH_PATH, headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace").strip()
    except (OSError, http.client.HTTPException) as error:
        return ProbeResult(False, f"health request failed: {error}")
    finally:
        connection.close()

    if 200 <= response.status < 400:
        return ProbeResult(True, f"HTTP {response.status}")
    detail = f"health endpoint returned HTTP {response.status}"
    if body:
        detail = f"{detail}: {body}"
    return ProbeResult(False, detail)


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o644)


def _failure_count() -> int:
    path = Path(os.environ[HEALTH_FAILURE_COUNT_FILE_ENV])
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return 0


def _record_live_result(result: ProbeResult, failure_threshold: int) -> None:
    count_path = Path(os.environ[HEALTH_FAILURE_COUNT_FILE_ENV])
    if result.healthy:
        _write_atomic(count_path, "0\n")
        Path(os.environ[HEALTH_TERMINATION_FILE_ENV]).unlink(missing_ok=True)
        return

    count = _failure_count() + 1
    _write_atomic(count_path, f"{count}\n")
    if count >= failure_threshold:
        termination_path = Path(os.environ[HEALTH_TERMINATION_FILE_ENV])
        _write_atomic(termination_path, f"Task health check failed {count} consecutive times: {result.detail}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("startup", "live"), required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--failure-threshold", type=int, default=1)
    args = parser.parse_args(argv)

    result = probe_health(args.timeout)
    if args.phase == "live":
        _record_live_result(result, args.failure_threshold)
    if result.healthy:
        return 0
    print(result.detail, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
