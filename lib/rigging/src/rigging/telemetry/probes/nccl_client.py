# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded client for NCCL's documented RAS text protocol."""

import argparse
import json
import math
import os
import socket
import sys

from pydantic import ValidationError

from rigging.telemetry.probes import nccl_ras
from rigging.timing import Deadline

DEFAULT_NCCL_RAS_ADDRESS = "localhost:28028"
NCCL_RAS_ADDRESS_ENV = "NCCL_RAS_ADDR"
NCCL_RAS_ENABLE_ENV = "NCCL_RAS_ENABLE"
# The full response is parsed and reduced in this process. This private memory
# bound therefore does not loosen the shared subprocess output bound.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
TIMEOUT_EXIT_CODE = 2
UNAVAILABLE_EXIT_CODE = 3
INVALID_CONFIG_EXIT_CODE = 4
OUTPUT_LIMIT_EXIT_CODE = 5
INVALID_PAYLOAD_EXIT_CODE = 6
REDUCED_OUTPUT_LIMIT_EXIT_CODE = 7
_READ_SIZE = 64 * 1024


class ResponseTooLargeError(RuntimeError):
    """The RAS response exceeded the client's fixed memory bound."""

    def __init__(self, observed_bytes: int, limit_bytes: int) -> None:
        super().__init__(f"NCCL RAS response exceeded {limit_bytes} bytes after reading {observed_bytes} bytes")
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes


def _parse_address(address: str) -> tuple[str, int]:
    if address.startswith("["):
        closing_bracket = address.find("]")
        if closing_bracket < 0 or address[closing_bracket + 1 : closing_bracket + 2] != ":":
            raise ValueError("bracketed NCCL RAS address must include a port")
        host = address[1:closing_bracket]
        raw_port = address[closing_bracket + 2 :]
    else:
        host, separator, raw_port = address.rpartition(":")
        if not separator or ":" in host:
            raise ValueError("NCCL RAS address must be host:port; bracket IPv6 addresses")
    if not host or not raw_port:
        raise ValueError("NCCL RAS address must include a host and port")
    try:
        port = int(raw_port)
    except ValueError:
        raise ValueError("NCCL RAS port must be an integer") from None
    if not 1 <= port <= 65_535:
        raise ValueError("NCCL RAS port must be between 1 and 65535")
    return host, port


def query_nccl_ras(*, address: str, timeout: float, verbose: bool = True) -> bytes:
    """Return one JSON status response from NCCL's local RAS service."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("NCCL RAS timeout must be positive and finite")

    server_timeout = max(1, math.ceil(timeout))
    status = "VERBOSE STATUS" if verbose else "STATUS"
    request = f"TIMEOUT {server_timeout}\nSET FORMAT json\n{status}\n".encode()
    deadline = Deadline.from_seconds(timeout)
    chunks: list[bytes] = []
    response_size = 0
    with socket.create_connection(_parse_address(address), timeout=timeout) as connection:
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        while True:
            remaining = deadline.remaining_seconds()
            if remaining <= 0:
                raise TimeoutError("NCCL RAS query deadline exceeded")
            connection.settimeout(remaining)
            chunk = connection.recv(_READ_SIZE)
            if not chunk:
                return b"".join(chunks)
            response_size += len(chunk)
            if response_size > MAX_RESPONSE_BYTES:
                raise ResponseTooLargeError(response_size, MAX_RESPONSE_BYTES)
            chunks.append(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query and reduce NCCL RAS JSON")
    parser.add_argument("--address", default=os.environ.get(NCCL_RAS_ADDRESS_ENV, DEFAULT_NCCL_RAS_ADDRESS))
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--detail", type=nccl_ras.RasDetail, choices=tuple(nccl_ras.RasDetail), required=True)
    args = parser.parse_args()
    try:
        response = query_nccl_ras(
            address=args.address,
            timeout=args.timeout,
            verbose=args.detail is nccl_ras.RasDetail.STALL,
        )
    except TimeoutError as error:
        _write_failure("client_timeout", error)
        return TIMEOUT_EXIT_CODE
    except ResponseTooLargeError as error:
        _write_failure(
            "client_response_limit",
            error,
            observed_bytes=error.observed_bytes,
            limit_bytes=error.limit_bytes,
        )
        return OUTPUT_LIMIT_EXIT_CODE
    except OSError as error:
        _write_failure("unavailable", error)
        return UNAVAILABLE_EXIT_CODE
    except ValueError as error:
        _write_failure("invalid_client_config", error)
        return INVALID_CONFIG_EXIT_CODE
    try:
        report = nccl_ras.reduce_response(response, detail=args.detail)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, ValueError) as error:
        _write_failure("invalid_payload", error)
        return INVALID_PAYLOAD_EXIT_CODE
    try:
        output = nccl_ras.serialize_success(report)
    except ValueError as error:
        _write_failure("reduced_output_limit", error)
        return REDUCED_OUTPUT_LIMIT_EXIT_CODE
    sys.stdout.buffer.write(output)
    return 0


def _write_failure(
    failure_kind: str,
    error: Exception,
    *,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> None:
    sys.stdout.buffer.write(
        nccl_ras.serialize_failure(
            failure_kind,
            str(error),
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
