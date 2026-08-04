# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded client for NCCL's documented RAS text protocol."""

import argparse
import math
import os
import socket
import sys

from rigging.timing import Deadline

DEFAULT_NCCL_RAS_ADDRESS = "localhost:28028"
MAX_RESPONSE_BYTES = 256 * 1024
TIMEOUT_EXIT_CODE = 2
UNAVAILABLE_EXIT_CODE = 3
INVALID_CONFIG_EXIT_CODE = 4
OUTPUT_LIMIT_EXIT_CODE = 5
_READ_SIZE = 64 * 1024


class ResponseTooLargeError(RuntimeError):
    """The RAS response exceeded the client's fixed memory bound."""


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


def query_nccl_ras(*, address: str, timeout: float) -> bytes:
    """Return one verbose JSON response from NCCL's local RAS service."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("NCCL RAS timeout must be positive and finite")

    server_timeout = max(1, math.ceil(timeout))
    request = f"TIMEOUT {server_timeout}\nSET FORMAT json\nVERBOSE STATUS\n".encode()
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
                raise ResponseTooLargeError(f"NCCL RAS response exceeded {MAX_RESPONSE_BYTES} bytes")
            chunks.append(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query NCCL RAS as JSON")
    parser.add_argument("--address", default=os.environ.get("NCCL_RAS_ADDR", DEFAULT_NCCL_RAS_ADDRESS))
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    try:
        response = query_nccl_ras(address=args.address, timeout=args.timeout)
    except TimeoutError as error:
        print(error, file=sys.stderr)
        return TIMEOUT_EXIT_CODE
    except ResponseTooLargeError as error:
        print(error, file=sys.stderr)
        return OUTPUT_LIMIT_EXIT_CODE
    except OSError as error:
        print(error, file=sys.stderr)
        return UNAVAILABLE_EXIT_CODE
    except ValueError as error:
        print(error, file=sys.stderr)
        return INVALID_CONFIG_EXIT_CODE
    sys.stdout.buffer.write(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
