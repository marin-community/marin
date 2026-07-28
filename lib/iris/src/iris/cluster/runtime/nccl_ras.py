# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query NCCL's process-local RAS service and analyze collective counters."""

import argparse
import json
import math
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rigging.timing import Deadline, Timestamp

DEFAULT_NCCL_RAS_HOST = "localhost"
DEFAULT_NCCL_RAS_PORT = 28028
DEFAULT_NCCL_RAS_TIMEOUT = 5.0
_READ_SIZE = 64 * 1024


class NcclRasFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class NcclRasFormatError(ValueError):
    """The RAS service did not return the requested machine-readable payload."""


@dataclass(frozen=True)
class CollectiveCountSkew:
    """One collective whose completed-call count differs across communicator ranks."""

    communicator_hash: str
    collective: str
    minimum: int
    maximum: int
    lagging_ranks: tuple[int, ...]


@dataclass(frozen=True)
class NcclRasSnapshot:
    """One live RAS response, preserving the vendor payload for post-mortem use."""

    captured_at: str
    response_format: NcclRasFormat
    raw_response: str
    report: dict[str, Any] | None

    def record(self) -> dict[str, Any]:
        """Return a JSON-serializable capture envelope."""
        return {
            "captured_at": self.captured_at,
            "response_format": self.response_format.value,
            "raw_response": self.raw_response,
            "report": self.report,
            "collective_count_skews": [asdict(skew) for skew in collective_count_skews(self.report or {})],
        }


def query_nccl_ras(
    *,
    host: str,
    port: int,
    timeout: float,
    response_format: NcclRasFormat,
) -> bytes:
    """Return one verbose NCCL RAS response over its documented text protocol."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    ras_timeout = max(1, math.ceil(timeout))
    request = f"TIMEOUT {ras_timeout}\nSET FORMAT {response_format.value}\nVERBOSE STATUS\n".encode()
    chunks: list[bytes] = []
    deadline = Deadline.from_seconds(timeout)
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        while not deadline.expired():
            connection.settimeout(deadline.remaining_seconds())
            try:
                chunk = connection.recv(_READ_SIZE)
            except TimeoutError:
                if chunks:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def parse_json_response(response: bytes) -> dict[str, Any]:
    """Extract the JSON object after any ``OK`` replies to preceding commands."""
    text = response.decode("utf-8", "replace")
    object_start = text.find("{")
    if object_start < 0:
        raise NcclRasFormatError("NCCL RAS response did not contain a JSON object")
    try:
        report = json.loads(text[object_start:])
    except json.JSONDecodeError as exc:
        raise NcclRasFormatError(f"NCCL RAS returned malformed JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise NcclRasFormatError("NCCL RAS JSON response must be an object")
    return report


def capture_nccl_ras(
    *,
    host: str = DEFAULT_NCCL_RAS_HOST,
    port: int = DEFAULT_NCCL_RAS_PORT,
    timeout: float = DEFAULT_NCCL_RAS_TIMEOUT,
) -> NcclRasSnapshot:
    """Capture RAS JSON when supported, falling back to the human-readable report."""
    response = query_nccl_ras(
        host=host,
        port=port,
        timeout=timeout,
        response_format=NcclRasFormat.JSON,
    )
    try:
        report = parse_json_response(response)
    except NcclRasFormatError:
        response = query_nccl_ras(
            host=host,
            port=port,
            timeout=timeout,
            response_format=NcclRasFormat.TEXT,
        )
        return NcclRasSnapshot(
            captured_at=Timestamp.now().as_naive_utc().isoformat(),
            response_format=NcclRasFormat.TEXT,
            raw_response=response.decode("utf-8", "replace"),
            report=None,
        )
    return NcclRasSnapshot(
        captured_at=Timestamp.now().as_naive_utc().isoformat(),
        response_format=NcclRasFormat.JSON,
        raw_response=response.decode("utf-8", "replace"),
        report=report,
    )


def _int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NcclRasFormatError(f"NCCL RAS field {field!r} must be an integer")
    return value


def collective_count_skews(report: Mapping[str, Any]) -> list[CollectiveCountSkew]:
    """Return per-communicator collective-count differences from a RAS JSON report.

    A non-empty result is a localization clue, not a standalone hang verdict:
    ranks can be transiently one call apart while the job is making progress.
    """
    communicators = report.get("communicators", [])
    if not isinstance(communicators, list):
        raise NcclRasFormatError("NCCL RAS field 'communicators' must be a list")

    skews: list[CollectiveCountSkew] = []
    for communicator in communicators:
        if not isinstance(communicator, dict):
            raise NcclRasFormatError("NCCL RAS communicator must be an object")
        communicator_hash = str(communicator.get("hash", "unknown"))
        ranks = communicator.get("ranks", [])
        if not isinstance(ranks, list):
            raise NcclRasFormatError("NCCL RAS communicator field 'ranks' must be a list")

        counts_by_collective: dict[str, list[tuple[int, int]]] = {}
        for rank_record in ranks:
            if not isinstance(rank_record, dict):
                raise NcclRasFormatError("NCCL RAS rank must be an object")
            rank = _int(rank_record.get("rank"), "rank")
            collective_counts = rank_record.get("collective_counts", {})
            if not isinstance(collective_counts, dict):
                raise NcclRasFormatError("NCCL RAS field 'collective_counts' must be an object")
            for collective, raw_count in collective_counts.items():
                count = _int(raw_count, f"collective_counts.{collective}")
                counts_by_collective.setdefault(str(collective), []).append((rank, count))

        for collective, rank_counts in counts_by_collective.items():
            counts = [count for _, count in rank_counts]
            minimum = min(counts)
            maximum = max(counts)
            if minimum == maximum:
                continue
            skews.append(
                CollectiveCountSkew(
                    communicator_hash=communicator_hash,
                    collective=collective,
                    minimum=minimum,
                    maximum=maximum,
                    lagging_ranks=tuple(rank for rank, count in rank_counts if count == minimum),
                )
            )
    return skews


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture NCCL RAS status as a JSON envelope")
    parser.add_argument("--host", default=DEFAULT_NCCL_RAS_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_NCCL_RAS_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_NCCL_RAS_TIMEOUT)
    args = parser.parse_args()
    snapshot = capture_nccl_ras(host=args.host, port=args.port, timeout=args.timeout)
    print(json.dumps(snapshot.record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
