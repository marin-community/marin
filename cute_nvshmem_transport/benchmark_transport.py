# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from dataclasses import asdict, dataclass

from cute_nvshmem_transport.correctness_pull import run_pull_correctness
from cute_nvshmem_transport.correctness_push import run_push_correctness
from cute_nvshmem_transport.pull_kernels import PullOperation
from cute_nvshmem_transport.push_kernels import PushOperation

DEFAULT_PAYLOAD_BYTES = (16, 64, 256, 1024, 4096, 6144, 24576, 98304, 393216)


@dataclass(frozen=True)
class BenchmarkRow:
    protocol: str
    operation: str
    num_pes: int
    num_slots: int
    payload_bytes: int
    num_epochs: int
    repetitions: int
    median_seconds: float
    latency_us: float
    effective_bandwidth_gbps: float
    aggregate_bandwidth_gbps: float
    min_rank_bandwidth_gbps: float
    max_rank_bandwidth_gbps: float


def _epochs_for_payload(payload_bytes: int) -> int:
    if payload_bytes <= 256:
        return 10_000
    target_bytes = 64 * 1024 * 1024
    return max(128, min(4096, target_bytes // payload_bytes))


def _benchmark_push(
    operation: PushOperation,
    *,
    num_pes: int,
    num_slots: int,
    payload_bytes: int,
    num_epochs: int,
    repetitions: int,
) -> BenchmarkRow:
    results = run_push_correctness(num_pes, num_epochs, num_slots, operation, payload_bytes, repetitions)
    rank_times = [result.elapsed_seconds for result in results]
    rank_bandwidths = [payload_bytes * num_epochs / elapsed / 1e9 for elapsed in rank_times]
    median_seconds = max(rank_times)
    effective_bandwidth = payload_bytes * num_epochs / median_seconds / 1e9
    return BenchmarkRow(
        protocol="push",
        operation=operation,
        num_pes=num_pes,
        num_slots=num_slots,
        payload_bytes=payload_bytes,
        num_epochs=num_epochs,
        repetitions=repetitions,
        median_seconds=median_seconds,
        latency_us=median_seconds / num_epochs * 1e6,
        effective_bandwidth_gbps=effective_bandwidth,
        aggregate_bandwidth_gbps=effective_bandwidth * num_pes,
        min_rank_bandwidth_gbps=min(rank_bandwidths),
        max_rank_bandwidth_gbps=max(rank_bandwidths),
    )


def _benchmark_pull(
    operation: PullOperation,
    *,
    num_pes: int,
    num_slots: int,
    payload_bytes: int,
    num_epochs: int,
    repetitions: int,
) -> BenchmarkRow:
    results = run_pull_correctness(num_pes, num_epochs, num_slots, operation, payload_bytes, repetitions)
    rank_times = [result.elapsed_seconds for result in results]
    rank_bandwidths = [payload_bytes * num_epochs / elapsed / 1e9 for elapsed in rank_times]
    median_seconds = max(rank_times)
    effective_bandwidth = payload_bytes * num_epochs / median_seconds / 1e9
    return BenchmarkRow(
        protocol="pull",
        operation=operation,
        num_pes=num_pes,
        num_slots=num_slots,
        payload_bytes=payload_bytes,
        num_epochs=num_epochs,
        repetitions=repetitions,
        median_seconds=median_seconds,
        latency_us=median_seconds / num_epochs * 1e6,
        effective_bandwidth_gbps=effective_bandwidth,
        aggregate_bandwidth_gbps=effective_bandwidth * num_pes,
        min_rank_bandwidth_gbps=min(rank_bandwidths),
        max_rank_bandwidth_gbps=max(rank_bandwidths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pes", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, nargs="+", default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for payload_bytes in args.payload_bytes:
        num_epochs = args.epochs or _epochs_for_payload(payload_bytes)
        for operation in PushOperation:
            row = _benchmark_push(
                operation,
                num_pes=args.num_pes,
                num_slots=args.num_slots,
                payload_bytes=payload_bytes,
                num_epochs=num_epochs,
                repetitions=args.repetitions,
            )
            rows.append(row)
            print(json.dumps(asdict(row), sort_keys=True), flush=True)
        for operation in PullOperation:
            row = _benchmark_pull(
                operation,
                num_pes=args.num_pes,
                num_slots=args.num_slots,
                payload_bytes=payload_bytes,
                num_epochs=num_epochs,
                repetitions=args.repetitions,
            )
            rows.append(row)
            print(json.dumps(asdict(row), sort_keys=True), flush=True)

    if args.output:
        with open(args.output, "w") as output_file:
            json.dump([asdict(row) for row in rows], output_file, indent=2, sort_keys=True)
            output_file.write("\n")


if __name__ == "__main__":
    main()
