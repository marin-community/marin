# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty


@dataclass(frozen=True)
class GemmResult:
    seconds: float
    tflops_per_pe: float


@dataclass(frozen=True)
class TransportResult:
    seconds: float
    bandwidth_gbps_per_pe: float


@dataclass(frozen=True)
class OverlapResult:
    protocol: str
    operation: str
    num_pes: int
    payload_bytes: int
    num_epochs: int
    standalone_communication_seconds: float
    standalone_compute_seconds: float
    concurrent_communication_seconds: float
    concurrent_compute_seconds: float
    concurrent_wall_seconds: float
    standalone_compute_tflops_per_pe: float
    concurrent_compute_tflops_per_pe: float
    compute_throughput_degradation: float
    standalone_communication_gbps_per_pe: float
    concurrent_communication_gbps_per_pe: float
    communication_throughput_degradation: float


def _gemm_worker(
    rank: int,
    matrix_size: int,
    iterations: int,
    ready_dir: str | None,
    start_file: str | None,
    results: multiprocessing.Queue,
) -> None:
    try:
        import jax
        import jax.numpy as jnp

        device = jax.devices("gpu")[rank]
        with jax.default_device(device):
            lhs = jnp.ones((matrix_size, matrix_size), dtype=jnp.bfloat16)
            rhs = jnp.ones((matrix_size, matrix_size), dtype=jnp.bfloat16)
            gemm = jax.jit(lambda x, y: x @ y)
            gemm(lhs, rhs).block_until_ready()
            if ready_dir is not None:
                Path(ready_dir, f"compute-{rank}").touch()
                assert start_file is not None
                while not Path(start_file).exists():
                    time.sleep(0.01)
            start = time.perf_counter()
            output = None
            for _ in range(iterations):
                output = gemm(lhs, rhs)
            assert output is not None
            output.block_until_ready()
            seconds = time.perf_counter() - start
        operations = 2 * matrix_size**3 * iterations
        results.put((rank, seconds, operations / seconds / 1e12))
    except BaseException as error:
        results.put((rank, type(error).__name__, str(error)))


def _run_gemm(
    num_pes: int,
    matrix_size: int,
    iterations: int,
    ready_dir: str | None = None,
    start_file: str | None = None,
) -> GemmResult:
    multiprocessing.set_start_method("spawn", force=True)
    results: multiprocessing.Queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_gemm_worker,
            args=(rank, matrix_size, iterations, ready_dir, start_file, results),
        )
        for rank in range(num_pes)
    ]
    for process in processes:
        process.start()
    rank_results: list[tuple[int, float, float]] = []
    for _ in processes:
        try:
            result = results.get(timeout=900)
        except Empty as error:
            raise TimeoutError("GEMM worker did not return") from error
        if len(result) != 3 or not isinstance(result[1], float):
            raise RuntimeError(f"GEMM worker {result[0]} failed: {result[1]}: {result[2]}")
        rank_results.append(result)
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"GEMM worker {process.pid} exited with {process.exitcode}")
    return GemmResult(
        seconds=max(result[1] for result in rank_results),
        tflops_per_pe=min(result[2] for result in rank_results),
    )


def _transport_command(protocol: str) -> list[str]:
    module = "correctness_push" if protocol == "push" else "correctness_pull"
    return [sys.executable, "-m", f"cute_nvshmem_transport.{module}"]


def _transport_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "NVTP_NUM_PES": str(args.num_pes),
            "NVTP_NUM_EPOCHS": str(args.num_epochs),
            "NVTP_NUM_SLOTS": str(args.num_slots),
            "NVTP_PAYLOAD_BYTES": str(args.payload_bytes),
            "NVTP_REPETITIONS": "1",
            f"NVTP_{args.protocol.upper()}_OPERATION": args.operation,
        }
    )
    return environment


def _parse_transport(stdout: str, payload_bytes: int, num_epochs: int) -> TransportResult:
    rows = json.loads(stdout)
    seconds = max(float(row["elapsed_seconds"]) for row in rows)
    return TransportResult(
        seconds=seconds,
        bandwidth_gbps_per_pe=payload_bytes * num_epochs / seconds / 1e9,
    )


def _run_transport(args: argparse.Namespace) -> TransportResult:
    completed = subprocess.run(
        _transport_command(args.protocol),
        cwd=Path(__file__).resolve().parents[1],
        env=_transport_environment(args),
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_transport(completed.stdout, args.payload_bytes, args.num_epochs)


def _wait_for_markers(directory: Path, prefixes: tuple[str, ...], num_pes: int, timeout: float) -> None:
    expected = {f"{prefix}-{rank}" for prefix in prefixes for rank in range(num_pes)}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        present = {path.name for path in directory.iterdir()}
        if expected <= present:
            return
        time.sleep(0.1)
    missing = sorted(expected - {path.name for path in directory.iterdir()})
    raise TimeoutError(f"timed out waiting for overlap workers: {missing}")


def _run_concurrent(args: argparse.Namespace) -> tuple[TransportResult, GemmResult, float]:
    with tempfile.TemporaryDirectory(prefix="nvtp-overlap-") as temporary_directory:
        ready_dir = Path(temporary_directory)
        start_file = ready_dir / "start"
        transport_environment = _transport_environment(args)
        transport_environment.update({"NVTP_READY_DIR": str(ready_dir), "NVTP_START_FILE": str(start_file)})
        transport_process = subprocess.Popen(
            _transport_command(args.protocol),
            cwd=Path(__file__).resolve().parents[1],
            env=transport_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        compute_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cute_nvshmem_transport.benchmark_overlap",
                "--gemm-only",
                "--num-pes",
                str(args.num_pes),
                "--matrix-size",
                str(args.matrix_size),
                "--gemm-iterations",
                str(args.gemm_iterations),
                "--ready-dir",
                str(ready_dir),
                "--start-file",
                str(start_file),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_markers(ready_dir, ("transport", "compute"), args.num_pes, args.compile_timeout)
            start_file.touch()
            transport_stdout, transport_stderr = transport_process.communicate(timeout=args.run_timeout)
            compute_stdout, compute_stderr = compute_process.communicate(timeout=args.run_timeout)
        except BaseException:
            transport_process.kill()
            compute_process.kill()
            raise
        if transport_process.returncode:
            raise RuntimeError(f"transport overlap process failed:\n{transport_stderr}")
        if compute_process.returncode:
            raise RuntimeError(f"compute overlap process failed:\n{compute_stderr}")
        transport = _parse_transport(transport_stdout, args.payload_bytes, args.num_epochs)
        gemm = GemmResult(**json.loads(compute_stdout))
        return transport, gemm, max(transport.seconds, gemm.seconds)


def _degradation(concurrent: float, standalone: float) -> float:
    return 1.0 - concurrent / standalone


def _main_benchmark(args: argparse.Namespace) -> None:
    standalone_transport = _run_transport(args)
    standalone_gemm = _run_gemm(args.num_pes, args.matrix_size, args.gemm_iterations)
    concurrent_transport, concurrent_gemm, wall_seconds = _run_concurrent(args)
    result = OverlapResult(
        protocol=args.protocol,
        operation=args.operation,
        num_pes=args.num_pes,
        payload_bytes=args.payload_bytes,
        num_epochs=args.num_epochs,
        standalone_communication_seconds=standalone_transport.seconds,
        standalone_compute_seconds=standalone_gemm.seconds,
        concurrent_communication_seconds=concurrent_transport.seconds,
        concurrent_compute_seconds=concurrent_gemm.seconds,
        concurrent_wall_seconds=wall_seconds,
        standalone_compute_tflops_per_pe=standalone_gemm.tflops_per_pe,
        concurrent_compute_tflops_per_pe=concurrent_gemm.tflops_per_pe,
        compute_throughput_degradation=_degradation(concurrent_gemm.tflops_per_pe, standalone_gemm.tflops_per_pe),
        standalone_communication_gbps_per_pe=standalone_transport.bandwidth_gbps_per_pe,
        concurrent_communication_gbps_per_pe=concurrent_transport.bandwidth_gbps_per_pe,
        communication_throughput_degradation=_degradation(
            concurrent_transport.bandwidth_gbps_per_pe,
            standalone_transport.bandwidth_gbps_per_pe,
        ),
    )
    output = json.dumps(asdict(result), indent=2, sort_keys=True)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("push", "pull"), default="push")
    parser.add_argument("--operation", default="put_signal_nbi_warp_quiet")
    parser.add_argument("--num-pes", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=6144)
    parser.add_argument("--num-epochs", type=int, default=200_000)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--gemm-iterations", type=int, default=2_000)
    parser.add_argument("--compile-timeout", type=float, default=600)
    parser.add_argument("--run-timeout", type=float, default=900)
    parser.add_argument("--output")
    parser.add_argument("--gemm-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready-dir", help=argparse.SUPPRESS)
    parser.add_argument("--start-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gemm_only:
        result = _run_gemm(
            args.num_pes,
            args.matrix_size,
            args.gemm_iterations,
            args.ready_dir,
            args.start_file,
        )
        print(json.dumps(asdict(result), sort_keys=True))
        return
    _main_benchmark(args)


if __name__ == "__main__":
    main()
