#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit Grug Muon benchmark jobs to Iris for multiple TPU topologies."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


DEFAULT_IRIS_CONFIG = "lib/iris/examples/marin.yaml"
DEFAULT_BENCHMARK_SCRIPT = "lib/levanter/scripts/bench/bench_grug_muon.py"
DEFAULT_TPU_TYPES = ("v5p-64", "v5p-32")
DEFAULT_ORTHOGONALIZATIONS = ("stack_batch_sharded", "vmap_replicated")
DEFAULT_JOB_PREFIX = "codex-grug-muon"
DEFAULT_LIBTPU_INIT_ARGS = "--xla_tpu_scoped_vmem_limit_kib=50000"
DEFAULT_CPU = 32
DEFAULT_MEMORY = "128GB"
DEFAULT_DISK = "50GB"

DEFAULT_VOCAB_SIZE = 128_256
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_NUM_LAYERS = 27
DEFAULT_NUM_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_EXPERTS = 64
DEFAULT_EXPERT_AXIS_SIZE = 4
DEFAULT_MODEL_AXIS_SIZE = 1
DEFAULT_NUM_EXPERTS_PER_TOKEN = 4
DEFAULT_ROUTED_EXPERT_WIDTH = 1024
DEFAULT_SHARED_EXPERT_WIDTH = 1024
DEFAULT_CAPACITY_FACTOR = 1.25
DEFAULT_STEPS = 3
DEFAULT_WARMUP_STEPS = 1


@dataclass(frozen=True)
class SubmissionResult:
    tpu_type: str
    orthogonalization: str
    job_name: str
    job_id: str | None
    exit_code: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_IRIS_CONFIG)
    parser.add_argument("--benchmark-script", default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument(
        "--tpu-type",
        dest="tpu_types",
        action="append",
        default=None,
        help="TPU type to submit. Repeatable. Defaults to v5p-64 and v5p-32.",
    )
    parser.add_argument(
        "--orthogonalization",
        dest="orthogonalizations",
        action="append",
        default=None,
        choices=DEFAULT_ORTHOGONALIZATIONS,
        help="Orthogonalization backend to submit. Repeatable. Defaults to both.",
    )
    parser.add_argument("--job-prefix", default=DEFAULT_JOB_PREFIX)
    parser.add_argument("--zone", default=None, help="Optional zone constraint passed through to Iris.")
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument("--libtpu-init-args", default=DEFAULT_LIBTPU_INIT_ARGS)
    parser.add_argument("--wait", action="store_true", help="Wait for each job instead of passing --no-wait.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--expert-axis-size", type=int, default=DEFAULT_EXPERT_AXIS_SIZE)
    parser.add_argument("--model-axis-size", type=int, default=DEFAULT_MODEL_AXIS_SIZE)
    parser.add_argument("--num-experts-per-token", type=int, default=DEFAULT_NUM_EXPERTS_PER_TOKEN)
    parser.add_argument("--routed-expert-width", type=int, default=DEFAULT_ROUTED_EXPERT_WIDTH)
    parser.add_argument("--shared-expert-width", type=int, default=DEFAULT_SHARED_EXPERT_WIDTH)
    parser.add_argument("--capacity-factor", type=float, default=DEFAULT_CAPACITY_FACTOR)
    return parser


def _job_id_from_stdout(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("/"):
            return stripped
    return None


def _benchmark_command(args: argparse.Namespace, orthogonalization: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        args.benchmark_script,
        "--orthogonalization",
        orthogonalization,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--vocab-size",
        str(args.vocab_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--num-experts",
        str(args.num_experts),
        "--expert-axis-size",
        str(args.expert_axis_size),
        "--model-axis-size",
        str(args.model_axis_size),
        "--num-experts-per-token",
        str(args.num_experts_per_token),
        "--routed-expert-width",
        str(args.routed_expert_width),
        "--shared-expert-width",
        str(args.shared_expert_width),
        "--capacity-factor",
        str(args.capacity_factor),
    ]


def _iris_command(
    args: argparse.Namespace, tpu_type: str, orthogonalization: str, suffix: str
) -> tuple[list[str], str]:
    job_name = f"{args.job_prefix}-{tpu_type}-{orthogonalization}-{suffix}"
    command = [
        "uv",
        "run",
        "iris",
        "--config",
        args.config,
        "job",
        "run",
        "--tpu",
        tpu_type,
        "--job-name",
        job_name,
        "--cpu",
        str(args.cpu),
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--extra",
        "tpu",
        "-e",
        "LIBTPU_INIT_ARGS",
        args.libtpu_init_args,
    ]
    if args.zone is not None:
        command.extend(["--zone", args.zone])
    if not args.wait:
        command.append("--no-wait")
    command.append("--")
    command.extend(_benchmark_command(args, orthogonalization))
    return command, job_name


def _run_submission(command: list[str], dry_run: bool) -> subprocess.CompletedProcess[str]:
    print(shlex.join(command))
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def main() -> int:
    args = build_parser().parse_args()
    tpu_types = tuple(args.tpu_types or DEFAULT_TPU_TYPES)
    orthogonalizations = tuple(args.orthogonalizations or DEFAULT_ORTHOGONALIZATIONS)
    suffix = f"l{args.num_layers:02d}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"

    results: list[SubmissionResult] = []
    for tpu_type in tpu_types:
        for orthogonalization in orthogonalizations:
            command, job_name = _iris_command(args, tpu_type, orthogonalization, suffix)
            completed = _run_submission(command, args.dry_run)
            results.append(
                SubmissionResult(
                    tpu_type=tpu_type,
                    orthogonalization=orthogonalization,
                    job_name=job_name,
                    job_id=_job_id_from_stdout(completed.stdout),
                    exit_code=completed.returncode,
                )
            )

    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0 if all(result.exit_code == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
