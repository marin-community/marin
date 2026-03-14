#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a CoreWeave H100x8 JAX DeepEP custom-call smoke/benchmark pod."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from iris.cluster.config import load_config
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerPhase
from iris.rpc import cluster_pb2

DEFAULT_IMAGE = "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel"
DEFAULT_CONFIG = Path("lib/iris/examples/coreweave.yaml")
DEFAULT_DEEPEP_REF = "567632dd59810d77b3cc05553df953cc0f779799"
BENCH_PATH = Path("lib/levanter/scripts/bench/bench_moe_hillclimb.py")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_repo_ref() -> str:
    return subprocess.check_output(
        ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _entrypoint(argv: list[str]) -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv))
    return entrypoint


def _download_unpack_block(url: str, target_dir: str) -> str:
    return f"""
/opt/conda/bin/python - <<'PY'
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

archive_url = "{url}"
download_path = Path(tempfile.mkdtemp(dir="/tmp")) / "archive.tar.gz"
target_dir = Path("{target_dir}")

if target_dir.exists():
    if target_dir.is_dir():
        for child in target_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        target_dir.unlink()
else:
    target_dir.mkdir(parents=True)

urllib.request.urlretrieve(archive_url, download_path)
with tarfile.open(download_path, "r:gz") as archive:
    temp_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    archive.extractall(temp_dir)
    extracted_root = next(temp_dir.iterdir())
    for child in extracted_root.iterdir():
        shutil.move(str(child), target_dir / child.name)
print("UNPACKED", target_dir)
PY
"""


def _smoke_block(args: argparse.Namespace) -> str:
    if args.skip_smoke:
        return ""
    return f"""
echo RUNNING_FFI_SMOKE
.venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp

from levanter.kernels.deepep import deepep_get_dispatch_layout

NUM_TOKENS = {args.smoke_tokens}
TOPK = {args.smoke_topk}
NUM_EXPERTS = {args.experts}
NUM_RANKS = {args.smoke_num_ranks}

experts_per_rank = NUM_EXPERTS // NUM_RANKS
topk_idx = ((jnp.arange(NUM_TOKENS * TOPK, dtype=jnp.int64).reshape(NUM_TOKENS, TOPK) * 7) + 3) % NUM_EXPERTS

rank_counts, expert_counts, is_token_in_rank = jax.jit(
    lambda x: deepep_get_dispatch_layout(x, num_ranks=NUM_RANKS, num_experts=NUM_EXPERTS)
)(topk_idx)

expected_expert_counts = jnp.bincount(topk_idx.reshape(-1), length=NUM_EXPERTS).astype(jnp.int32)
expected_rank_counts = expected_expert_counts.reshape(NUM_RANKS, experts_per_rank).sum(axis=1)
expected_is_token_in_rank = (topk_idx[..., None] // experts_per_rank == jnp.arange(NUM_RANKS)).any(axis=1)

rank_counts, expert_counts, is_token_in_rank = jax.device_get((rank_counts, expert_counts, is_token_in_rank))
expected_expert_counts, expected_rank_counts, expected_is_token_in_rank = jax.device_get(
    (expected_expert_counts, expected_rank_counts, expected_is_token_in_rank)
)

assert jnp.array_equal(expert_counts, expected_expert_counts), (expert_counts, expected_expert_counts)
assert jnp.array_equal(rank_counts, expected_rank_counts), (rank_counts, expected_rank_counts)
assert jnp.array_equal(is_token_in_rank, expected_is_token_in_rank), (
    is_token_in_rank,
    expected_is_token_in_rank,
)
print("FFI_SMOKE_OK")
print("FFI_SMOKE_DEVICES", jax.devices())
print("FFI_SMOKE_RANK_COUNTS", rank_counts.tolist())
print("FFI_SMOKE_EXPERT_COUNT_SUM", int(expert_counts.sum()))
PY
"""


def _bench_block(args: argparse.Namespace) -> str:
    if args.skip_bench:
        return ""
    kernels = " ".join(args.kernels.split(","))
    distributions = " ".join(args.distributions.split(","))
    topk_values = " ".join(args.topk_list.split(","))
    return f"""
echo RUNNING_BENCH_MATRIX
for distribution in {distributions}; do
  for topk in {topk_values}; do
    for kernel in {kernels}; do
      echo "BENCH_START kernel=$kernel distribution=$distribution topk=$topk"
      .venv/bin/python {BENCH_PATH} \\
        --tokens {args.tokens} \\
        --hidden {args.hidden} \\
        --mlp-dim {args.mlp_dim} \\
        --experts {args.experts} \\
        --shared-expert-dim {args.shared_expert_dim} \\
        --topk "$topk" \\
        --distribution "$distribution" \\
        --bench-pass {args.bench_pass} \\
        --kernel "$kernel" \\
        --ep-list {args.ep_list} \\
        --warmup {args.warmup} \\
        --iters {args.iters}
      echo "BENCH_END kernel=$kernel distribution=$distribution topk=$topk"
    done
  done
done
"""


def _build_run_script(args: argparse.Namespace) -> str:
    repo_archive_url = f"https://codeload.github.com/marin-community/marin/tar.gz/{args.repo_ref}"
    deepep_archive_url = f"https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/{args.deepep_ref}"
    return f"""set -euxo pipefail
echo HOSTNAME=$(hostname)
echo PYTHON=$(/opt/conda/bin/python -c 'import sys; print(sys.executable)')
/opt/conda/bin/python --version
command -v nvcc
nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader
/opt/conda/bin/python -m pip install --no-cache-dir uv
if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git
fi
{_download_unpack_block(repo_archive_url, "/app")}
{_download_unpack_block(deepep_archive_url, "/tmp/DeepEP")}
cd /app
export DEEPEP_SRC_ROOT=/tmp/DeepEP
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
env | grep -E '^(DEEPEP_SRC_ROOT|JAX_ENABLE_X64|JAX_PLATFORMS)=' | sort
uv sync --quiet --frozen --link-mode symlink --python 3.11 --package levanter --no-group dev --extra gpu
.venv/bin/python - <<'PY'
import jax
import jaxlib
print("JAX_VERSION", jax.__version__)
print("JAXLIB_VERSION", jaxlib.__version__)
print("JAX_DEVICES", jax.devices())
PY
{_smoke_block(args)}
{_bench_block(args)}
"""


def _make_container_config(args: argparse.Namespace) -> ContainerConfig:
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=args.cpu * 1000,
        memory_bytes=args.memory_gib * 1024**3,
        disk_bytes=args.disk_gib * 1024**3,
    )
    resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="H100", count=args.gpus))
    run_script = _build_run_script(args)
    return ContainerConfig(
        image=args.image,
        entrypoint=_entrypoint(["bash", "-lc", run_script]),
        env={},
        workdir="/app",
        task_id=args.task_id,
        resources=resources,
        timeout_seconds=args.timeout_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--repo-ref", default=_current_repo_ref())
    parser.add_argument("--deepep-ref", default=DEFAULT_DEEPEP_REF)
    parser.add_argument("--task-id", default=f"deepep-jax-krt-bench-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=256)
    parser.add_argument("--disk-gib", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--smoke-tokens", type=int, default=256)
    parser.add_argument("--smoke-topk", type=int, default=2)
    parser.add_argument("--smoke-num-ranks", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--shared-expert-dim", type=int, default=2048)
    parser.add_argument("--topk-list", default="2,8")
    parser.add_argument("--distributions", default="random,runs")
    parser.add_argument(
        "--kernels",
        default="current,ragged_a2a,deepep_layout_ragged_a2a",
    )
    parser.add_argument("--bench-pass", choices=("forward", "forward_backward"), default="forward_backward")
    parser.add_argument("--ep-list", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    namespace = config.platform.coreweave.namespace or "iris"
    runtime = KubernetesRuntime(namespace=namespace)
    handle = runtime.create_container(_make_container_config(args))
    reader = None
    try:
        handle.run()
        reader = handle.log_reader()
        print(f"POD_NAME={handle.container_id}", flush=True)
        print(f"TASK_ID={args.task_id}", flush=True)
        print(f"REPO_REF={args.repo_ref}", flush=True)
        print(f"DEEPEP_REF={args.deepep_ref}", flush=True)
        deadline = time.monotonic() + args.timeout_seconds
        while True:
            if reader is not None:
                for line in reader.read():
                    print(line.data, flush=True)
            status = handle.status()
            if status.phase == ContainerPhase.STOPPED:
                print(f"EXIT_CODE={status.exit_code}", flush=True)
                return 0 if status.exit_code == 0 else int(status.exit_code or 1)
            if time.monotonic() > deadline:
                raise TimeoutError(f"pod {handle.container_id} did not finish in {args.timeout_seconds}s")
            time.sleep(args.poll_seconds)
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
