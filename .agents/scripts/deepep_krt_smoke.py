#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a direct CoreWeave KubernetesRuntime DeepEP smoke/bench pod."""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

from iris.cluster.config import load_config
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerPhase
from iris.rpc import cluster_pb2

DEFAULT_IMAGE = "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel"
DEFAULT_WORKTREE = Path("/Users/romain/marin-wt/moe-deepep-hybrid-ep")
BENCH_PATH = Path("lib/levanter/scripts/bench/bench_deepep_torch.py")


def _entrypoint(argv: list[str], *, workdir_files: dict[str, bytes]) -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv))
    entrypoint.workdir_files.update(workdir_files)
    return entrypoint


def _build_run_script(args: argparse.Namespace, *, bench_file_b64: str) -> str:
    bench_cmd = ""
    if args.run_bench:
        bench_cmd = f"""
echo RUNNING_BENCH
torchrun --standalone --nproc_per_node={args.gpus} /app/{BENCH_PATH} \\
  --tokens-per-rank {args.tokens_per_rank} \\
  --hidden {args.hidden} \\
  --topk {args.topk} \\
  --distribution {args.distribution} \\
  --num-local-experts {args.num_local_experts} \\
  --kernel {args.kernel} \\
  --warmup {args.warmup} \\
  --iters {args.iters}
"""

    return f"""set -euxo pipefail
echo HOSTNAME=$(hostname)
echo PYTHON=$(/opt/conda/bin/python -c 'import sys; print(sys.executable)')
/opt/conda/bin/python --version
mkdir -p /app/{BENCH_PATH.parent}
/opt/conda/bin/python - <<'PY'
from pathlib import Path
import base64

target = Path("/app/{BENCH_PATH}")
target.write_bytes(base64.b64decode("{bench_file_b64}"))
print("WROTE_BENCH", target)
PY
ls -al /app/{BENCH_PATH.parent}
command -v nvcc
nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader
/opt/conda/bin/python -m pip install --no-cache-dir setuptools wheel ninja packaging
mkdir -p /tmp/no_nvshmem/nvidia
: > /tmp/no_nvshmem/nvidia/__init__.py
ORIGINAL_PYTHONPATH="${{PYTHONPATH:-}}"
export PYTHONPATH="/tmp/no_nvshmem${{ORIGINAL_PYTHONPATH:+:$ORIGINAL_PYTHONPATH}}"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
/opt/conda/bin/python - <<'PY'
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

archive_url = "https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/refs/heads/hybrid-ep"
download_path = Path("/tmp/DeepEP-hybrid-ep.tar.gz")
target_dir = Path("/tmp/DeepEP")

if target_dir.exists():
    shutil.rmtree(target_dir)
if download_path.exists():
    download_path.unlink()

urllib.request.urlretrieve(archive_url, download_path)
with tarfile.open(download_path, "r:gz") as archive:
    temp_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    archive.extractall(temp_dir)
    extracted_root = next(temp_dir.iterdir())
    extracted_root.rename(target_dir)
PY
NVTX_LIB_DIR=$(/opt/conda/bin/python - <<'PY'
from pathlib import Path
import sys

base = Path(sys.executable).resolve().parent.parent
candidate = (
    base
    / "lib"
    / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}"
    / "site-packages"
    / "nvidia"
    / "nvtx"
    / "lib"
)
if not candidate.is_dir():
    raise SystemExit(f"missing NVTX lib dir: {{candidate}}")
print(candidate)
PY
)
echo NVTX_LIB_DIR="$NVTX_LIB_DIR"
ls -al "$NVTX_LIB_DIR"
mkdir -p /tmp/nvtxshim
ln -sf "$NVTX_LIB_DIR/libnvToolsExt.so.1" /tmp/nvtxshim/libnvtx3interop.so
export LIBRARY_PATH="/tmp/nvtxshim:$NVTX_LIB_DIR${{LIBRARY_PATH:+:$LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/tmp/nvtxshim:$NVTX_LIB_DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
/opt/conda/bin/python - <<'PY'
import importlib.util
import torch

print("NVSHMEM_SPEC", importlib.util.find_spec("nvidia.nvshmem"))
print("TORCH_OK", torch.__version__, torch.version.cuda)
PY
/opt/conda/bin/python -m pip install --no-build-isolation /tmp/DeepEP
if [ -n "$ORIGINAL_PYTHONPATH" ]; then
  export PYTHONPATH="$ORIGINAL_PYTHONPATH"
else
  unset PYTHONPATH
fi
/opt/conda/bin/python - <<'PY'
import deep_ep

print("IMPORT_OK", deep_ep.__file__)
print("HAS_HYBRID", hasattr(deep_ep, "HybridEPBuffer"))
PY
/opt/conda/bin/python /app/{BENCH_PATH} --help
{bench_cmd}
"""


def _make_container_config(args: argparse.Namespace) -> ContainerConfig:
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=args.cpu * 1000,
        memory_bytes=args.memory_gib * 1024**3,
        disk_bytes=args.disk_gib * 1024**3,
    )
    resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="H100", count=args.gpus))
    bench_file_b64 = base64.b64encode((args.worktree / BENCH_PATH).read_bytes()).decode("ascii")
    run_script = _build_run_script(args, bench_file_b64=bench_file_b64)
    return ContainerConfig(
        image=args.image,
        entrypoint=_entrypoint(["bash", "-lc", run_script], workdir_files={}),
        env={},
        workdir="/app",
        task_id=args.task_id,
        resources=resources,
        timeout_seconds=args.timeout_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("lib/iris/examples/coreweave.yaml"))
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--task-id", default=f"deepep-krt-smoke-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=256)
    parser.add_argument("--disk-gib", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--run-bench", action="store_true")
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--distribution", choices=("random", "runs"), default="random")
    parser.add_argument("--num-local-experts", type=int, default=16)
    parser.add_argument(
        "--kernel",
        choices=("deep_ep", "hybrid_ep", "hybrid_ep_permute", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
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
