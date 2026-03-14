#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a CoreWeave H100x8 Megatron-style Qwen MoE benchmark pod."""

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
DEFAULT_WORKTREE = Path("/Users/romain/marin-wt/moe-megatron-qwen-scale")
DEFAULT_CONFIG = Path("lib/iris/examples/coreweave.yaml")
DEFAULT_MEGATRON_REF = "f8becec65f47982c80c3d397bef7c3fba65f9efc"
DEFAULT_DEEPEP_REF = "7febc6e25660af0f54d95dd781ecdcd62265ecca"
BENCH_PATH = Path(".agents/scripts/megatron_qwen_moe_perf.py")


def _entrypoint(argv: list[str], *, workdir_files: dict[str, bytes]) -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv))
    entrypoint.workdir_files.update(workdir_files)
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
    shutil.rmtree(target_dir)

urllib.request.urlretrieve(archive_url, download_path)
with tarfile.open(download_path, "r:gz") as archive:
    temp_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    archive.extractall(temp_dir)
    extracted_root = next(temp_dir.iterdir())
    extracted_root.rename(target_dir)
print("UNPACKED", target_dir)
PY
"""


def _build_run_script(args: argparse.Namespace, *, bench_file_b64: str) -> str:
    megatron_url = f"https://codeload.github.com/NVIDIA/Megatron-LM/tar.gz/{args.megatron_ref}"
    deepep_url = f"https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/{args.deepep_ref}"
    hybrid_ep_patch_block = ""
    if args.patch_hybrid_ep_space_cluster:
        hybrid_ep_patch_block = r"""
/opt/conda/bin/python - <<'PY'
from pathlib import Path

backend = Path("/tmp/DeepEP/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh")
text = backend.read_text()
old = "cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,\n"
new = "cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster,\n"
count = text.count(old)
if count == 0:
    raise SystemExit("PATCH_FAILED no space_shared cp_async_bulk callsites found")
backend.write_text(text.replace(old, new))
print("PATCHED_HYBRID_EP_SPACE_CLUSTER", count)
PY
rg -n "cp_async_bulk\\(cuda::ptx::space_(cluster|shared)" /tmp/DeepEP/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh
"""

    bench_cmd = f"""
echo RUNNING_MEGATRON_QWEN_BENCH
torchrun --standalone --nproc_per_node={args.gpus} /app/{BENCH_PATH} \\
  --cases {args.cases} \\
  --dispatchers {args.dispatchers} \\
  --warmup-iters {args.warmup_iters} \\
  --measure-iters {args.measure_iters} \\
  --dummy-gemm-size {args.dummy_gemm_size} \\
  --output-jsonl /app/megatron_qwen_results.jsonl
echo BENCH_DONE
cat /app/megatron_qwen_results.jsonl
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
if [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends libibverbs-dev rdma-core
  rm -rf /var/lib/apt/lists/*
fi
/opt/conda/bin/python -m pip install --no-cache-dir setuptools wheel ninja packaging
{_download_unpack_block(megatron_url, "/tmp/Megatron-LM")}
{_download_unpack_block(deepep_url, "/tmp/DeepEP")}
{hybrid_ep_patch_block}
ORIGINAL_PYTHONPATH="${{PYTHONPATH:-}}"
CUDA_ROOT=/usr/local/cuda
CUDNN_ROOT=$(/opt/conda/bin/python - <<'PY'
from pathlib import Path
import sys

base = Path(sys.executable).resolve().parent.parent
site_packages = (
    base / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "site-packages"
)
candidates = (
    site_packages / "nvidia" / "cudnn",
    Path("/usr/local/cuda"),
    Path("/usr"),
)
for candidate in candidates:
    include_path = candidate / "include" / "cudnn.h"
    has_library_dir = any(path.is_dir() for path in (candidate / "lib", candidate / "lib64"))
    if include_path.is_file() and has_library_dir:
        print(candidate)
        raise SystemExit(0)
raise SystemExit(f"missing usable cuDNN root under: {{candidates}}")
PY
)
NVSHMEM_ROOT=$(/opt/conda/bin/python - <<'PY'
from pathlib import Path
import sys

base = Path(sys.executable).resolve().parent.parent
site_packages = (
    base / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "site-packages"
)
candidate = site_packages / "nvidia" / "nvshmem"
if not candidate.is_dir():
    raise SystemExit(f"missing NVSHMEM root: {{candidate}}")
print(candidate)
PY
)
echo CUDNN_ROOT="$CUDNN_ROOT"
echo NVSHMEM_ROOT="$NVSHMEM_ROOT"
ls -al "$CUDNN_ROOT"
ls -al "$CUDNN_ROOT/include"
if [ -d "$CUDNN_ROOT/lib" ]; then
  ls -al "$CUDNN_ROOT/lib"
fi
if [ -d "$CUDNN_ROOT/lib64" ]; then
  ls -al "$CUDNN_ROOT/lib64"
fi
ls -al "$NVSHMEM_ROOT"
ls -al "$NVSHMEM_ROOT/include"
ls -al "$NVSHMEM_ROOT/lib"
NVSHMEM_HOST_LIB=$(basename "$(ls -1 "$NVSHMEM_ROOT/lib"/libnvshmem_host.so.* | sort | tail -1)")
ln -sf "$NVSHMEM_HOST_LIB" "$NVSHMEM_ROOT/lib/libnvshmem_host.so"
ls -al "$NVSHMEM_ROOT/lib"/libnvshmem_host.so*
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDNN_HOME="$CUDNN_ROOT"
export CUDNN_PATH="$CUDNN_ROOT"
export NVSHMEM_DIR="$NVSHMEM_ROOT"
export CPATH="$CUDNN_ROOT/include:$NVSHMEM_ROOT/include${{CPATH:+:$CPATH}}"
export CPLUS_INCLUDE_PATH="$CUDNN_ROOT/include:$NVSHMEM_ROOT/include${{CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}}"
export CMAKE_PREFIX_PATH="$CUDNN_ROOT:$NVSHMEM_ROOT${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
export MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=1
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export TORCH_CUDA_ARCH_LIST="{args.torch_cuda_arch_list}"
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
mkdir -p /tmp/nvtxshim
ln -sf "$NVTX_LIB_DIR/libnvToolsExt.so.1" /tmp/nvtxshim/libnvtx3interop.so
LIB_DIRS="$CUDNN_ROOT/lib:$NVSHMEM_ROOT/lib"
if [ -d "$CUDNN_ROOT/lib64" ]; then
  LIB_DIRS="$LIB_DIRS:$CUDNN_ROOT/lib64"
fi
export LIBRARY_PATH="$LIB_DIRS:/tmp/nvtxshim:$NVTX_LIB_DIR${{LIBRARY_PATH:+:$LIBRARY_PATH}}"
export LD_LIBRARY_PATH="$LIB_DIRS:/tmp/nvtxshim:$NVTX_LIB_DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
/opt/conda/bin/python -m pip install --no-cache-dir --no-build-isolation \\
  "transformer-engine[pytorch]" \\
  "nv-grouped-gemm~=1.1"
/opt/conda/bin/python -m pip install --no-build-isolation /tmp/DeepEP
if [ -n "$ORIGINAL_PYTHONPATH" ]; then
  export PYTHONPATH="/tmp/Megatron-LM:$ORIGINAL_PYTHONPATH"
else
  export PYTHONPATH="/tmp/Megatron-LM"
fi
/opt/conda/bin/python - <<'PY'
import deep_ep
import megatron.core.parallel_state as ps
import transformer_engine
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP, HAVE_HYBRIDEP

print("TRANSFORMER_ENGINE_OK", getattr(transformer_engine, "__version__", "unknown"))
print("DEEPEP_OK", deep_ep.__file__)
print("MEGATRON_OK", ps.__file__)
print("HAVE_DEEP_EP", HAVE_DEEP_EP)
print("HAVE_HYBRIDEP", HAVE_HYBRIDEP)
PY
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--task-id", default=f"megatron-qwen-krt-bench-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--megatron-ref", default=DEFAULT_MEGATRON_REF)
    parser.add_argument("--deepep-ref", default=DEFAULT_DEEPEP_REF)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=256)
    parser.add_argument("--disk-gib", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--dispatchers", default="all")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--dummy-gemm-size", type=int, default=8192)
    parser.add_argument("--torch-cuda-arch-list", default="9.0")
    parser.add_argument("--patch-hybrid-ep-space-cluster", action="store_true", default=True)
    parser.add_argument(
        "--no-patch-hybrid-ep-space-cluster",
        dest="patch_hybrid_ep_space_cluster",
        action="store_false",
    )
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
        print(f"MEGATRON_REF={args.megatron_ref}", flush=True)
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
