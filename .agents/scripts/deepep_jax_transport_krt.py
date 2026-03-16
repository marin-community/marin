#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a CoreWeave H100x8 pure-JAX DeepEP transport benchmark pod."""

from __future__ import annotations

import argparse
import base64
import subprocess
import time
from pathlib import Path

from iris.cluster.config import load_config
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerPhase
from iris.rpc import cluster_pb2

DEFAULT_IMAGE = "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel"
DEFAULT_CONFIG = Path("lib/iris/examples/coreweave.yaml")
DEFAULT_WORKTREE = Path("/Users/romain/marin-wt/moe-jax-megatron-root-cause")
DEFAULT_DEEPEP_REF = "7febc6e25660af0f54d95dd781ecdcd62265ecca"
BENCH_PATH = Path("lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py")
PATCH_SCRIPT_PATH = Path(".agents/scripts/patch_deepep_intranode_launch_debug.py")
LOCAL_STAGE_PATHS: tuple[Path, ...] = ()
PY_COMPILE_PATHS = (
    PATCH_SCRIPT_PATH,
    Path("lib/levanter/src/levanter/kernels/deepep/__init__.py"),
    Path("lib/levanter/src/levanter/kernels/deepep/layout_ffi.py"),
    Path("lib/levanter/src/levanter/kernels/deepep/transport_ffi.py"),
    BENCH_PATH,
)


def _current_worktree_ref(worktree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
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
import time
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

last_error = None
for attempt in range(1, 6):
    try:
        with urllib.request.urlopen(archive_url, timeout=120) as response:
            download_path.write_bytes(response.read())
        break
    except Exception as exc:
        last_error = exc
        if attempt == 5:
            raise
        time.sleep(min(30, 5 * attempt))
else:
    raise RuntimeError(f"failed to download {{archive_url}}: {{last_error}}")

with tarfile.open(download_path, "r:gz") as archive:
    temp_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    archive.extractall(temp_dir)
    extracted_root = next(temp_dir.iterdir())
    for child in extracted_root.iterdir():
        shutil.move(str(child), target_dir / child.name)
print("UNPACKED", target_dir)
PY
"""


def _stage_file_block(path: Path, encoded: str) -> str:
    return f"""
/opt/conda/bin/python - <<'PY'
import base64
from pathlib import Path

target = Path("/app/{path}")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(base64.b64decode("{encoded}"))
print("STAGED", target)
PY
"""


def _bench_block(args: argparse.Namespace) -> str:
    commands: list[str] = []
    python_runner = "/opt/conda/bin/python"
    if args.compute_sanitizer:
        python_runner = "compute-sanitizer --tool memcheck --target-processes all /opt/conda/bin/python"
    probe_flags = ""
    if args.probe_only:
        probe_flags = f"  --probe-only \\\n  --probe-max-elements {args.probe_max_elements} \\\n"
    if args.host_kernel_probe_only:
        probe_flags += "  --host-kernel-probe-only \\\n"
    if args.host_dispatch_round_only:
        probe_flags += "  --host-dispatch-round-only \\\n"
    if args.dispatch_num_sms is not None:
        probe_flags += f"  --dispatch-num-sms {args.dispatch_num_sms} \\\n"
    if args.dispatch_num_max_send_tokens is not None:
        probe_flags += f"  --dispatch-num-max-send-tokens {args.dispatch_num_max_send_tokens} \\\n"
    if args.dispatch_num_max_recv_tokens is not None:
        probe_flags += f"  --dispatch-num-max-recv-tokens {args.dispatch_num_max_recv_tokens} \\\n"
    if args.combine_num_sms is not None:
        probe_flags += f"  --combine-num-sms {args.combine_num_sms} \\\n"
    if args.combine_num_max_send_tokens is not None:
        probe_flags += f"  --combine-num-max-send-tokens {args.combine_num_max_send_tokens} \\\n"
    if args.combine_num_max_recv_tokens is not None:
        probe_flags += f"  --combine-num-max-recv-tokens {args.combine_num_max_recv_tokens} \\\n"
    for distribution in args.distributions.split(","):
        for topk_text in args.topk_list.split(","):
            topk = int(topk_text)
            commands.append(
                f"""
echo "BENCH_START distribution={distribution} topk={topk}"
{python_runner} {BENCH_PATH} \\
  --tokens {args.tokens} \\
  --hidden {args.hidden} \\
  --experts {args.experts} \\
  --topk {topk} \\
  --distribution {distribution} \\
  --execution-model {args.execution_model} \\
  --seed {args.seed} \\
  --warmup {args.warmup} \\
  --iters {args.iters} \\
{probe_flags}\
  --check
echo "BENCH_END distribution={distribution} topk={topk}"
"""
            )
    return "\n".join(commands)


def _launch_debug_block(args: argparse.Namespace) -> str:
    if not args.launch_debug:
        return ""
    return f"""
/opt/conda/bin/python /app/{PATCH_SCRIPT_PATH} --root /tmp/DeepEP
export LEVANTER_DEEPEP_LAUNCH_DEBUG=1
export LEVANTER_DEEPEP_LAUNCH_DEBUG_LABEL={args.launch_debug_label}
"""


def _build_run_script(args: argparse.Namespace, *, staged_files: dict[Path, str]) -> str:
    repo_archive_url = f"https://codeload.github.com/marin-community/marin/tar.gz/{args.repo_ref}"
    deepep_archive_url = f"https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/{args.deepep_ref}"
    stage_block = "\n".join(_stage_file_block(path, encoded) for path, encoded in staged_files.items())
    py_compile_targets = " ".join(str(path) for path in PY_COMPILE_PATHS)
    return f"""set -euxo pipefail
echo HOSTNAME=$(hostname)
echo PYTHON=$(/opt/conda/bin/python -c 'import sys; print(sys.executable)')
/opt/conda/bin/python --version
command -v nvcc
command -v c++
{"command -v compute-sanitizer" if args.compute_sanitizer else ""}
nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader
{_download_unpack_block(repo_archive_url, "/app")}
{stage_block}
{_download_unpack_block(deepep_archive_url, "/tmp/DeepEP")}
{_launch_debug_block(args)}
cd /app
export DEEPEP_SRC_ROOT=/tmp/DeepEP
export DISABLE_SM90_FEATURES={1 if args.disable_sm90_features else 0}
export DEEPEP_BUILD_WITH_TORCH_EXTENSION={1 if args.build_with_torch_extension else 0}
export DEEPEP_LOAD_AS_PYTHON_MODULE={1 if args.load_as_python_module else 0}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export MAX_JOBS=8
/opt/conda/bin/python - <<'PY'
import os

for key in sorted(
    (
        "DEEPEP_BUILD_WITH_TORCH_EXTENSION",
        "DEEPEP_LOAD_AS_PYTHON_MODULE",
        "DEEPEP_SRC_ROOT",
        "DISABLE_SM90_FEATURES",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "MAX_JOBS",
    )
):
    print(f"{{key}}={{os.environ[key]}}")
PY
/opt/conda/bin/python -m pip install --no-cache-dir "jax[cuda12]==0.8.0"
/opt/conda/bin/python -m py_compile {py_compile_targets}
/opt/conda/bin/python - <<'PY'
import jax
import jaxlib
import importlib.util
import sys
from pathlib import Path

root = Path("/app/lib/levanter/src/levanter/kernels/deepep")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {{path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

layout = load("levanter_deepep_layout_ffi", root / "layout_ffi.py")
transport = load("levanter_deepep_transport_ffi", root / "transport_ffi.py")

print("JAX_VERSION", jax.__version__)
print("JAXLIB_VERSION", jaxlib.__version__)
print("JAX_DEVICES", jax.devices())
print(
    "DEEPEP_SYMBOLS",
    transport.deepep_dispatch_intranode.__name__,
    transport.deepep_combine_intranode.__name__,
    layout.deepep_get_dispatch_layout.__name__,
)
PY
{_bench_block(args)}
"""


def _make_container_config(args: argparse.Namespace) -> ContainerConfig:
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=args.cpu * 1000,
        memory_bytes=args.memory_gib * 1024**3,
        disk_bytes=args.disk_gib * 1024**3,
    )
    resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="H100", count=args.gpus))
    staged_files = {
        path: base64.b64encode((args.worktree / path).read_bytes()).decode("ascii") for path in LOCAL_STAGE_PATHS
    }
    run_script = _build_run_script(args, staged_files=staged_files)
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
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--repo-ref")
    parser.add_argument("--deepep-ref", default=DEFAULT_DEEPEP_REF)
    parser.add_argument("--task-id", default=f"deepep-jax-transport-krt-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=256)
    parser.add_argument("--disk-gib", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk-list", default="2")
    parser.add_argument("--distributions", default="random")
    parser.add_argument("--execution-model", choices=("shard_map", "pmap"), default="shard_map")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--disable-sm90-features", action="store_true")
    parser.add_argument("--build-with-torch-extension", action="store_true")
    parser.add_argument("--load-as-python-module", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--host-kernel-probe-only", action="store_true")
    parser.add_argument("--host-dispatch-round-only", action="store_true")
    parser.add_argument("--probe-max-elements", type=int, default=256)
    parser.add_argument("--dispatch-num-sms", type=int)
    parser.add_argument("--dispatch-num-max-send-tokens", type=int)
    parser.add_argument("--dispatch-num-max-recv-tokens", type=int)
    parser.add_argument("--combine-num-sms", type=int)
    parser.add_argument("--combine-num-max-send-tokens", type=int)
    parser.add_argument("--combine-num-max-recv-tokens", type=int)
    parser.add_argument("--launch-debug", action="store_true")
    parser.add_argument("--launch-debug-label", default="jax")
    parser.add_argument("--compute-sanitizer", action="store_true")
    args = parser.parse_args()
    if args.repo_ref is None:
        args.repo_ref = _current_worktree_ref(args.worktree)
    return args


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
