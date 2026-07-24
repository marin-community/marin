# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the historical TransformerEngine NCCL-EP reproducer in NVIDIA JAX."""

import argparse
import runpy
import shlex
import sys
from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.grug.dispatch import dispatch_grug_training_run

NVIDIA_JAX_IMAGE = "nvcr.io/nvidia/jax:26.06-py3"
NCCLEP_STASH = "s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep"
NCCLEP_WORK = "/tmp/ncclep-ngc"
NCCL_RUNTIME_VERSION = "2.30.7"
NCCL_LIBRARY_PATH = "/app/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"


@dataclass(frozen=True)
class StandaloneTrial:
    arguments: tuple[str, ...]


def _run_trial(trial: StandaloneTrial) -> None:
    sys.argv = ["grug_moe_mfu.py", *trial.arguments]
    runpy.run_module("experiments.grug.moe.standalone.grug_moe_mfu", run_name="__main__")


def ncclep_setup_script(stash: str = NCCLEP_STASH, work: str = NCCLEP_WORK) -> str:
    """Install the historical NCCL-EP wheel without replacing NGC dependencies."""
    return f"""set -eu
work={shlex.quote(work)}
stash={shlex.quote(stash)}
rm -rf "$work"
mkdir -p "$work"
"$IRIS_VENV/bin/python" - "$stash" "$work" <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

import fsspec

stash = sys.argv[1].rstrip("/")
work = Path(sys.argv[2])
filesystem, stash_path = fsspec.core.url_to_fs(stash)
wheels = sorted(filesystem.glob(stash_path + "/wheels/*.whl"))
assert wheels, f"no wheels under {{stash}}/wheels/"
wheel_path = wheels[-1]
wheel = work / Path(wheel_path).name
headers = work / "nccl-ep-jit-headers.tgz"
filesystem.get(wheel_path, str(wheel))
filesystem.get(stash_path + "/jit/nccl-ep-jit-headers.tgz", str(headers))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()

print(f"NCCL-EP wheel {{wheel.name}} sha256={{sha256(wheel)}}")
print(f"NCCL-EP headers sha256={{sha256(headers)}}")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "nvidia-nccl-cu13=={NCCL_RUNTIME_VERSION}"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
jit_include = work / "jit-include"
jit_include.mkdir()
with tarfile.open(headers) as archive:
    archive.extractall(jit_include, filter="data")
for source in sorted(jit_include.rglob("nccl_ep.cc")):
    lines = source.read_text().splitlines()
    print(f"NCCL-EP source {{source}}")
    for line_number in range(1645, min(1660, len(lines)) + 1):
        print(f"{{line_number}}: {{lines[line_number - 1]}}")
PY
test -x /usr/local/cuda/bin/nvcc
"$IRIS_VENV/bin/python" - <<'PY'
import os
import ctypes
import subprocess

import jax
import jaxlib
import transformer_engine
from transformer_engine.jax.ep import ep_bootstrap
import transformer_engine_jax

assert jax.__file__.startswith("/opt/jax/"), jax.__file__
assert jaxlib.__file__.startswith("/opt/jaxlibs/"), jaxlib.__file__
assert transformer_engine.__file__.startswith(os.environ["IRIS_VENV"] + "/"), transformer_engine.__file__
assert callable(ep_bootstrap)
print(f"TransformerEngine JAX extension {{transformer_engine_jax.__file__}}")
libnccl = ctypes.CDLL("libnccl.so.2")
nccl_version = ctypes.c_int()
assert libnccl.ncclGetVersion(ctypes.byref(nccl_version)) == 0
print(f"runtime NCCL version {{nccl_version.value}}")
subprocess.run(["ldd", transformer_engine_jax.__file__], check=True)
print(f"NCCL-EP TransformerEngine {{transformer_engine.__version__}} from {{transformer_engine.__file__}}")
print(f"preserved NGC JAX {{jax.__version__}} from {{jax.__file__}}")
print(f"preserved NGC JAXLIB {{jaxlib.__version__}} from {{jaxlib.__file__}}")
PY
"""


def _xla_flags(cmd_buffer: str, extra_xla_flags: str) -> str:
    """Mirror run_bench_gang.sh's NCCLEP_CMD_BUFFER semantics.

    ``off`` disables capture globally (the NCCLEP baseline), ``default``
    omits the flag (XLA's default capture set), anything else is an explicit
    CommandBufferCmdType list.
    """
    if cmd_buffer == "off":
        flags = "--xla_gpu_enable_command_buffer="
    elif cmd_buffer == "default":
        flags = ""
    else:
        flags = f"--xla_gpu_enable_command_buffer={cmd_buffer}"
    return f"{flags} {extra_xla_flags}".strip()


def launch_trial(
    *,
    run_id: str,
    arguments: tuple[str, ...],
    replicas: int,
    gpus_per_node: int,
    cmd_buffer: str = "off",
    extra_xla_flags: str = "",
) -> None:
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=gpus_per_node,
        cpu=32,
        ram="256g",
        disk="256g",
        replicas=replicas,
        image=NVIDIA_JAX_IMAGE,
    )
    dispatch_grug_training_run(
        run_id=run_id,
        config=StandaloneTrial(("--run-id", run_id, *arguments)),
        local_entrypoint=_run_trial,
        resources=resources,
        max_retries_failure=0,
        processes_per_task=gpus_per_node,
        task_env_vars={
            "IRIS_MULTIGPU_ISOLATE_CUDA_VISIBLE_DEVICES": "1",
            "LD_LIBRARY_PATH": NCCL_LIBRARY_PATH,
            "JAX_COMPILATION_CACHE_DIR": "/tmp/jax-compile-cache",
            "NCCL_EP_JIT_BUILD_INCLUDE_DIR": f"{NCCLEP_WORK}/jit-include",
            "NCCL_EP_JIT_SOURCE_DIR": f"{NCCLEP_WORK}/jit-include/nccl_ep",
            "NVTE_EP_HANDLE_CACHE_SIZE": "-1",
            "XLA_FLAGS": _xla_flags(cmd_buffer, extra_xla_flags),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        },
        task_setup_scripts=(ncclep_setup_script(),),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--cmd-buffer", default="off", help="off | default | explicit CommandBufferCmdType list")
    parser.add_argument("--extra-xla-flags", default="")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    if not arguments:
        raise ValueError("Pass standalone grug_moe_mfu.py arguments after '--'.")
    launch_trial(
        run_id=args.run_id,
        arguments=arguments,
        replicas=args.replicas,
        gpus_per_node=args.gpus_per_node,
        cmd_buffer=args.cmd_buffer,
        extra_xla_flags=args.extra_xla_flags,
    )


if __name__ == "__main__":
    main()
