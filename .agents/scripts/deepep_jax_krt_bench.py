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
DEFAULT_DEEPEP_REF = "7febc6e25660af0f54d95dd781ecdcd62265ecca"
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
expected_is_token_in_rank = (topk_idx[..., None] // experts_per_rank == jnp.arange(NUM_RANKS)).any(axis=1)
expected_rank_counts = expected_is_token_in_rank.astype(jnp.int32).sum(axis=0)

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
    profile_flag = f" \\\n        --profile-root {args.profile_root}" if args.profile_root else ""
    w13_layout_flag = " \\\n        --w13-out-first" if args.w13_out_first else ""
    w2_layout_flag = " \\\n        --w2-out-first" if args.w2_out_first else ""
    w13_expert_padded_flag = " \\\n        --w13-expert-padded" if args.w13_expert_padded else ""
    w2_expert_padded_flag = " \\\n        --w2-expert-padded" if args.w2_expert_padded else ""
    shared_mlp_explicit_bwd_flag = " \\\n        --shared-mlp-explicit-bwd" if args.shared_mlp_explicit_bwd else ""
    shared_mlp_fast_accum_flag = " \\\n        --shared-mlp-fast-accum" if args.shared_mlp_fast_accum else ""
    combine_fast_accum_flag = " \\\n        --combine-fast-accum" if args.combine_fast_accum else ""
    collapse_impl_flag = (
        ""
        if args.deepep_collapse_impl == "segment_sum"
        else f" \\\n        --deepep-collapse-impl {args.deepep_collapse_impl}"
    )
    deepep_dispatch_num_sms_flag = (
        "" if args.deepep_dispatch_num_sms is None else f" \\\n        --deepep-dispatch-num-sms {args.deepep_dispatch_num_sms}"
    )
    deepep_dispatch_num_max_send_tokens_flag = (
        ""
        if args.deepep_dispatch_num_max_send_tokens is None
        else f" \\\n        --deepep-dispatch-num-max-send-tokens {args.deepep_dispatch_num_max_send_tokens}"
    )
    deepep_dispatch_num_max_recv_tokens_flag = (
        ""
        if args.deepep_dispatch_num_max_recv_tokens is None
        else f" \\\n        --deepep-dispatch-num-max-recv-tokens {args.deepep_dispatch_num_max_recv_tokens}"
    )
    deepep_combine_num_sms_flag = (
        "" if args.deepep_combine_num_sms is None else f" \\\n        --deepep-combine-num-sms {args.deepep_combine_num_sms}"
    )
    deepep_combine_num_max_send_tokens_flag = (
        ""
        if args.deepep_combine_num_max_send_tokens is None
        else f" \\\n        --deepep-combine-num-max-send-tokens {args.deepep_combine_num_max_send_tokens}"
    )
    deepep_combine_num_max_recv_tokens_flag = (
        ""
        if args.deepep_combine_num_max_recv_tokens is None
        else f" \\\n        --deepep-combine-num-max-recv-tokens {args.deepep_combine_num_max_recv_tokens}"
    )
    timeout_prefix = ""
    timeout_suffix = ""
    if args.per_bench_timeout_seconds is not None:
        timeout_prefix = f"timeout -k {args.per_bench_kill_after_seconds}s {args.per_bench_timeout_seconds}s "
        timeout_suffix = """
      bench_status=$?
      if [ "$bench_status" -eq 124 ] || [ "$bench_status" -eq 137 ]; then
        echo "BENCH_TIMEOUT kernel=$kernel distribution=$distribution topk=$topk"
      elif [ "$bench_status" -ne 0 ]; then
        exit "$bench_status"
      fi
"""
    return f"""
echo RUNNING_BENCH_MATRIX
for distribution in {distributions}; do
  for topk in {topk_values}; do
    for kernel in {kernels}; do
      echo "BENCH_START kernel=$kernel distribution=$distribution topk=$topk"
      {timeout_prefix}.venv/bin/python {BENCH_PATH} \\
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
        --iters {args.iters}{profile_flag}{w13_layout_flag}{w2_layout_flag}{w13_expert_padded_flag}{w2_expert_padded_flag}{shared_mlp_explicit_bwd_flag}{shared_mlp_fast_accum_flag}{combine_fast_accum_flag}{collapse_impl_flag}{deepep_dispatch_num_sms_flag}{deepep_dispatch_num_max_send_tokens_flag}{deepep_dispatch_num_max_recv_tokens_flag}{deepep_combine_num_sms_flag}{deepep_combine_num_max_send_tokens_flag}{deepep_combine_num_max_recv_tokens_flag}
{timeout_suffix}
      echo "BENCH_END kernel=$kernel distribution=$distribution topk=$topk"
    done
  done
done
"""


def _post_bench_block(args: argparse.Namespace) -> str:
    if args.post_bench_sleep_seconds <= 0:
        return ""

    profile_listing = ""
    if args.profile_root is not None:
        profile_listing = f"""
if [ -d "{args.profile_root}" ]; then
  echo PROFILE_ROOT={args.profile_root}
  find "{args.profile_root}" -maxdepth 5 -type f | sort
fi
"""

    return f"""
{profile_listing}
echo POST_BENCH_SLEEP_SECONDS={args.post_bench_sleep_seconds}
sleep {args.post_bench_sleep_seconds}
"""


def _build_run_script(args: argparse.Namespace) -> str:
    repo_archive_url = f"https://codeload.github.com/marin-community/marin/tar.gz/{args.repo_ref}"
    deepep_archive_url = f"https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/{args.deepep_ref}"
    trust_runtime_recv_count_flag = (
        f"\nexport LEVANTER_DEEPEP_TRUST_RUNTIME_RECV_COUNT={1 if args.deepep_trust_runtime_recv_count else 0}"
        if args.deepep_trust_runtime_recv_count
        else ""
    )
    host_dispatch_debug_flag = (
        f"\nexport LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG={1 if args.deepep_host_dispatch_debug else 0}"
        if args.deepep_host_dispatch_debug
        else ""
    )
    return f"""set -euxo pipefail
echo HOSTNAME=$(hostname)
echo PYTHON=$(/opt/conda/bin/python -c 'import sys; print(sys.executable)')
/opt/conda/bin/python --version
command -v nvcc
nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader
/opt/conda/bin/python -m pip install --no-cache-dir uv
if ! command -v npx >/dev/null 2>&1; then
  apt-get update
  apt-get install -y nodejs npm
fi
{_download_unpack_block(repo_archive_url, "/app")}
{_download_unpack_block(deepep_archive_url, "/tmp/DeepEP")}
cd /app
export DEEPEP_SRC_ROOT=/tmp/DeepEP
export DEEPEP_BUILD_WITH_TORCH_EXTENSION={1 if args.build_with_torch_extension else 0}
export DEEPEP_LOAD_AS_PYTHON_MODULE={1 if args.load_as_python_module else 0}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export PYTHONPATH=/opt/conda/lib/python3.11/site-packages${{PYTHONPATH:+:$PYTHONPATH}}
{trust_runtime_recv_count_flag}
{host_dispatch_debug_flag}
uv sync --quiet --frozen --link-mode symlink --python 3.11 --package levanter --no-group dev --extra gpu
.venv/bin/python - <<'PY'
import jax
import jaxlib
import os
import torch
print("JAX_VERSION", jax.__version__)
print("JAXLIB_VERSION", jaxlib.__version__)
print("JAX_DEVICES", jax.devices())
print("DEEPEP_BUILD_WITH_TORCH_EXTENSION", os.environ["DEEPEP_BUILD_WITH_TORCH_EXTENSION"])
print("DEEPEP_LOAD_AS_PYTHON_MODULE", os.environ["DEEPEP_LOAD_AS_PYTHON_MODULE"])
print("LEVANTER_DEEPEP_TRUST_RUNTIME_RECV_COUNT", os.environ.get("LEVANTER_DEEPEP_TRUST_RUNTIME_RECV_COUNT", "0"))
print("LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG", os.environ.get("LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG", "0"))
print("TORCH_VERSION", torch.__version__)
PY
{_smoke_block(args)}
{_bench_block(args)}
{_post_bench_block(args)}
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
        node_selector=_parse_node_selectors(args.node_selector),
    )


def _parse_node_selectors(raw_selectors: list[str]) -> dict[str, str]:
    selectors: dict[str, str] = {}
    for item in raw_selectors:
        if "=" not in item:
            raise ValueError(f"node selector must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"node selector must be KEY=VALUE, got: {item}")
        selectors[key] = value
    return selectors


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
    parser.add_argument("--per-bench-timeout-seconds", type=int, default=None)
    parser.add_argument("--per-bench-kill-after-seconds", type=int, default=10)
    parser.add_argument("--build-with-torch-extension", action="store_true")
    parser.add_argument("--load-as-python-module", action="store_true")
    parser.add_argument("--profile-root", default=None)
    parser.add_argument("--post-bench-sleep-seconds", type=int, default=0)
    parser.add_argument("--w13-out-first", action="store_true")
    parser.add_argument("--w2-out-first", action="store_true")
    parser.add_argument("--w13-expert-padded", action="store_true")
    parser.add_argument("--w2-expert-padded", action="store_true")
    parser.add_argument("--shared-mlp-explicit-bwd", action="store_true")
    parser.add_argument("--shared-mlp-fast-accum", action="store_true")
    parser.add_argument("--combine-fast-accum", action="store_true")
    parser.add_argument("--deepep-dispatch-num-sms", type=int, default=None)
    parser.add_argument("--deepep-dispatch-num-max-send-tokens", type=int, default=None)
    parser.add_argument("--deepep-dispatch-num-max-recv-tokens", type=int, default=None)
    parser.add_argument("--deepep-combine-num-sms", type=int, default=None)
    parser.add_argument("--deepep-combine-num-max-send-tokens", type=int, default=None)
    parser.add_argument("--deepep-combine-num-max-recv-tokens", type=int, default=None)
    parser.add_argument(
        "--deepep-collapse-impl",
        choices=("segment_sum", "sorted_segment_sum", "scatter_add", "lax_scatter"),
        default="segment_sum",
    )
    parser.add_argument("--deepep-trust-runtime-recv-count", action="store_true")
    parser.add_argument("--deepep-host-dispatch-debug", action="store_true")
    parser.add_argument(
        "--node-selector",
        action="append",
        default=[],
        help="Repeatable Kubernetes node selector in KEY=VALUE form.",
    )
    parser.add_argument("--skip-cleanup", action="store_true")
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
        if not args.skip_cleanup:
            runtime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
