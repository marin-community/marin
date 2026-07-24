# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small client-side composition for a federated Iris inference endpoint."""

from __future__ import annotations

import re
import subprocess
from urllib.parse import urlsplit


_CAPABILITY_URL = re.compile(r"https://\S+")
DEFAULT_PARENT_CLUSTER = "marin"
DEFAULT_PARENT_INGRESS = "https://iris.oa.dev"
DUMMY_API_KEY = "capability-url-no-auth-header"


def build_marin_serve_command(args) -> list[str]:
    """Build a parent-delegated ``marin-serve`` command for an external eval."""
    command = [
        "uv",
        "run",
        "--project",
        "lib/marin",
        "marin-serve",
        "iris",
        args.external_serve_model,
        "--cluster",
        args.external_parent_cluster,
        "--target-cluster",
        args.target_cluster,
        "--gpu",
        args.gpu,
        "--name",
        args.external_serve_name,
        "--endpoint-name",
        args.external_endpoint_name,
        "--max-model-len",
        str(args.external_serve_max_model_len),
        "--max-num-batched-tokens",
        str(args.external_serve_max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.external_serve_tensor_parallel_size),
        "--dtype",
        "bfloat16",
        "--vllm-source",
        "marin-fork",
        "--proxy-timeout",
        "600",
        "--cpu",
        "48",
        "--memory",
        "1024g",
        "--disk",
        "512g",
        "--priority",
        args.priority,
        "--idle-timeout-hours",
        str(args.external_serve_idle_timeout_hours),
        "--no-wait",
        f"--vllm-arg=--data-parallel-size={args.external_serve_data_parallel_size}",
        f"--vllm-arg=--max-num-seqs={args.external_serve_max_num_seqs}",
    ]
    command.extend(f"--vllm-arg={value}" for value in args.external_serve_vllm_arg)
    return command


def build_wait_and_mint_command(args) -> list[str]:
    """Use the Iris CLI's canonical non-interactive federated endpoint path."""
    return [
        args.iris_bin,
        "--cluster",
        args.external_parent_cluster,
        "endpoints",
        "wait-and-mint",
        args.external_endpoint_name,
        "--require-peer",
        "--ttl-hours",
        str(args.external_ttl_hours),
        "--timeout-seconds",
        str(args.external_ready_timeout_seconds),
    ]


def mint_external_api_base(args, *, run=subprocess.run) -> str:
    """Return a parent-scoped OpenAI base URL without exposing its token on failure."""
    result = run(build_wait_and_mint_command(args), capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Iris endpoint wait-and-mint failed; inspect the controller job diagnostics.")
    urls = _CAPABILITY_URL.findall(result.stdout)
    if len(urls) != 1:
        raise RuntimeError("Iris endpoint wait-and-mint returned no unambiguous capability URL.")
    expected_host = urlsplit(args.external_parent_ingress_host).hostname
    if urlsplit(urls[0]).hostname != expected_host:
        raise RuntimeError("Iris parent returned a capability URL for an unexpected ingress host.")
    return f"{urls[0].rstrip('/')}/v1"


def durable_harbor_jobs_dir(s3_output_root: str, job_name: str) -> str:
    root = s3_output_root.rstrip("/")
    if not root.startswith("s3://"):
        raise ValueError("--external-s3-output-root must start with s3://")
    if not job_name or "/" in job_name:
        raise ValueError("--job-name must be a non-empty Iris job-name component")
    return f"{root}/{job_name}/trace_jobs"
