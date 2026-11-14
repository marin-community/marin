# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune the fused-MoE Triton config for Hero's EP decode shapes on one GB200 host.

Runs moe_tune_kernel.py inside the same isolated vLLM environment the server uses (the release's
wheel, torch and CUDA toolchain) and uploads the result JSON. Defaults match Hero at EP16:
384 experts, top-8, latent/intermediate 3072, 24 local experts per rank.

Usage (from a marin checkout):
    uv run python experiments/evaluation/hero_vllm_hillclimb/moe_tune.py --name moe-tune-ep16
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from fray.iris_backend import convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.vllm_backend import vllm_launcher
from marin.inference.vllm_server import _CUDA_NVCC_BOOTSTRAP
from rigging.filesystem.factory import open_url
from rigging.log_setup import configure_logging

from experiments.evaluation.hero_vllm_hillclimb.arm import OUTPUT_ROOT, RELEASES

CLUSTER = "cw-us-east-08a"
# Gathered tokens per decode step: 16 ranks x the captured batch sizes up to 128.
EP16_MS = "16,32,64,128,256,384,512,768,1024,1536,2048"
LOCAL_RESULT = "/tmp/moe_tune.json"


def run_tune(release: str, output_uri: str, kernel_args: tuple[str, ...]) -> None:
    """Iris entrypoint: run the tuner in the release's vLLM environment and upload its JSON."""
    configure_logging()
    launcher = vllm_launcher(
        VllmEngineConfig(launcher=VllmLauncherType.CUDA, source=VllmSource.MARIN_FORK, gpu_release=RELEASES[release])
    )
    command = launcher.command()
    command = [
        *command[: command.index(_CUDA_NVCC_BOOTSTRAP) + 1],
        "python",
        str(Path(__file__).with_name("moe_tune_kernel.py")),
        LOCAL_RESULT,
        *kernel_args,
    ]
    subprocess.run(command, env={**os.environ, **launcher.env()}, check=True)
    with open(LOCAL_RESULT, "rb") as source, open_url(f"{output_uri}/moe_tune.json", "wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), default="0ac9dcb")
    parser.add_argument("--local-experts", type=int, default=24)
    parser.add_argument("--ms", default=EP16_MS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    name = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    output_uri = f"{OUTPUT_ROOT}/{name}"
    kernel_args = (
        "--experts",
        "384",
        "--local-experts",
        str(args.local_experts),
        "--topk",
        "8",
        "--hidden",
        "3072",
        "--intermediate",
        "3072",
        "--ms",
        args.ms,
    )
    resources = ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="400g", disk="200g", replicas=1, preemptible=False)
    print(f"{name} -> {output_uri} {kernel_args}")
    if args.dry_run:
        return
    workspace = Path.cwd()
    with connect_controller(cluster_name=CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=workspace, credentials=endpoint.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_tune, args.release, output_uri, kernel_args),
                name=f"hero-vllm-{name}",
                resources=convert_resources(resources),
                environment=EnvironmentSpec(),
                replicas=1,
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                timeout=None,
            )
            print(f"submitted {job}")


if __name__ == "__main__":
    main()
