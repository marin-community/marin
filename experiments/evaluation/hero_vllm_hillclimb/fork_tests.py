# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a vLLM fork test selection on one GB200 host against a released (or overlay) wheel.

The fork's ``tests/`` tree is uploaded as a tarball (it is not in the wheel); the tests run in the
same isolated environment the server uses, from the extracted tree so ``tests.*`` imports resolve.

Usage (from a marin checkout):
    tar -C <fork checkout> -czf tests.tgz tests && <upload to object storage>
    uv run python experiments/evaluation/hero_vllm_hillclimb/fork_tests.py --name fork-tests \
        --tests-tgz s3://.../tests.tgz --overlay-wheel FILE:SHA256 -- tests/models/test_grugmoe.py -k graph
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from fray.iris_backend import convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.external_dependencies import VllmGpuRelease
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.vllm_backend import vllm_launcher
from marin.inference.vllm_server import _CUDA_NVCC_BOOTSTRAP
from rigging.filesystem.factory import open_url
from rigging.log_setup import configure_logging

from experiments.evaluation.hero_vllm_hillclimb.arm import OUTPUT_ROOT, RELEASES, _overlay

CLUSTER = "cw-us-east-08a"
TEST_ROOT = "/tmp/vllm-fork-tests"
TEST_REQUIREMENTS = ("pytest", "pytest-asyncio", "anthropic", "tblib", "pillow")


def run_fork_tests(release: VllmGpuRelease, tests_tgz: str, output_uri: str, pytest_args: tuple[str, ...]) -> None:
    """Iris entrypoint: extract the fork tests and run pytest in the release's vLLM environment."""
    configure_logging()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    os.makedirs(TEST_ROOT)
    with open_url(tests_tgz, "rb") as source, tarfile.open(fileobj=source, mode="r:gz") as archive:
        archive.extractall(TEST_ROOT, filter="data")
    launcher = vllm_launcher(
        VllmEngineConfig(launcher=VllmLauncherType.CUDA, source=VllmSource.MARIN_FORK, gpu_release=release)
    )
    command = launcher.command()
    head = command[: command.index(_CUDA_NVCC_BOOTSTRAP) + 1]
    extras = [item for requirement in TEST_REQUIREMENTS for item in ("--with", requirement)]
    command = [head[0], *extras, *head[1:], "python", "-m", "pytest", "-p", "no:cacheprovider", *pytest_args]
    log_path = Path(TEST_ROOT) / "pytest.log"
    with log_path.open("wb") as log:
        result = subprocess.run(
            command, env={**os.environ, **launcher.env()}, cwd=TEST_ROOT, stdout=log, stderr=subprocess.STDOUT
        )
    with log_path.open("rb") as source, open_url(f"{output_uri}/pytest.log", "wb") as target:
        shutil.copyfileobj(source, target)
    if result.returncode != 0:
        raise RuntimeError(f"pytest exited {result.returncode}; log at {output_uri}/pytest.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), default="0ac9dcb")
    parser.add_argument("--overlay-wheel", help="FILE:SHA256 of an aarch64 overlay wheel URL")
    parser.add_argument("--tests-tgz", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configure_logging()
    release = RELEASES[args.release]
    if args.overlay_wheel:
        release = _overlay(release, args.overlay_wheel)
    name = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    output_uri = f"{OUTPUT_ROOT}/{name}"
    pytest_args = tuple(arg for arg in args.pytest_args if arg != "--")
    resources = ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="800g", disk="200g", replicas=1, preemptible=False)
    print(f"{name} -> {output_uri} pytest {pytest_args}")
    workspace = Path.cwd()
    with connect_controller(cluster_name=CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=workspace, credentials=endpoint.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_fork_tests, release, args.tests_tgz, output_uri, pytest_args),
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
