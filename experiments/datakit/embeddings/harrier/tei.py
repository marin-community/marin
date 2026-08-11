# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a pool of Iris-managed TEI workers."""

import contextlib
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fray.client import JobHandle
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import default_setup_script
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from rigging.timing import Deadline

from experiments.datakit.embeddings.harrier.config import TEI_REQUEST_BATCH_SIZE

logger = logging.getLogger(__name__)

TEI_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:hopper-latest@"
    "sha256:45dd59a35a1ee98cc5c56548bbd3c9cccf418724b83606b9ae4a11bcfeadb52f"
)
TEI_PORT_NAME = "http"
TEI_PROMETHEUS_PORT_NAME = "metrics"
TEI_MAX_BATCH_TOKENS = 131_072
TEI_MAX_BATCH_REQUESTS = 2_048
TEI_TOKENIZATION_WORKERS = 4
TEI_PAYLOAD_LIMIT = 16_000_000
TEI_READY_TIMEOUT = 600
TEI_READY_POLL_DELAY = 2
TEI_MIN_READY_INSTANCES = 16


@dataclass(frozen=True)
class TeiServiceConfig:
    endpoint_name: str
    model_archive: str
    max_input_tokens: int


def _download_model(config: TeiServiceConfig, root: Path) -> Path:
    archive_path = root / "checkpoint.tar"
    with StoragePath(config.model_archive).open("rb") as source, archive_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    with tarfile.open(archive_path) as archive:
        archive.extractall(root, filter="data")

    config_paths = list(root.glob("*/config.json"))
    if len(config_paths) != 1:
        raise ValueError(f"Expected one model config in the archive, found {len(config_paths)}")
    model_path = config_paths[0].parent
    config_path = model_path / "config.json"
    model_config = json.loads(config_path.read_text())
    model_config["max_position_embeddings"] = config.max_input_tokens
    config_path.write_text(json.dumps(model_config))
    return model_path


def _wait_until_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = Deadline.from_seconds(TEI_READY_TIMEOUT)
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"TEI exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return
        except urllib.error.URLError:
            deadline.raise_if_expired("TEI did not become healthy")
            time.sleep(1)


def run_tei_service(config: TeiServiceConfig) -> None:
    """Run TEI and register its endpoint for the lifetime of the process."""
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("TEI service must run inside an Iris job")
    port = job_info.ports[TEI_PORT_NAME]
    prometheus_port = job_info.ports[TEI_PROMETHEUS_PORT_NAME]

    configure_logging()
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = _download_model(config, Path(temporary_directory))
        process = subprocess.Popen(
            [
                "text-embeddings-router",
                "--model-id",
                str(model_path),
                "--port",
                str(port),
                "--max-batch-tokens",
                str(TEI_MAX_BATCH_TOKENS),
                "--max-batch-requests",
                str(TEI_MAX_BATCH_REQUESTS),
                "--max-client-batch-size",
                str(TEI_REQUEST_BATCH_SIZE),
                "--tokenization-workers",
                str(TEI_TOKENIZATION_WORKERS),
                "--prometheus-port",
                str(prometheus_port),
                "--payload-limit",
                str(TEI_PAYLOAD_LIMIT),
            ]
        )
        try:
            _wait_until_ready(process, port)
            address = f"http://{job_info.advertise_host}:{port}"
            with iris_ctx().registry.registered(config.endpoint_name, address, {"backend": "tei"}):
                return_code = process.wait()
                raise RuntimeError(f"TEI exited with code {return_code}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)


def _setup_script() -> str:
    bootstrap = """set -e
apt-get update
apt-get install -y --no-install-recommends curl python3 python3-venv
curl -LsSf https://astral.sh/uv/0.10.3/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
"""
    workspace = default_setup_script(
        packages=["marin-fray", "marin-iris", "marin-rigging"],
        pip_packages=["s3fs"],
        python_version="3.12",
    )
    return bootstrap + workspace


@contextlib.contextmanager
def tei_service_pool(
    model_archive: str,
    instances: int,
    max_input_tokens: int,
    min_ready_instances: int = TEI_MIN_READY_INSTANCES,
) -> Iterator[str]:
    """Start TEI workers and yield their shared Iris endpoint name."""
    if instances <= 0:
        raise ValueError("instances must be positive")
    if min_ready_instances <= 0:
        raise ValueError("min_ready_instances must be positive")
    if get_job_info() is None:
        raise RuntimeError("tei_service_pool must run inside an Iris job")

    run_id = uuid.uuid4().hex[:8]
    endpoint_name = f"{iris_ctx().job_id}/tei-harrier-{run_id}"
    client = current_client()
    jobs: list[JobHandle] = []
    try:
        for index in range(instances):
            jobs.append(
                client.submit(
                    JobRequest(
                        name=f"tei-harrier-{run_id}-{index:03}",
                        entrypoint=Entrypoint.from_callable(
                            run_tei_service,
                            args=(
                                TeiServiceConfig(
                                    endpoint_name=endpoint_name,
                                    model_archive=model_archive,
                                    max_input_tokens=max_input_tokens,
                                ),
                            ),
                        ),
                        resources=ResourceConfig.with_gpu(
                            "H100",
                            count=1,
                            cpu=0,
                            ram="16g",
                            disk="32g",
                            image=TEI_IMAGE,
                        ),
                        environment=create_environment(
                            docker_image=TEI_IMAGE,
                            setup_scripts=[_setup_script()],
                        ),
                        ports=[TEI_PORT_NAME, TEI_PROMETHEUS_PORT_NAME],
                        max_retries_failure=3,
                        max_retries_preemption=10,
                    )
                )
            )

        ready_target = min(instances, min_ready_instances)
        deadline = Deadline.from_seconds(TEI_READY_TIMEOUT)
        while True:
            ready = len(iris_ctx().resolver.resolve(endpoint_name).endpoints)
            if ready >= ready_target:
                logger.info("TEI pool ready with %d/%d registered instances", ready, instances)
                yield endpoint_name
                return
            if all(JobStatus.finished(job.status()) for job in jobs):
                raise RuntimeError("All TEI jobs finished before the endpoint pool became ready")
            deadline.raise_if_expired(f"Timed out waiting for {ready_target} TEI instances at {endpoint_name}")
            time.sleep(TEI_READY_POLL_DELAY)
    finally:
        for job in jobs:
            try:
                job.terminate()
            except Exception:
                logger.warning("Failed to terminate TEI job job_id=%s", job.job_id, exc_info=True)
