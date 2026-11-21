# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A run-owned, regional OpenResearcher BM25 service with private discovery."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from fray.current_client import current_client
from fray.types import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.cluster_config import require_regional_coreweave_bucket
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import COREWEAVE_SECRET_NAMES, SecretBehavior, require_secret
from rigging.timing import Duration
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.openresearcher import OpenResearcherState

from marin.inference.iris import _terminate_job, _wait_for_endpoint
from marin.rl.retrieval_backend import BM25_RELATIVE_PATH, HEALTH_PATH

RETRIEVAL_PORT_NAME = "retrieval"


@dataclass(frozen=True)
class RetrievalServiceConfig:
    state: OpenResearcherState
    index_uri: str
    region: str
    root: str
    backend_python: str
    max_concurrent_requests: int
    endpoint_name: str


def stage_index(config: RetrievalServiceConfig) -> None:
    """Copy missing regional index files and verify each manifest entry before readiness."""
    if not config.index_uri.startswith("s3://"):
        raise ValueError("Retrieval index must be staged in the run regional bucket")
    source = StoragePath(config.index_uri)
    manifest = (source / "index-manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != config.state.index_manifest_sha256:
        raise ValueError("Index manifest differs from frozen retrieval state")
    root = Path(config.root) / BM25_RELATIVE_PATH
    root.mkdir(parents=True, exist_ok=True)
    for entry in json.loads(manifest):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Index manifest contains an unsafe path")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with (source / "bm25" / entry["path"]).open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        actual = file_sha256(StoragePath(str(target)))
        if target.stat().st_size != entry["bytes"] or actual != entry["sha256"]:
            raise ValueError(f"Index file differs from frozen manifest: {relative}")


def run_retrieval_service(config: RetrievalServiceConfig) -> None:
    """Load once, register privately after health is ready, and serve until parent cancellation."""
    context = iris_ctx()
    info = get_job_info()
    if info is None:
        raise RuntimeError("Retrieval service must run inside its owning Iris job tree")
    if info.worker_region is not None and info.worker_region.lower() != config.region.lower():
        raise ValueError("Retrieval service worker is outside the configured run region")
    require_regional_coreweave_bucket(config.index_uri, config.region)
    configure_coreweave_s3()
    stage_index(config)
    port = context.get_port(RETRIEVAL_PORT_NAME)
    address = f"http://{info.advertise_host}:{port}"
    state_path = Path(config.root) / "service-state.json"
    state_path.write_text(config.state.model_dump_json())
    command = [
        config.backend_python,
        str(Path(__file__).with_name("retrieval_backend.py")),
        "--root",
        config.root,
        "--state",
        str(state_path),
        "--port",
        str(port),
        "--concurrency",
        str(config.max_concurrent_requests),
    ]
    # The immutable image interpreter owns source dependencies; the control venv must not replace them.
    with subprocess.Popen(command, env={**os.environ, "PYTHONPATH": "/opt/offline-search"}) as process:
        try:
            with httpx.Client(timeout=5, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Pinned retrieval backend ended before readiness")
                    try:
                        response = client.get(f"http://127.0.0.1:{port}{HEALTH_PATH}")
                    except httpx.ConnectError:
                        time.sleep(0.1)
                        continue
                    response.raise_for_status()
                    if response.json()["state"] != config.state.model_dump(mode="json"):
                        raise ValueError("Pinned retrieval backend health differs")
                    break
            with context.registry.registered(config.endpoint_name, address):
                if process.wait() != 0:
                    raise RuntimeError("Pinned retrieval backend failed")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@dataclass(frozen=True)
class RetrievalServiceLaunch:
    config: RetrievalServiceConfig
    resources: ResourceConfig
    environment: EnvironmentConfig
    startup_timeout: float
    lifetime: Duration


@contextmanager
def retrieval_service(launch: RetrievalServiceLaunch) -> Iterator[str]:
    """Yield the ready private HTTP endpoint of a service owned by the current RL run."""
    if launch.resources.target_cluster is not None:
        raise ValueError("Retrieval service must inherit the RL run cluster")
    if launch.resources.regions is not None and tuple(launch.resources.regions) != (launch.config.region,):
        raise ValueError("Retrieval service placement must match its run region")
    resources = replace(launch.resources, regions=(launch.config.region,))
    storage = {
        name: launch.environment.secrets[name] if name in launch.environment.secrets else require_secret(name)
        for name in COREWEAVE_SECRET_NAMES
    }
    environment = replace(
        launch.environment,
        secrets={**launch.environment.secrets, **storage},
        secret_behavior=SecretBehavior.ISOLATE,
    )
    job = current_client().submit(
        JobRequest(
            name=launch.config.endpoint_name,
            entrypoint=Entrypoint.from_callable(run_retrieval_service, args=(launch.config,)),
            resources=resources,
            environment=environment,
            timeout=launch.lifetime,
            max_retries_failure=0,
            max_retries_preemption=0,
            ports=(RETRIEVAL_PORT_NAME,),
        ),
        adopt_existing=False,
    )
    try:
        address, _ = _wait_for_endpoint(job, launch.config.endpoint_name, launch.startup_timeout)
        response = httpx.get(address + HEALTH_PATH, timeout=launch.startup_timeout, trust_env=False)
        response.raise_for_status()
        if response.json()["state"] != launch.config.state.model_dump(mode="json"):
            raise ValueError("Retrieval service health differs from the run pinned state")
        yield address
    finally:
        _terminate_job(job)
