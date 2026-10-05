# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a task's DockerBuild in an Iris CPU job and push it to the task registry, pinned by digest.

    uv run scripts/build_image_job.py --context DIR --repository capability-infra/taskforge-smoke

``IrisImageBuilder`` implements ``taskforge.sandbox.images.ImageBuilder`` with two Iris jobs on
CoreWeave (``cw-us-east-02a``, DEFAULT container profile), so the registry credential never
shares a container with the model-authored Dockerfile:

1. The build job runs a pinned ``bash`` image (Iris requires ``bash`` in the task image, so the
   kaniko image itself cannot be the task image). It fetches a sha256-checked ``crane``, exports
   the pinned kaniko debug image's ``/kaniko`` with it, and runs kaniko on the context for
   linux/amd64 with ``--no-push``, writing ``image.tar`` into ``$IRIS_OUTPUT_DIR``. kaniko needs no
   daemon, mounts or privileges, only the default capabilities and egress, which CoreWeave's
   DEFAULT profile has (buildah failed there on remount; GCP gVisor has no network and GCP DEFAULT
   drops the capabilities kaniko needs).
   This job never sees the credential. Its only outputs are its exit state and the output archive
   that Iris's own uploader sidecar writes to the cluster's temporary bucket, recording the sha256.
2. The push job runs ``push_image_task.py`` in a fresh, pinned Python image. It downloads that
   archive, checks the sha256 Iris recorded, re-fetches and checks ``crane``, and pushes
   ``image.tar`` under a tag unique to this publish. The credential travels in the job's
   environment as ``REGISTRY_AUTH``, which Iris redacts from job descriptions and does not log.
   No code from the build runs in this job; ``crane`` parsing the untrusted tarball is the
   remaining surface.
3. The submitter reads the manifest back by that tag, takes its digest from the bytes, checks the
   digest resolves to the same bytes and the config says linux/amd64, and returns the pinned
   ``RegistryImage``.

The registry speaks only Basic auth (``WWW-Authenticate: Basic realm="envgen"``), so there is no
short-lived scoped token to hand the push job instead. Exposure that remains: the push job's pod
spec carries the credential (cluster operators can read it), and every CoreWeave task pod carries
the cluster's object-store keys (``task_env``), which the build's RUN steps can read.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from iris.cli.connect import connect_controller
from iris.client import IrisClient, Job
from iris.client.workload import JobState, TaskOutputArchive, TaskOutputArchiveState
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from rigging.timing import Duration
from taskcompendium.environment import DockerBuild, RegistryImage

from taskforge.sandbox.images import (
    BUILD_LIMITS,
    REPOSITORY,
    build_context_archive,
    build_digest,
    check_build_limits,
    docker_build_from_directory,
    pinned_image,
)

logger = logging.getLogger(__name__)

SECRET_PROJECT = "hai-gcp-models"
SECRET_NAME = "capability-registry-publisher"
SECRET_VERSION = "1"
BUILD_IMAGE = (
    "docker.io/library/bash:5.2-alpine3.22@sha256:a54fb4422b18f05dd3107c36f39d67b26334fda7ec89f4126052b45e228e2f15"
)
# python:3.14-slim-bookworm, linux/amd64 manifest. 3.14 for stdlib zstd.
PUSH_IMAGE = (
    "docker.io/library/python:3.14-slim-bookworm@sha256:d1e795fbdab8a4744432467f32f348c6baa99f07abc05ffde710913f65c8261d"
)
# v1.26.1-debug, linux/amd64 manifest
KANIKO_IMAGE = "ghcr.io/osscontainertools/kaniko@sha256:5403600dff8ed44de5d643b07e4962dfed1078d188dcc52f8719f79ae6a4d482"
CRANE_URL = (
    "https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz"
)
CRANE_SHA256 = "0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0"
TARGET_CLUSTER = "cw-us-east-02a"
CONTEXT_FILE = "context.tar.gz"
PUSH_SCRIPT = Path(__file__).with_name("push_image_task.py")
POLL_INTERVAL = 5.0
JOB_TIMEOUT = 3600
SCHEDULING_TIMEOUT = 900
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)
# Iris copies the submitter's HF_TOKEN and WANDB_API_KEY into every job unless overridden.
BLANKED_ENV = {"HF_TOKEN": "", "WANDB_API_KEY": ""}
# rigging.redaction treats "auth" as a sensitive key word, so Iris redacts this variable's value.
REGISTRY_AUTH_ENV = "REGISTRY_AUTH"

# Runs under bash in BUILD_IMAGE. kaniko replaces the container's root filesystem while it builds,
# so it is exec'd last; its exit code is the job's.
BUILD_SCRIPT = r"""
set -euo pipefail
K=/kaniko
mkdir -p "$K" /tmp/kroot /app/context "$IRIS_OUTPUT_DIR"
wget -q -O /tmp/crane.tgz "$CRANE_URL"
echo "$CRANE_SHA256  /tmp/crane.tgz" | sha256sum -c - > /dev/null
tar xzf /tmp/crane.tgz -C /tmp crane
/tmp/crane export "$KANIKO_IMAGE" - | tar x -C /tmp/kroot kaniko
cp -a /tmp/kroot/kaniko/. "$K/"
mkdir -p "$K/ssl/certs"
cp /etc/ssl/certs/ca-certificates.crt "$K/ssl/certs/"
tar xzf "/app/$CONTEXT_FILE" -C /app/context
SSL_CERT_DIR="$K/ssl/certs" exec "$K/executor" --context dir:///app/context --dockerfile "/app/context$DOCKERFILE" \
  --destination "$DESTINATION" --no-push --tar-path "$IRIS_OUTPUT_DIR/image.tar" --custom-platform linux/amd64
"""


@dataclass(frozen=True)
class RegistryCredential:
    registry: str
    user: str
    password: str = field(repr=False)

    def docker_config(self) -> bytes:
        auth = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        return json.dumps({"auths": {self.registry: {"auth": auth}}}).encode()


def registry_credential() -> RegistryCredential:
    """Read the capability-publisher credential from Secret Manager without echoing it."""
    secret = subprocess.run(
        (
            "gcloud",
            "secrets",
            "versions",
            "access",
            SECRET_VERSION,
            f"--project={SECRET_PROJECT}",
            f"--secret={SECRET_NAME}",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(secret.stdout)
    return RegistryCredential(registry=value["registry"], user=value["user"], password=value["password"])


class BuildFailed(RuntimeError):
    """A build or push job ended without a pushed image."""


@dataclass(frozen=True)
class Published:
    image: RegistryImage
    build_digest: str
    build_job_id: str
    push_job_id: str
    seconds_to_built: float
    seconds_total: float


def manifest_readback(credential: RegistryCredential, repository: str, tag: str) -> str:
    """Return the manifest digest ``tag`` resolves to, after checking it is addressable and linux/amd64."""
    base = f"https://{credential.registry}/v2/{repository}"
    auth = (credential.user, credential.password)
    headers = {"Accept": ", ".join(MANIFEST_TYPES)}
    with httpx.Client(auth=auth, headers=headers, timeout=60, follow_redirects=False) as client:
        by_tag = client.get(f"{base}/manifests/{tag}")
        by_tag.raise_for_status()
        digest = f"sha256:{hashlib.sha256(by_tag.content).hexdigest()}"
        by_digest = client.get(f"{base}/manifests/{digest}")
        by_digest.raise_for_status()
        if by_digest.content != by_tag.content:
            raise BuildFailed(f"{digest} does not resolve to the manifest bytes of tag {tag}")
        config_digest = by_tag.json()["config"]["digest"]
        config = client.get(f"{base}/blobs/{config_digest}", follow_redirects=True)
        config.raise_for_status()
        platform = (config.json()["os"], config.json()["architecture"])
        if platform != ("linux", "amd64"):
            raise BuildFailed(f"image platform is {platform}, not linux/amd64")
    return digest


def output_archive(job: Job) -> TaskOutputArchive:
    """The uploaded output archive of a one-task job's single attempt."""
    [attempt] = job.task(0).status().attempts
    archive = attempt.output_archive
    if archive is None or archive.state != TaskOutputArchiveState.UPLOADED:
        detail = "none" if archive is None else f"{archive.state}: {archive.error_message}"
        raise BuildFailed(f"build job {job.job_id} left no uploaded output archive ({detail})")
    return archive


class IrisImageBuilder:
    """Publish DockerBuilds through a build job and a separate push job (see the module docstring)."""

    def __init__(self, credential: RegistryCredential, *, cluster: str, target_cluster: str):
        self.credential = credential
        self.cluster = cluster
        self.target_cluster = target_cluster

    async def publish(self, build: DockerBuild, repository: str) -> RegistryImage:
        return (await self.publish_with_timings(build, repository)).image

    async def publish_with_timings(self, build: DockerBuild, repository: str) -> Published:
        return await asyncio.to_thread(self._publish, build, repository)

    def _submit(self, client: IrisClient, name: str, entrypoint: Entrypoint, image: str, env: dict[str, str]) -> Job:
        return client.submit(
            entrypoint=entrypoint,
            name=name,
            environment=EnvironmentSpec(setup_scripts=[], env_vars={**BLANKED_ENV, **env}),
            resources=ResourceSpec(cpu=2, memory=4 * 1024**3, disk=20 * 1024**3),
            task_image=image,
            container_profile=job_pb2.CONTAINER_PROFILE_DEFAULT,
            constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=self.target_cluster)],
            scheduling_timeout=Duration.from_seconds(SCHEDULING_TIMEOUT),
            timeout=Duration.from_seconds(JOB_TIMEOUT),
            max_retries_failure=0,
            max_retries_preemption=0,
        )

    def _publish(self, build: DockerBuild, repository: str) -> Published:
        check_build_limits(build, BUILD_LIMITS)
        if not REPOSITORY.fullmatch(repository):
            raise ValueError(f"not an OCI repository name: {repository!r}")
        digest = build_digest(build)
        run_id = uuid.uuid4().hex[:12]
        # Builds are not reproducible, so every publish gets its own tag; the result is pinned by digest.
        tag = f"build-{digest.removeprefix('sha256:')[:32]}-{run_id}"
        destination = f"{self.credential.registry}/{repository}:{tag}"
        start = time.monotonic()
        with connect_controller(cluster_name=self.cluster) as endpoint:
            client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
            jobs: list[Job] = []
            try:
                build_job = self._submit(
                    client,
                    f"taskforge-image-build-{run_id}",
                    Entrypoint(
                        command=["bash", "-c", BUILD_SCRIPT],
                        workdir_files={CONTEXT_FILE: build_context_archive(build)},
                    ),
                    BUILD_IMAGE,
                    {
                        "CRANE_URL": CRANE_URL,
                        "CRANE_SHA256": CRANE_SHA256,
                        "KANIKO_IMAGE": KANIKO_IMAGE,
                        "CONTEXT_FILE": CONTEXT_FILE,
                        "DOCKERFILE": build.dockerfile,
                        "DESTINATION": destination,
                    },
                )
                jobs.append(build_job)
                logger.info("submitted build job %s for %s", build_job.job_id, destination)
                wait_succeeded(build_job)
                built = time.monotonic() - start
                archive = output_archive(build_job)
                logger.info("build archive %s (%d bytes)", archive.uri, archive.size_bytes)
                push_job = self._submit(
                    client,
                    f"taskforge-image-push-{run_id}",
                    Entrypoint(
                        command=["python3", PUSH_SCRIPT.name],
                        workdir_files={PUSH_SCRIPT.name: PUSH_SCRIPT.read_bytes()},
                    ),
                    PUSH_IMAGE,
                    {
                        "CRANE_URL": CRANE_URL,
                        "CRANE_SHA256": CRANE_SHA256,
                        "ARCHIVE_URI": archive.uri,
                        "ARCHIVE_SHA256": archive.sha256,
                        "DESTINATION": destination,
                        REGISTRY_AUTH_ENV: base64.b64encode(self.credential.docker_config()).decode(),
                    },
                )
                jobs.append(push_job)
                logger.info("submitted push job %s", push_job.job_id)
                wait_succeeded(push_job)
            finally:
                for job in jobs:
                    if job.state not in TERMINAL_JOB_STATES:
                        job.cancel()
                client.shutdown()
        manifest_digest = manifest_readback(self.credential, repository, tag)
        image = pinned_image(self.credential.registry, repository, manifest_digest)
        return Published(image, digest, str(build_job.job_id), str(push_job.job_id), built, time.monotonic() - start)


TERMINAL_JOB_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.KILLED, JobState.WORKER_FAILED, JobState.UNSCHEDULABLE}
)


def wait_succeeded(job: Job) -> None:
    """Wait for ``job`` to end; raise ``BuildFailed`` with Iris's error (which carries the log tail) otherwise."""
    status = job.wait(timeout=SCHEDULING_TIMEOUT + JOB_TIMEOUT, poll_interval=POLL_INTERVAL, raise_on_failure=False)
    if status.state != JobState.SUCCEEDED:
        raise BuildFailed(f"job {job.job_id} ended {status.state}: {status.error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--context", type=Path, required=True, help="build context directory with a Dockerfile")
    parser.add_argument("--repository", required=True, help="repository under the registry, e.g. capability-infra/x")
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--target-cluster", default=TARGET_CLUSTER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # blob redirects carry presigned object-store URLs
    build = docker_build_from_directory(args.context, BUILD_LIMITS)
    builder = IrisImageBuilder(registry_credential(), cluster=args.cluster, target_cluster=args.target_cluster)
    published = asyncio.run(builder.publish_with_timings(build, args.repository))
    print(
        json.dumps(
            {
                "image": published.image.reference,
                "build_digest": published.build_digest,
                "build_job_id": published.build_job_id,
                "push_job_id": published.push_job_id,
                "seconds_to_built": round(published.seconds_to_built, 1),
                "seconds_total": round(published.seconds_total, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
