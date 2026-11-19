# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a DockerBuild in an Iris CPU job and push it to the task registry, pinned by digest.

``IrisImageBuilder`` implements ``taskforge.sandbox.images.ImageBuilder`` with two Iris jobs on
CoreWeave (``TARGET_CLUSTER`` by default, DEFAULT container profile), so the registry credential
never shares a container with the model-authored Dockerfile.
``scripts/build_image_job.py`` runs one from a laptop.

1. The build job runs a pinned ``bash`` image (Iris requires ``bash`` in the task image, so the
   kaniko image itself cannot be the task image). It fetches a sha256-checked ``crane``, exports
   the pinned kaniko debug image's ``/kaniko`` with it, clears the bootstrap filesystem with
   kaniko's ``--pre-cleanup``, and runs kaniko on the context for
   linux/amd64 with ``--no-push``, writing ``image.tar`` into ``$IRIS_OUTPUT_DIR``. kaniko needs no
   daemon, mounts or privileges, only the default capabilities and egress, which CoreWeave's
   DEFAULT profile has (buildah failed there on remount; GCP gVisor has no network and GCP DEFAULT
   drops the capabilities kaniko needs).
   This job never sees the credential. Its only outputs are its exit state and the output archive
   that Iris's own uploader sidecar writes to the cluster's temporary bucket, recording the sha256.
2. The push job runs ``image_push`` in a fresh, pinned Python image. It downloads that
   archive, checks the sha256 Iris recorded, re-fetches and checks ``crane``, and pushes
   ``image.tar`` under a tag unique to this publish. The credential is the job's Iris secret
   ``REGISTRY_AUTH``, delivered as a file, never in its environment or pod spec.
   No code from the build runs in this job; ``crane`` parsing the untrusted tarball is the
   remaining surface.
3. The submitter reads the manifest back by that tag, takes its digest from the bytes, checks the
   digest resolves to the same bytes and the config says linux/amd64, and returns the
   digest-pinned reference (``registry/repository@sha256:...``).

A publish fails in one of two ways. ``BuildFailed`` is the build's own: the build job ran and failed,
kaniko on the Dockerfile and its context or the job's execution timeout, which a Dockerfile change can
fix. ``ImageJobFailed`` is the host's: a job did not schedule, its worker was lost, it was killed, the
build bootstrap or push job failed, Iris kept no output archive, or the registry does not
serve back what was pushed. No Dockerfile change fixes those.

The builder takes the credential at construction and reads no secret store itself, so it runs on
a host without ``gcloud`` or Google credentials, such as a coordinator in an Iris task. The
credential is the Secret Manager secret ``SECRET_NAME``, a JSON object with ``registry``, ``user``
and ``password``. ``registry_credential_json`` reads it with ``gcloud`` and is for laptops only:
``scripts/build_image_job.py`` parses its result in process. A coordinator job receives it as the
Iris secret ``REGISTRY_CREDENTIAL_ENV``, a file under ``rigging.secrets.SECRETS_DIR``
(``iris job run --secret TASKFORGE_REGISTRY_CREDENTIAL REGISTRY_CREDENTIAL_SOURCE``, which the
submitting client resolves), and reads it with ``rigging.secrets.require_secret``.

Iris copies a parent task's explicit environment into every job the task submits, and by default
its mounted secrets too. ``submit_job`` therefore blanks inherited variables and isolates the
secret map: both jobs receive only the CoreWeave storage credentials as Iris secrets, and the push
job alone also receives the registry credential, as the secret ``REGISTRY_AUTH``. The exception is
a parent that carries its CoreWeave keys as explicit environment (``iris job run -e CW_KEY_ID ...``):
Iris copies those into both jobs' environments, where they override the secrets, and blanking them
would leave the jobs with empty storage credentials.

The default registry takes writes with Basic auth only (``WWW-Authenticate: Basic realm="envgen"``), so
there is no short-lived scoped token to hand the push job instead. Exposure that remains: secret
values persist in the controller database, and both image jobs carry the submitter's CoreWeave
storage credentials, which Iris requires on every CoreWeave job and the output uploader uses. The
build's RUN steps can read those storage credentials from the mounted secret files; a key scoped
to the output-archive prefix would narrow that, and none exists yet.
"""

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
from urllib.request import parse_http_list, parse_keqv_list

import httpx
from iris.cli.connect import connect_controller
from iris.client import IrisClient, Job
from iris.client.workload import JobState, TaskOutputArchive, TaskOutputArchiveState
from iris.cluster.client import get_job_info
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from rigging.secrets import COREWEAVE_SECRET_NAMES, SecretBehavior, get_secret
from rigging.timing import Duration

from taskforge.sandbox import image_push
from taskforge.sandbox.images import (
    BUILD_LIMITS,
    REPOSITORY,
    DockerBuild,
    build_context_archive,
    build_digest,
    check_build_limits,
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
PUSH_SOURCE = Path(image_push.__file__)
"""``image_push``, shipped to the push job as a workdir file: its pinned image has no Taskforge."""
BUILD_SETUP_FAILED_EXIT = 125
POLL_INTERVAL = 5.0
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)
REGISTRY_CREDENTIAL_ENV = "TASKFORGE_REGISTRY_CREDENTIAL"
"""The Iris secret name a coordinator job receives the credential JSON under."""
CREDENTIAL_FIELDS = ("registry", "user", "password")
REGISTRY_CREDENTIAL_SOURCE = f"gcp-secret://projects/{SECRET_PROJECT}/secrets/{SECRET_NAME}/versions/{SECRET_VERSION}"
"""The credential's ``rigging.secrets`` source reference, for ``iris job run --secret``."""

# Runs under bash in BUILD_IMAGE. kaniko replaces the container's root filesystem while it builds,
# so it is exec'd last; its exit code is the job's. The EXIT trap marks failures before exec.
BUILD_SCRIPT = r"""
set -euo pipefail
trap 'exit "$BUILD_SETUP_FAILED_EXIT"' EXIT
K=/kaniko
mkdir -p "$K/context" /tmp/kroot "$IRIS_OUTPUT_DIR"
wget -q -O /tmp/crane.tgz "$CRANE_URL"
echo "$CRANE_SHA256  /tmp/crane.tgz" | sha256sum -c - > /dev/null
tar xzf /tmp/crane.tgz -C /tmp crane
/tmp/crane export "$KANIKO_IMAGE" - | tar x -C /tmp/kroot kaniko
cp -a /tmp/kroot/kaniko/. "$K/"
mkdir -p "$K/ssl/certs"
cp /etc/ssl/certs/ca-certificates.crt "$K/ssl/certs/"
tar xzf "/app/$CONTEXT_FILE" -C "$K/context"
# /kaniko is excluded from cleanup; the source context must survive bootstrap removal.
SSL_CERT_DIR="$K/ssl/certs" exec "$K/executor" --context "dir://$K/context" --dockerfile "$K/context/$DOCKERFILE" \
  --pre-cleanup --destination "$DESTINATION" --no-push --tar-path "$IRIS_OUTPUT_DIR/image.tar" \
  --custom-platform linux/amd64
"""


@dataclass(frozen=True)
class RegistryCredential:
    registry: str
    user: str
    password: str = field(repr=False)

    @classmethod
    def from_json(cls, text: str) -> "RegistryCredential":
        """Parse the secret's JSON object. Errors name what is wrong, never the text."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            # The default message carries no input, but its exception's ``doc`` holds all of it.
            raise ValueError(f"the registry credential is not JSON: {error.msg}") from None
        if not isinstance(value, dict) or not all(isinstance(value.get(name), str) for name in CREDENTIAL_FIELDS):
            raise ValueError(f"the registry credential is not a JSON object of strings {', '.join(CREDENTIAL_FIELDS)}")
        return cls(registry=value["registry"], user=value["user"], password=value["password"])

    def docker_config(self) -> str:
        auth = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        return json.dumps({"auths": {self.registry: {"auth": auth}}})


def registry_credential_json() -> str:
    """The capability-publisher credential's JSON text, read from Secret Manager with ``gcloud``.

    For laptops and job launchers only; the library path never calls it (see the module docstring).
    Nothing is echoed: ``gcloud``'s output is captured, and a failure names the command only.
    """
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
    return secret.stdout


class BuildFailed(RuntimeError):
    """The build job ran and failed: kaniko refused the Dockerfile or its context, or the build outran the job."""


class ImageJobFailed(RuntimeError):
    """A publish failed for a host reason, so no change to the Dockerfile helps (see the module docstring)."""


@dataclass(frozen=True)
class Published:
    image: str
    build_digest: str
    build_job_id: str
    push_job_id: str
    seconds_to_built: float
    seconds_total: float


@dataclass(frozen=True)
class ImageJobPolicy:
    """Scheduling priority and deadlines shared by an image build and its push."""

    priority_band: job_pb2.PriorityBand
    timeout: Duration
    scheduling_timeout: Duration


def manifest_readback(credential: RegistryCredential, repository: str, tag: str) -> str:
    """Return the manifest digest ``tag`` resolves to, after checking it is addressable and linux/amd64."""
    base = f"https://{credential.registry}/v2/{repository}"
    auth = (credential.user, credential.password)
    headers = {"Accept": ", ".join(MANIFEST_TYPES)}
    try:
        with httpx.Client(auth=auth, headers=headers, timeout=60, follow_redirects=False) as client:
            by_tag = client.get(f"{base}/manifests/{tag}")
            scheme, _, parameters = by_tag.headers.get("www-authenticate", "").partition(" ")
            if by_tag.status_code == 401 and scheme.lower() == "bearer":
                challenge = parse_keqv_list(parse_http_list(parameters))
                realm = httpx.URL(challenge["realm"])
                registry = httpx.URL(f"https://{credential.registry}")
                if (
                    (realm.scheme, realm.host, realm.port) != (registry.scheme, registry.host, registry.port)
                    or realm.username
                    or realm.password
                ):
                    raise ValueError("Registry token endpoint must share the credential's HTTPS origin")
                authorization = client.get(realm, params={"service": challenge["service"], "scope": challenge["scope"]})
                authorization.raise_for_status()
                payload = authorization.json()
                if not isinstance(payload, dict):
                    raise ValueError("Registry token response must be a JSON object")
                token = payload.get("token") or payload["access_token"]
                if not isinstance(token, str) or not token:
                    raise ValueError("Registry token response must contain a nonempty token")
                client.auth = httpx.Auth()
                client.headers["Authorization"] = f"Bearer {token}"
                by_tag = client.get(f"{base}/manifests/{tag}")
            by_tag.raise_for_status()
            digest = f"sha256:{hashlib.sha256(by_tag.content).hexdigest()}"
            by_digest = client.get(f"{base}/manifests/{digest}")
            by_digest.raise_for_status()
            if by_digest.content != by_tag.content:
                raise ImageJobFailed(f"{digest} does not resolve to the manifest bytes of tag {tag}")
            config_digest = by_tag.json()["config"]["digest"]
            config = client.get(f"{base}/blobs/{config_digest}", follow_redirects=True)
            config.raise_for_status()
            platform = (config.json()["os"], config.json()["architecture"])
            if platform != ("linux", "amd64"):
                raise ImageJobFailed(f"image platform is {platform}, not linux/amd64")
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise ImageJobFailed(f"registry readback failed for {credential.registry}/{repository}:{tag}") from error
    return digest


def output_archive(job: Job) -> TaskOutputArchive:
    """The uploaded output archive of a one-task job's single attempt."""
    [attempt] = job.task(0).status().attempts
    archive = attempt.output_archive
    if archive is None or archive.state != TaskOutputArchiveState.UPLOADED:
        detail = "none" if archive is None else f"{archive.state}: {archive.error_message}"
        raise ImageJobFailed(f"build job {job.job_id} left no uploaded output archive ({detail})")
    return archive


def submit_job(
    client: IrisClient,
    *,
    name: str,
    entrypoint: Entrypoint,
    image: str,
    env: dict[str, str],
    secrets: dict[str, str],
    target_cluster: str,
    policy: ImageJobPolicy,
) -> Job:
    """Submit an image job with explicit values and the storage credentials it needs.

    Inherited environment values are blanked and mounted-secret inheritance is disabled. The job
    receives explicit ``env`` values, and as Iris secrets (files) only ``secrets`` and the
    submitter's CoreWeave storage credentials. Inherited CoreWeave keys are not blanked: Iris lets
    a CoreWeave key in the environment override the secret of the same name, so a blank would
    replace the storage credential with an empty one.
    """
    job_info = get_job_info()
    inherited = {} if job_info is None else {name: "" for name in job_info.env if name not in COREWEAVE_SECRET_NAMES}
    storage = {name: value for name in COREWEAVE_SECRET_NAMES if (value := get_secret(name)) is not None}
    return client.submit(
        entrypoint=entrypoint,
        name=name,
        environment=EnvironmentSpec(
            setup_scripts=[],
            env_vars={**inherited, **env},
            secrets={**storage, **secrets},
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        resources=ResourceSpec(cpu=2, memory=4 * 1024**3, disk=20 * 1024**3),
        task_image=image,
        container_profile=job_pb2.CONTAINER_PROFILE_DEFAULT,
        constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster)],
        scheduling_timeout=policy.scheduling_timeout,
        timeout=policy.timeout,
        priority_band=policy.priority_band,
        max_retries_failure=0,
        max_retries_preemption=0,
    )


class IrisImageBuilder:
    """Publish DockerBuilds through a build job and a separate push job (see the module docstring).

    The jobs are submitted through the controller of the named Iris ``cluster`` or at
    ``controller_url`` (exactly one), run on ``target_cluster``, and push to ``credential.registry``
    with ``credential``. Constructing a builder contacts nothing.
    """

    def __init__(
        self,
        *,
        cluster: str | None = None,
        controller_url: str | None = None,
        target_cluster: str,
        credential: RegistryCredential,
        policy: ImageJobPolicy,
    ):
        if (cluster is None) == (controller_url is None):
            raise ValueError("Specify exactly one Iris cluster or controller URL")
        self.cluster = cluster
        self.controller_url = controller_url
        self.target_cluster = target_cluster
        self.credential = credential
        self.policy = policy

    async def publish(self, build: DockerBuild, repository: str) -> str:
        """Publish ``build`` under ``repository`` and return its digest-pinned reference.

        The reference is ``registry/repository@sha256:...``. A failed build raises ``BuildFailed``,
        and a failure of the host or the registry raises ``ImageJobFailed``.
        """
        # A publish waits through both declared deadlines; keep the event loop free.
        return (await asyncio.to_thread(self.publish_with_timings, build, repository)).image

    def publish_with_timings(self, build: DockerBuild, repository: str) -> Published:
        """Publish ``build`` under ``repository``, blocking until the image is pushed and read back."""
        check_build_limits(build, BUILD_LIMITS)
        if not REPOSITORY.fullmatch(repository):
            raise ValueError(f"not an OCI repository name: {repository!r}")
        credential = self.credential
        digest = build_digest(build)
        run_id = uuid.uuid4().hex[:12]
        # Builds are not reproducible, so every publish gets its own tag; the result is pinned by digest.
        tag = f"build-{digest.removeprefix('sha256:')[:32]}-{run_id}"
        destination = f"{credential.registry}/{repository}:{tag}"
        start = time.monotonic()
        with connect_controller(cluster_name=self.cluster, controller_url=self.controller_url) as endpoint:
            client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
            jobs: list[Job] = []
            try:
                build_job = submit_job(
                    client,
                    name=f"taskforge-image-build-{run_id}",
                    entrypoint=Entrypoint(
                        command=["bash", "-c", BUILD_SCRIPT],
                        workdir_files={CONTEXT_FILE: build_context_archive(build)},
                    ),
                    image=BUILD_IMAGE,
                    env={
                        "CRANE_URL": CRANE_URL,
                        "CRANE_SHA256": CRANE_SHA256,
                        "KANIKO_IMAGE": KANIKO_IMAGE,
                        "CONTEXT_FILE": CONTEXT_FILE,
                        "DOCKERFILE": "Dockerfile",
                        "BUILD_SETUP_FAILED_EXIT": str(BUILD_SETUP_FAILED_EXIT),
                        "DESTINATION": destination,
                    },
                    secrets={},
                    target_cluster=self.target_cluster,
                    policy=self.policy,
                )
                jobs.append(build_job)
                logger.info("submitted build job %s for %s", build_job.job_id, destination)
                wait_succeeded(build_job, BuildFailed, self.policy)
                built = time.monotonic() - start
                archive = output_archive(build_job)
                logger.info("build archive %s (%d bytes)", archive.uri, archive.size_bytes)
                push_job = submit_job(
                    client,
                    name=f"taskforge-image-push-{run_id}",
                    entrypoint=Entrypoint(
                        command=["python3", "-m", PUSH_SOURCE.stem],
                        workdir_files={PUSH_SOURCE.name: PUSH_SOURCE.read_bytes()},
                    ),
                    image=PUSH_IMAGE,
                    env={
                        image_push.CRANE_URL_ENV: CRANE_URL,
                        image_push.CRANE_SHA256_ENV: CRANE_SHA256,
                        image_push.ARCHIVE_URI_ENV: archive.uri,
                        image_push.ARCHIVE_SHA256_ENV: archive.sha256,
                        image_push.DESTINATION_ENV: destination,
                    },
                    secrets={image_push.REGISTRY_AUTH_SECRET: credential.docker_config()},
                    target_cluster=self.target_cluster,
                    policy=self.policy,
                )
                jobs.append(push_job)
                logger.info("submitted push job %s", push_job.job_id)
                wait_succeeded(push_job, ImageJobFailed, self.policy)
            finally:
                for job in jobs:
                    if not is_job_finished(job.state):
                        job.cancel()
                client.shutdown()
        manifest_digest = manifest_readback(credential, repository, tag)
        image = pinned_image(credential.registry, repository, manifest_digest)
        return Published(image, digest, str(build_job.job_id), str(push_job.job_id), built, time.monotonic() - start)


HOST_JOB_STATES = frozenset({JobState.UNSCHEDULABLE, JobState.WORKER_FAILED, JobState.KILLED})
"""How a job ends when the host stopped it: never scheduled, its worker lost, or cancelled from outside.

A job that ran and exited nonzero, or outran its execution timeout, ends ``FAILED``.
"""


def wait_succeeded(job: Job, failed: type[BuildFailed | ImageJobFailed], policy: ImageJobPolicy) -> None:
    """Wait for ``job`` to succeed, raising with Iris's error, which carries the log tail, when it does not.

    Raises:
        ImageJobFailed: host termination, build bootstrap failure, or exceeding the wait deadline.
        BuildFailed | ImageJobFailed: ``failed``, when the job ran and ended ``FAILED``.
    """
    try:
        status = job.wait(
            timeout=policy.scheduling_timeout.to_seconds() + policy.timeout.to_seconds(),
            poll_interval=POLL_INTERVAL,
            raise_on_failure=False,
        )
    except TimeoutError as error:
        raise ImageJobFailed(f"job {job.job_id} did not end within its scheduling and execution timeouts") from error
    if status.state == JobState.SUCCEEDED:
        return
    # The controller's JobStatus carries aggregate counts only: no per-task statuses and a job-level
    # exit code of 0, so the build's exit code is read from its one task.
    host_failed = status.state in HOST_JOB_STATES or job.task(0).status().exit_code == BUILD_SETUP_FAILED_EXIT
    error = ImageJobFailed if host_failed else failed
    raise error(f"job {job.job_id} ended {status.state}: {status.error_message}")
