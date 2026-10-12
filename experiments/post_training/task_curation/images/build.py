# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build declared environments and record each as an artifact under MARIN_PREFIX.

An ``Environment`` without an image is built once, as the artifact ``images/env-<identity[:16]>``.
Its identity is the SHA-256 of everything that determines the build: the pypi pins or the lock's
bytes, the apt packages, the data, the Python version and platform, the digest-pinned base image,
the files of the runtime packages every environment carries (verifyit), and where the environment
runs. The build stores the environment's hash lock in the artifact. When the environment needs apt
packages the worker image lacks, it also builds an image from a generated Dockerfile, pushes it,
and records its digest. A run whose declaration is unchanged finds the artifact and builds nothing.
"""

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from marin.execution.artifact import Artifact, read_record
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.runtime.local import (
    EXECUTABLE_MODE,
    IDENTITY_CHARS,
    LOCK_FILE,
    RUNTIME_PACKAGES,
    RUNTIME_PTH,
    context_files,
    nltk_packages,
    runtime_files,
)

from experiments.post_training.task_curation.environment import Environment, Placement, placement

# New GHCR packages are created org-internal and the Iris workers pull anonymously, so built
# images are tagged into the public iris-task package until a public task-curation package exists.
DEFAULT_REPOSITORY = "ghcr.io/marin-community/iris-task"
BASE_IMAGE = "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
"""The image built environments start from: iris-task as of 2026-10-07, python:3.12-slim with uv."""
PLATFORM = "linux/amd64"
PYTHON_VERSION = "3.12"
PYTHON_PLATFORM = "x86_64-unknown-linux-gnu"
ENVIRONMENT_ARTIFACT_VERSION = "2026.10.08"
BUILD_COMMAND = "uv run python -m experiments.post_training.task_curation.images --identity {identity}"


class MissingEnvironmentArtifact(RuntimeError):
    """A declaration names an environment whose current identity has no built artifact."""


class EnvironmentArtifact(Artifact):
    """A built environment: its hash lock, stored beside the record, and the image built for it, if any."""

    identity: str
    lock_sha256: str
    apt: list[str]
    data: list[str]
    python: str
    base_image: str
    image: str | None = None
    tag: str | None = None

    @property
    def lock_url(self) -> str:
        return str(StoragePath(self.path) / LOCK_FILE)


@dataclass(frozen=True)
class EnvironmentBuild:
    identity: str
    repository: str
    output_path: str


def environment_identity(environment: Environment) -> dict[str, Any]:
    """Everything that determines what building ``environment`` produces."""
    if environment.image is not None:
        raise ValueError(f"An image environment is used as-is, not built: {environment.image}")
    return {
        "pypi": sorted(environment.pypi),
        "lock_sha256": hashlib.sha256(environment.lock.read_bytes()).hexdigest() if environment.lock else None,
        "apt": sorted(environment.apt),
        "data": sorted(environment.data),
        "python": PYTHON_VERSION,
        "platform": PLATFORM,
        "base_image": BASE_IMAGE,
        "runtime": runtime_files(),
        "placement": placement(environment).value,
    }


def identity_digest(environment: Environment) -> str:
    return hashlib.sha256(canonical_json(environment_identity(environment)).encode()).hexdigest()


def image_tag(repository: str, identity: str) -> str:
    return f"{repository}:task-curation-env-{identity[:IDENTITY_CHARS]}"


def _environment_build(identity: str, ctx: StepContext) -> EnvironmentBuild:
    return EnvironmentBuild(identity=identity, repository=ctx.runtime_arg("repository"), output_path=ctx.output_path)


def environment_artifact(
    environment: Environment, repository: str = DEFAULT_REPOSITORY
) -> ArtifactStep[EnvironmentArtifact]:
    """The ``images/env-<identity[:16]>`` artifact; the repository is where a build pushes, not identity."""
    identity = identity_digest(environment)
    return ArtifactStep(
        name=f"images/env-{identity[:IDENTITY_CHARS]}",
        version=ENVIRONMENT_ARTIFACT_VERSION,
        artifact_type=EnvironmentArtifact,
        run=partial(build_environment, environment),
        build_config=partial(_environment_build, identity),
        runtime_args={"repository": repository},
    )


def compile_lock(requirements: tuple[str, ...], output: Path) -> None:
    """Compile exact pins into a lock with hashes for the workers' Python version and platform."""
    with TemporaryDirectory() as directory:
        source = Path(directory) / "requirements.in"
        source.write_text("".join(f"{requirement}\n" for requirement in requirements))
        # Running from an empty directory keeps uv from reading the settings of a project in the working directory.
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--generate-hashes",
                "--no-header",
                "--python-version",
                PYTHON_VERSION,
                "--python-platform",
                PYTHON_PLATFORM,
                "--output-file",
                str(output),
                str(source),
            ],
            check=True,
            cwd=directory,
        )


def dockerfile(environment: Environment) -> str:
    """A Dockerfile that installs ``environment`` on the base image from a context holding its lock.

    The runtime packages arrive as named build contexts, ``--build-context <name>=<directory>``.
    """
    lines = [f"FROM {BASE_IMAGE}"]
    if environment.apt:
        lines.append(
            "RUN apt-get update"
            f" && apt-get install -y --no-install-recommends {' '.join(sorted(environment.apt))}"
            " && rm -rf /var/lib/apt/lists/*"
        )
    lines.append(f"COPY {LOCK_FILE} /opt/task-curation/{LOCK_FILE}")
    lines.append(f"RUN uv pip sync --system --no-cache --require-hashes /opt/task-curation/{LOCK_FILE}")
    packages = nltk_packages(environment.data)
    if packages:
        # NLTK looks under /usr/local/share/nltk_data without an environment variable.
        lines.append(f"RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data {' '.join(packages)}")
    lines.extend(
        f"COPY --from={package.name} . /opt/task-curation/runtime/{package.name}" for package in RUNTIME_PACKAGES
    )
    lines.append(
        "RUN find /opt/task-curation/runtime -name __pycache__ -prune -exec rm -rf {} +"
        " && echo /opt/task-curation/runtime"
        f' > "$(python3 -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')/{RUNTIME_PTH}"'
        " && python3 -c 'from verifyit.grade import main'"
    )
    return "\n".join(lines) + "\n"


def write_context(environment: Environment, directory: Path) -> None:
    """Write the lock and the Dockerfile an image of ``environment`` builds from into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / LOCK_FILE
    if environment.lock is not None:
        shutil.copyfile(environment.lock, lock)
    else:
        compile_lock(environment.pypi, lock)
    (directory / "Dockerfile").write_text(dockerfile(environment))


def _docker_config() -> dict[str, Any]:
    directory = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker"))
    path = directory / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _require_docker(repository: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Building an image requires the docker CLI with buildx")
    subprocess.run(["docker", "buildx", "version"], check=True, capture_output=True)
    host = repository.split("/", 1)[0]
    config = _docker_config()
    if host in config.get("auths", {}) or host in config.get("credHelpers", {}) or config.get("credsStore"):
        return
    raise RuntimeError(f"Docker has no credentials for {host}; run `docker login {host}` before building")


def _git_tracked(directory: Path, pathspec: str) -> set[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", pathspec], cwd=directory, check=True, capture_output=True, text=True
    ).stdout
    return {path for path in listed.split("\0") if path}


def _require_tracked(environment: Environment) -> None:
    """The files the identity hashes must be exactly git-tracked files, and package files have no exec bits.

    Consumers recompute the identity from a workspace bundle that has no git metadata and no exec bits,
    so an untracked or executable file here would give them a different identity.
    """
    if environment.lock is not None and environment.lock.name not in _git_tracked(
        environment.lock.parent, environment.lock.name
    ):
        raise ValueError(f"{environment.lock} must be tracked by git")
    for package in RUNTIME_PACKAGES:
        files = context_files(package)
        found = {file.path for file in files}
        expected = _git_tracked(package, ".")
        if found != expected:
            raise ValueError(
                f"{package} must contain exactly its git-tracked files; "
                f"untracked: {sorted(found - expected)}, missing: {sorted(expected - found)}"
            )
        executable = [file.path for file in files if file.mode == EXECUTABLE_MODE]
        if executable:
            raise ValueError(f"{package} has executable files {executable}")


def _push_image(context: Path, tag: str) -> str:
    """Build ``context`` for linux/amd64, push it as ``tag``, and return the pushed manifest digest."""
    package_contexts = [f"--build-context={package.name}={package}" for package in RUNTIME_PACKAGES]
    # Without provenance the pushed reference is one platform manifest rather than an attestation index.
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            f"--platform={PLATFORM}",
            "--provenance=false",
            "--push",
            f"--tag={tag}",
            *package_contexts,
            str(context),
        ],
        check=True,
    )
    inspected = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", tag, "--format", "{{json .Manifest}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(inspected)["digest"]


def build_environment(environment: Environment, build: EnvironmentBuild) -> EnvironmentArtifact:
    """Store ``environment``'s hash lock and, when it needs packages the worker image lacks, push its image."""
    if identity_digest(environment) != build.identity:
        raise RuntimeError("The environment changed after its artifact was planned; rerun the build")
    builds_image = placement(environment) == Placement.BUILT_IMAGE
    if builds_image:
        _require_docker(build.repository)
    _require_tracked(environment)
    image = tag = None
    with TemporaryDirectory() as directory:
        context = Path(directory)
        write_context(environment, context)
        lock = (context / LOCK_FILE).read_bytes()
        if builds_image:
            tag = image_tag(build.repository, build.identity)
            image = f"{build.repository}@{_push_image(context, tag)}"
    (StoragePath(build.output_path) / LOCK_FILE).write_bytes(lock, auto_mkdir=True)
    return EnvironmentArtifact(
        path=build.output_path,
        identity=build.identity,
        lock_sha256=hashlib.sha256(lock).hexdigest(),
        apt=sorted(environment.apt),
        data=list(environment.data),
        python=PYTHON_VERSION,
        base_image=BASE_IMAGE,
        image=image,
        tag=tag,
    )


# Built artifacts by path; every declaration that names an environment reads the same record.
_built_environments: dict[str, EnvironmentArtifact] = {}


def built_environment(environment: Environment) -> EnvironmentArtifact:
    """The built artifact for the environment's current identity under MARIN_PREFIX."""
    step = environment_artifact(environment)
    path = step.path()
    if path not in _built_environments:
        if read_record(path) is None:
            identity = step.name.removeprefix("images/env-")
            raise MissingEnvironmentArtifact(
                f"The environment {step.name} is not built ({path} has no artifact); "
                f"run: {BUILD_COMMAND.format(identity=identity)}"
            )
        _built_environments[path] = EnvironmentArtifact.raw_load(path)
    return _built_environments[path]
