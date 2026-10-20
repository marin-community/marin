# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build declared environments and record each as an artifact under MARIN_PREFIX.

An ``Environment`` without an image is built once, as the artifact ``images/env-<identity[:16]>``.
Its identity is the SHA-256 of everything that determines the build: the requirements, the apt
packages, the files and the archives they come from, the env, the Dockerfile fragment's bytes, the Python
version and platform, the digest-pinned base image, the files of the runtime packages every
environment carries (verifyit), and where the environment runs. The build compiles the requirements
to a hash lock and materializes the files, and stores both in the artifact. When the environment
names a Dockerfile fragment or needs apt packages the worker image lacks, it also builds an image
from a generated Dockerfile, pushes it, and records its digest. A run whose declaration is unchanged
finds the artifact and builds nothing.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from marin.execution.artifact import Artifact, read_record
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.runtime.local import (
    ENVIRONMENT_MANIFEST,
    EXECUTABLE_MODE,
    FILES_ARCHIVE,
    FILES_DIRECTORY,
    IDENTITY_CHARS,
    LOCK_FILE,
    RUNTIME_PACKAGES,
    RUNTIME_PTH,
    EnvironmentManifest,
    context_files,
    dump_manifest,
    runtime_files,
)

from experiments.post_training.task_curation.environment import Archive, Environment, Placement, placement

# New GHCR packages are created org-internal and the Iris workers pull anonymously, so built
# images are tagged into the public iris-task package until a public task-curation package exists.
DEFAULT_REPOSITORY = "ghcr.io/marin-community/iris-task"
BASE_IMAGE = "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
"""The image built environments start from: iris-task as of 2026-10-07, python:3.12-slim with uv."""
PLATFORM = "linux/amd64"
PYTHON_VERSION = "3.12"
PYTHON_PLATFORM = "x86_64-unknown-linux-gnu"
ENVIRONMENT_ARTIFACT_VERSION = "2026.10.08"
IMAGE_COMPRESSION = "gzip"
EXCLUDE_NEWER = "2026-10-14"
"""Transitive packages resolve among releases uploaded before this date; move it forward with the pins."""
BUILD_COMMAND = "uv run python -m experiments.post_training.task_curation.images --identity {identity}"


class MissingEnvironmentArtifact(RuntimeError):
    """A declaration names an environment whose current identity has no built artifact."""


class EnvironmentArtifact(Artifact):
    """A built environment: its hash lock, placed files and env, stored beside the record, and its image, if any.

    A worker receives only the lock's URL; ``taskcompendium.runtime.local`` reads the environment
    manifest and files archive from beside it. The build inputs are in the identity, so the record does
    not repeat them.
    """

    identity: str
    image: str | None = None

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
    identity = {
        "requirements": sorted(environment.requirements),
        "apt": sorted(environment.apt),
        "files": {
            path: {"url": archive.url, "sha256": archive.sha256, "member": archive.member}
            for path, archive in sorted(environment.files.items())
        },
        "env": dict(sorted(environment.env.items())),
        "dockerfile_sha256": _fragment_sha256(environment),
        "exclude_newer": EXCLUDE_NEWER,
        "python": PYTHON_VERSION,
        "platform": PLATFORM,
        "base_image": BASE_IMAGE,
        "runtime": runtime_files(),
        "placement": placement(environment).value,
    }
    if placement(environment) == Placement.BUILT_IMAGE:
        identity["image_compression"] = IMAGE_COMPRESSION
    return identity


def _fragment_sha256(environment: Environment) -> str | None:
    return hashlib.sha256(environment.dockerfile.read_bytes()).hexdigest() if environment.dockerfile else None


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
    """Compile exact pins into a lock with hashes for the workers' Python version and platform.

    Transitive packages resolve to the newest release before ``EXCLUDE_NEWER``, so two builds of one
    identity resolve the same lock.
    """
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
                "--exclude-newer",
                EXCLUDE_NEWER,
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


def download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _archive_members(data: bytes, member: str) -> dict[str, bytes]:
    """The files of ``member`` (itself, or everything below it) in a zip or tar archive, by path below ``member``."""
    prefix = member.rstrip("/") + "/"
    selected = {}
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir() or (info.filename != member and not info.filename.startswith(prefix)):
                    continue
                selected[info.filename] = archive.read(info)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for info in archive.getmembers():
                if not info.isfile() or (info.name != member and not info.name.startswith(prefix)):
                    continue
                extracted = archive.extractfile(info)
                assert extracted is not None
                selected[info.name] = extracted.read()
    if not selected:
        raise ValueError(f"The archive has no member {member}")
    escaping = [name for name in selected if ".." in PurePosixPath(name).parts]
    if escaping:
        raise ValueError(f"Archive members escape their directory: {escaping}")
    return {name.removeprefix(prefix): content for name, content in selected.items()}


def materialize(archive: Archive, target: Path) -> None:
    """Download ``archive``, verify its digest, and place its member at ``target``, a file or a directory."""
    data = download(archive.url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != archive.sha256:
        raise ValueError(f"{archive.url} has SHA-256 {digest}, not the declared {archive.sha256}")
    for name, content in _archive_members(data, archive.member).items():
        path = target if name == archive.member else target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def dockerfile(environment: Environment) -> str:
    """A Dockerfile that installs ``environment`` on the base image from a context holding its lock and files.

    The runtime packages arrive as named build contexts, ``--build-context <name>=<directory>``.
    """
    lines = [f"FROM {BASE_IMAGE}"]
    if environment.apt:
        lines.append(
            "RUN apt-get update"
            f" && apt-get install -y --no-install-recommends {' '.join(sorted(environment.apt))}"
            " && rm -rf /var/lib/apt/lists/*"
        )
    if environment.files:
        # Before the fragment and the lock sync, so the files' layer stays cached across their changes.
        lines.append(f"COPY {FILES_DIRECTORY}/ /")
    lines.extend(f"ENV {name}={json.dumps(value)}" for name, value in sorted(environment.env.items()))
    if environment.dockerfile is not None:
        # Before the lock sync, so toolchain layers stay cached across lock changes.
        lines.append(environment.dockerfile.read_text().strip())
    lines.append(f"COPY {LOCK_FILE} /opt/task-curation/{LOCK_FILE}")
    lines.append(f"RUN uv pip sync --system --no-cache --require-hashes /opt/task-curation/{LOCK_FILE}")
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
    """Write the lock, the placed files and the Dockerfile an image of ``environment`` builds from into ``directory``.

    The files lie under ``files/`` at their guest paths, so ``COPY files/ /`` places them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    compile_lock(environment.requirements, directory / LOCK_FILE)
    for path, archive in environment.files.items():
        materialize(archive, directory / FILES_DIRECTORY / path.removeprefix("/"))
    (directory / "Dockerfile").write_text(dockerfile(environment))


def _without_mtime(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.mtime = 0
    return member


def files_archive(files: Path) -> bytes:
    """A tar of every file under ``files`` by its path relative to ``files``, as the worker extracts it.

    Member order and timestamps are fixed so the same files give the same archive digest.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(file for file in files.rglob("*") if file.is_file()):
            tar.add(path, arcname=path.relative_to(files).as_posix(), filter=_without_mtime)
    return buffer.getvalue()


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


def _reject_executable_runtime_files() -> None:
    """Iris source bundles drop executable permission bits from runtime packages."""
    for package in RUNTIME_PACKAGES:
        files = context_files(package)
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
            # Export inherited layers with the selected compression too, so OCI
            # consumers can convert the image to Docker's schema2 format.
            f"--output=type=registry,compression={IMAGE_COMPRESSION},force-compression=true",
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
    """Store ``environment``'s lock, files and env and, when it needs what the worker image lacks, push its image."""
    if identity_digest(environment) != build.identity:
        raise RuntimeError("The environment changed after its artifact was planned; rerun the build")
    builds_image = placement(environment) == Placement.BUILT_IMAGE
    if builds_image:
        _require_docker(build.repository)
    _reject_executable_runtime_files()
    image = None
    with TemporaryDirectory() as directory:
        context = Path(directory)
        write_context(environment, context)
        lock = (context / LOCK_FILE).read_bytes()
        files = files_archive(context / FILES_DIRECTORY) if environment.files else None
        if builds_image:
            tag = image_tag(build.repository, build.identity)
            image = f"{build.repository}@{_push_image(context, tag)}"
    output = StoragePath(build.output_path)
    (output / LOCK_FILE).write_bytes(lock, auto_mkdir=True)
    if files is not None:
        (output / FILES_ARCHIVE).write_bytes(files)
    if environment.files or environment.env:
        files_sha256 = hashlib.sha256(files).hexdigest() if files is not None else None
        manifest = EnvironmentManifest(tuple(environment.files), files_sha256, environment.env)
        (output / ENVIRONMENT_MANIFEST).write_text(dump_manifest(manifest))
    return EnvironmentArtifact(path=build.output_path, identity=build.identity, image=image)


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
