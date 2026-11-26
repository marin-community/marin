# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a frozen lock-only grader environment to its composed toolchain image.

A TaskSpec role whose environment is ``HostImage`` + ``packages_lock`` stores the
lock it was sealed with and nothing else: the machine it names is the worker's
host image, which image backends (Daytona) cannot host at all. The serving wave
resolves such an environment to its composed digest-pinned toolchain image and
delivers the grader machine from it through ``lock_satisfied_by_image``
(047bc68189). This module is the shared binding both the wave and the backend
survey use: it matches the frozen lock against the declared environments'
compiled locks, resolves the composed image for the match, and — when no build
exists — composes one that bakes exactly the frozen lock and the worker's host
programs, so image backends become candidates and the role's disk quota rides
the machine spec's own storage sizing. The stored task bytes, requirements and
grader semantics never change; only the machine source the role runs from does.
"""

import logging
import re
import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path

from marin.execution.artifact import read_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.environments import BUILDABLE_ENVIRONMENTS
from experiments.post_training.task_curation.difficulty_artifacts import REGIONAL_ENVIRONMENT_ROOT
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import (
    DEFAULT_REPOSITORY,
    ENVIRONMENT_ARTIFACT_VERSION,
    IDENTITY_CHARS,
    compile_lock,
    identity_digest,
)

LOGGER = logging.getLogger(__name__)

_LOCK_TEMPDIR_COMMENT = re.compile(r"^\s*#.*requirements\.in")


def normalized_lock(lock: str) -> str:
    """The lock's pinned content, without the compile comment lines that name the staging directory."""
    return "".join(line for line in lock.splitlines(keepends=True) if not _LOCK_TEMPDIR_COMMENT.match(line))


_declared_locks: dict[str, str | None] = {}


def declared_lock(environment: Environment) -> str | None:
    """The declared environment's compiled lock content, or None when its lock cannot compile."""
    identity = identity_digest(environment)
    if identity not in _declared_locks:
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "requirements.lock"
                compile_lock(environment.requirements, output)
                _declared_locks[identity] = normalized_lock(output.read_text())
        except Exception as error:
            LOGGER.warning("Declared environment lock compile failed: %s: %s", type(error).__name__, error)
            _declared_locks[identity] = None
    return _declared_locks[identity]


def read_lock(lock_url: str) -> str | None:
    """The frozen lock's normalized content, or None when it cannot be read."""
    try:
        return normalized_lock(StoragePath(lock_url).read_text())
    except Exception as error:
        LOGGER.warning("Frozen lock read failed for %s: %s: %s", lock_url, type(error).__name__, error)
        return None


def built_image(environment: Environment, *, repository: str = DEFAULT_REPOSITORY) -> str | None:
    """The environment's composed digest-pinned image from its build record, when one exists."""
    identity = identity_digest(environment)
    try:
        record = read_record(
            f"{REGIONAL_ENVIRONMENT_ROOT}/env-{identity[:IDENTITY_CHARS]}/{ENVIRONMENT_ARTIFACT_VERSION}"
        )
    except Exception:
        return None
    image = (record.result or {}).get("image") if record is not None else None
    return image if isinstance(image, str) else None


def matched_environment(lock: str, environments: tuple[Environment, ...] = BUILDABLE_ENVIRONMENTS) -> Environment | None:
    """The declared environment whose compiled lock carries exactly the frozen lock's pins.

    Fragment-bearing environments match first: their composed image is the one a
    build produces and records, so the binding names a machine that exists. A
    worker-placement match (no fragment, no image) is the fallback for locks no
    fragment variant carries; the compose path converts it.
    """
    fallback = None
    for environment in environments:
        declared = declared_lock(environment)
        if declared is None or declared != lock:
            continue
        if environment.dockerfile is not None:
            return environment
        if fallback is None:
            fallback = environment
    return fallback


_resolved_images: dict[str, str | None] = {}


def resolved_toolchain_image(
    lock_url: str, *, environments: tuple[Environment, ...] = BUILDABLE_ENVIRONMENTS
) -> str | None:
    """The composed digest-pinned image for a frozen lock-only environment, when a build exists.

    The survey binds a role through this: without a built image there is no
    binding, and the fit keeps the pre-binding verdict rather than promising a
    machine that was never pushed.
    """
    if lock_url not in _resolved_images:
        _resolved_images[lock_url] = _resolve_image(lock_url, environments)
    return _resolved_images[lock_url]


def _resolve_image(lock_url: str, environments: tuple[Environment, ...]) -> str | None:
    lock = read_lock(lock_url)
    if lock is None:
        return None
    environment = matched_environment(lock, environments)
    if environment is None:
        return None
    return built_image(environment)


@cache
def resolved_toolchain_reference(lock_url: str, *, repository: str = DEFAULT_REPOSITORY) -> str | None:
    """The composed image reference for a frozen lock-only environment, for the serving wave.

    Unlike :func:`resolved_toolchain_image` this falls back to the identity tag
    when no build record carries the image: the wave pins the digest it pulls
    (``pinned_registry_reference``), so a tag resolves at use time.
    """
    from experiments.post_training.task_curation.images.build import image_tag  # noqa: PLC0415

    lock = read_lock(lock_url)
    if lock is None:
        return None
    environment = matched_environment(lock)
    if environment is None:
        return None
    return built_image(environment) or image_tag(repository, identity_digest(environment))


def compose_toolchain_image(lock_url: str, *, repository: str = DEFAULT_REPOSITORY) -> str:
    """Compose, push and record the toolchain image one frozen lock-only environment binds to.

    The matched environment's composed image bakes exactly the frozen lock and the
    worker's host programs. An environment whose placement is the worker composes
    as its toolchain variant — the same compiled lock with the host-programs
    fragment — so the build produces and records a digest-pinned image. Returns
    the digest-pinned reference; the survey and the wave then bind through it.
    """
    from marin.execution.lazy import run  # noqa: PLC0415

    from experiments.post_training.task_curation.datasets.environments import TOOLCHAIN_GRADER_PACKAGES  # noqa: PLC0415
    from experiments.post_training.task_curation.images.build import environment_artifact  # noqa: PLC0415

    lock = read_lock(lock_url)
    if lock is None:
        raise ValueError(f"the frozen lock cannot be read: {lock_url}")
    environment = matched_environment(lock)
    if environment is None:
        raise ValueError("the frozen lock matches no declared environment; compose one from its pins first")
    converted = replace(environment, dockerfile=TOOLCHAIN_GRADER_PACKAGES.dockerfile)
    if built_image(converted, repository=repository) is None:
        configure_coreweave_s3()
        built = run(environment_artifact(converted, repository))
        built = built[0] if isinstance(built, list) else built
        if built.image is None:
            raise ValueError(f"the composed toolchain build recorded no image for {lock_url}")
    image = built_image(converted, repository=repository)
    assert image is not None
    return image
