# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the distinct Daytona snapshot identities a task cohort provisions.

Shellbox names every Daytona snapshot from one identity string
(``shellbox.backends.daytona.machine._snapshot``)::

    identity = f"{image_source_key(source)}\\0{json.dumps(asdict(Resources(...)), sort_keys=True)}"
    name = f"shellbox-{hashlib.sha256(identity.encode()).hexdigest()[:40]}"

A cohort therefore needs one snapshot per distinct (build context, resources)
pair, and running it within the Daytona budget means that count stays at or
below ``DAYTONA_SAFE_SNAPSHOT_LIMIT``. The helpers here recompute that identity
from pinned task payloads — native release tars and lowered Harbor specs — so a
catalog verdict can cite a measured count instead of an inference.
"""

import base64
import gzip
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field

from daytona import Resources

AGENTIC_DEFAULT_IMAGE_KEY = "shellbox-default-agent"
"""Shared identity key for task machines that use the shellbox default agent image."""

DEFAULT_RESOURCES = json.dumps(asdict(Resources()), sort_keys=True)
"""Resources JSON for tasks whose payload specifies no machine shape."""

IDENTITY_RULE = "shellbox: sha256(image_source_key + NUL + resources-json), first 40 hex characters"


@dataclass(frozen=True)
class ContextFile:
    """One entry of an in-memory Docker build context.

    Directories carry a mode and no content; symlinks carry ``link``; regular
    files carry ``content``.
    """

    name: str
    mode: int
    content: bytes | None = None
    link: str | None = None


def image_source_key_from_context(dockerfile_name: str, files: Iterable[ContextFile]) -> str:
    """Hash a build context exactly like :func:`shellbox.image.image_source_key`.

    The dockerfile name is relative to the context root. Entries hash in Path
    order (per path segment, not plain string order) as ``path\\0mode\\0`` followed
    by the symlink target or file bytes; directories contribute no bytes.
    """
    digest = hashlib.sha256()
    digest.update(f"{dockerfile_name}\0".encode())
    for context_file in sorted(files, key=lambda item: tuple(item.name.split("/"))):
        digest.update(f"{context_file.name}\0".encode())
        digest.update(f"{context_file.mode}\0".encode())
        if context_file.link is not None:
            digest.update(context_file.link.encode())
        elif context_file.content is not None:
            digest.update(context_file.content)
    return f"dockerfile:{digest.hexdigest()}"


def registry_image_key(reference: str) -> str:
    """Return the shellbox identity key for a pinned registry reference."""
    return f"registry:{reference}"


def snapshot_identity(image_key: str, resources_json: str = DEFAULT_RESOURCES) -> str:
    """Return the 40-hex snapshot identity shellbox would name for this pair."""
    return hashlib.sha256(f"{image_key}\0{resources_json}".encode()).hexdigest()[:40]


def resources_json(
    cpu: int | None = None,
    memory_mb: int | None = None,
    storage_mb: int | None = None,
    gpus: int | None = None,
) -> str:
    """Serialize a machine shape the way ``_prepare`` builds ``Resources``."""
    return json.dumps(
        asdict(
            Resources(
                cpu=cpu,
                memory=math.ceil(memory_mb / 1024) if memory_mb is not None else None,
                disk=math.ceil(storage_mb / 1024) if storage_mb is not None else None,
                gpu=gpus or None,
            )
        ),
        sort_keys=True,
    )


def _context_files_from_tar(archive: tarfile.TarFile, root: str) -> list[ContextFile]:
    """Extract the ``root`` subtree of a task tar as build-context entries."""
    members = [member for member in archive.getmembers() if member.name.startswith(root)]
    if not members:
        raise ValueError(f"task tar has no {root} build context")
    files = []
    for member in members:
        relative = member.name[len(root):]
        if not relative:
            continue
        if member.issym():
            files.append(ContextFile(name=relative, mode=member.mode, link=member.linkname))
        elif member.isdir():
            files.append(ContextFile(name=relative, mode=member.mode))
        else:
            files.append(ContextFile(name=relative, mode=member.mode, content=archive.extractfile(member).read()))
    return files


def identities_from_task_tars(task_payloads: Iterable[bytes]) -> Iterator[str]:
    """Yield one snapshot identity per native task tar, resources defaulted."""
    for payload in task_payloads:
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload)) as archive:
            files = _context_files_from_tar(archive, "environment/")
            yield snapshot_identity(image_source_key_from_context("Dockerfile", files))


def _decoded_source(source: Mapping) -> bytes:
    if source.get("kind") == "inline_file":
        return base64.b64decode(source.get("content_base64", ""))
    return json.dumps(source, sort_keys=True).encode()


def _machine_shape(machine: Mapping) -> str:
    return resources_json(
        cpu=machine.get("cpus"),
        memory_mb=machine.get("memory_mb"),
        storage_mb=machine.get("storage_mb"),
        gpus=machine.get("gpus"),
    )


def _lowered_verifier_key(spec: Mapping) -> str:
    resources = (spec.get("task") or {}).get("resources", {})
    items = list(resources.get("verifier", [])) + list(resources.get("all", []))
    prefix = "environment/"
    files = []
    for item in items:
        path = item.get("path", "")
        if not path.startswith(prefix) or item.get("source", {}).get("kind") == "directory":
            continue
        files.append(
            ContextFile(
                name=path[len(prefix):],
                mode=int(item.get("mode", "644"), 8),
                content=_decoded_source(item["source"]),
            )
        )
    if not files:
        return AGENTIC_DEFAULT_IMAGE_KEY
    return image_source_key_from_context("Dockerfile", files)


def identities_from_lowered_specs(specs: Iterable[Mapping]) -> Iterator[str]:
    """Yield snapshot identities for lowered Harbor task specs.

    Each spec contributes its verifier build context (the ``environment/`` files of
    the verifier resource list) with the verifier machine shape, plus the task
    machine, which shares the default agent image and takes its shape from
    ``runtime.task_machine`` when set.
    """
    for spec in specs:
        task = spec.get("task") or {}
        runtime = spec.get("runtime") or {}
        yield snapshot_identity(_lowered_verifier_key(spec), _machine_shape(runtime.get("verifier_machine") or {}))
        docker_image = (task.get("environment_requirements") or {}).get("docker_image")
        task_key = registry_image_key(docker_image) if docker_image else AGENTIC_DEFAULT_IMAGE_KEY
        yield snapshot_identity(task_key, _machine_shape(runtime.get("task_machine") or {}))


@dataclass
class SnapshotIdentityCount:
    """Measured distinct snapshot identities with per-identity cohort sizes."""

    distinct: int
    identity_rule: str = IDENTITY_RULE
    sizes: dict[str, int] = field(default_factory=dict)


def count_identities(identities: Iterable[str]) -> SnapshotIdentityCount:
    """Count distinct identities, largest cohorts first in ``sizes``."""
    sizes = Counter(identities)
    return SnapshotIdentityCount(
        distinct=len(sizes),
        sizes=dict(sorted(sizes.items(), key=lambda item: -item[1])),
    )
