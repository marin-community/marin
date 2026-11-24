# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify and prepare Daytona images.

One image corresponds to one image source: a registry reference, or the
contents of a Docker build context. Shellbox derives the image name from the
source, so Harbor and the operational tooling always address the same image.
Sandbox CPU, memory and disk are set on each sandbox, so they are not part of
the identity.
"""

import hashlib
from asyncio import sleep
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path

from daytona import AsyncDaytonaV2Client, DaytonaConflictError, DaytonaError, DaytonaNotFoundError
from daytona.v2 import Image
from rigging.timing import ExponentialBackoff

from shellbox.backends.daytona.errors import DaytonaProvisioningError
from shellbox.backends.daytona.transient import is_ambiguous_daytona_error, retry_backoff, retry_transient
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

IMAGE_PREFIX = "shellbox-"
READY_IMAGE_STATUS = "ready"
FAILED_IMAGE_STATUSES = frozenset({"failed", "error"})


@dataclass(frozen=True)
class ImageRequest:
    """The validated source that identifies one image."""

    source: RegistryImage | DockerfileSource

    @property
    def name(self) -> str:
        return image_name(self.source)

    def as_dict(self) -> dict[str, str]:
        """Return the JSON interchange consumed by ``plan`` and ``ensure``."""
        if isinstance(self.source, RegistryImage):
            return {"name": self.name, "image": self.source.reference}
        return {"name": self.name, "dockerfile": str(self.source.dockerfile)}

    @classmethod
    def from_dict(cls, spec: Mapping[str, object]) -> "ImageRequest":
        """Parse the interchange emitted by :meth:`as_dict`."""
        unknown = set(spec) - {"name", "image", "dockerfile"}
        if unknown:
            raise ValueError(f"Unknown image spec fields: {sorted(unknown)}")
        image = spec.get("image")
        dockerfile = spec.get("dockerfile")
        if (image is None) == (dockerfile is None):
            raise ValueError("Image spec requires exactly one of image or dockerfile")
        if not isinstance(image, str | None) or not isinstance(dockerfile, str | None):
            raise ValueError("Image spec image and dockerfile must be strings")
        source = (
            RegistryImage(image) if image is not None else DockerfileSource(Path(dockerfile).parent, Path(dockerfile))
        )
        request = cls(source=source)
        recorded = spec.get("name")
        if recorded is not None and recorded != request.name:
            raise ValueError(f"Image spec name {recorded} does not match the derived name {request.name}")
        return request


def validate_source(
    source: RegistryImage | DockerfileSource,
) -> RegistryImage | DockerfileSource:
    """Reject Daytona build inputs the provider cannot use."""
    if isinstance(source, RegistryImage):
        return source
    source = DockerfileSource(source.context.resolve(), source.dockerfile.resolve())
    if source.dockerfile.parent != source.context:
        raise UnsupportedMachineSpec("Daytona requires the Dockerfile at the build context root")
    return source


def image_name(source: RegistryImage | DockerfileSource) -> str:
    """Name the image by its source identity.

    A registry image is identified by reference, so a moved tag keeps its name.
    A Docker build context is identified by its current contents.
    """
    return f"{IMAGE_PREFIX}{hashlib.sha256(image_source_key(source).encode()).hexdigest()[:40]}"


def image_request_for_spec(spec: MachineSpec) -> ImageRequest:
    """Derive a machine spec's validated image source."""
    source = spec.source
    if not isinstance(source, (RegistryImage, DockerfileSource)):
        raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")
    return ImageRequest(source=validate_source(source))


async def _submit(client: AsyncDaytonaV2Client, request: ImageRequest) -> None:
    source = request.source
    if isinstance(source, RegistryImage):
        await client.images.pull(source.reference, name=request.name, wait=False)
    else:
        await client.images.build_from_context(source.context, dockerfile=source.dockerfile.name, name=request.name)


async def _submit_once(client: AsyncDaytonaV2Client, request: ImageRequest, backoff: ExponentialBackoff) -> None:
    """Submit the image; a collision or a submission whose answer was lost is left for the next read by name."""
    try:
        await _submit(client, request)
    except DaytonaConflictError:
        pass
    except DaytonaError as error:
        if not is_ambiguous_daytona_error(error):
            raise
        await sleep(backoff.next_interval())


async def ensure_image(client: AsyncDaytonaV2Client, request: ImageRequest) -> str:
    """Create or await the ready image identified by ``request`` and return its name.

    The name is derived from the source, so a name another rollout already
    submitted is the same image and is awaited instead. Every submission is
    read back by name, so a collision or a lost response is not a failure.
    A failed image is submitted once more, which the provider builds or imports
    again: concurrent identical builds can report a transient failure.
    """
    name = request.name
    resubmitted = False
    backoff = retry_backoff()
    while True:
        try:
            image = await retry_transient(lambda: client.images.get(name))
        except DaytonaNotFoundError:
            await _submit_once(client, request, backoff)
            continue
        while image.status not in FAILED_IMAGE_STATUSES | {READY_IMAGE_STATUS}:
            # One server long poll per request; callers bound the whole wait with their own deadline.
            image = await retry_transient(partial(client.images.get, image.id, wait="ready"))
        if image.status == READY_IMAGE_STATUS:
            return name
        if resubmitted:
            raise DaytonaProvisioningError(f"Daytona image {name} is {image.status}: {image.error}")
        resubmitted = True
        await _submit_once(client, request, backoff)


async def list_images(client: AsyncDaytonaV2Client) -> list[Image]:
    """Read every named image in the authenticated project."""
    return [image async for image in client.images.iter()]


class PruneAction(StrEnum):
    WOULD_DELETE = "would delete"
    DELETED = "deleted"
    IN_USE = "in use"
    GONE = "gone"


@dataclass(frozen=True)
class PrunedImage:
    name: str
    created_at: datetime
    action: PruneAction


def prune_candidates(images: list[Image], *, keep: frozenset[str], created_before: datetime | None) -> list[Image]:
    """The Shellbox images that no kept spec names, and that predate ``created_before`` when it is set."""
    return [
        image
        for image in images
        if image.name is not None
        and image.name.startswith(IMAGE_PREFIX)
        and image.name not in keep
        and (created_before is None or image.created_at < created_before)
    ]


async def prune_images(
    client: AsyncDaytonaV2Client,
    *,
    keep: frozenset[str],
    created_before: datetime | None,
    delete: bool,
) -> list[PrunedImage]:
    """Delete, or only list, the Shellbox images that ``prune_candidates`` selects.

    Images that Shellbox did not name are never selected. An image that a sandbox
    still uses is kept and reported as in use.
    """
    pruned = []
    for image in prune_candidates(await list_images(client), keep=keep, created_before=created_before):
        assert image.name is not None
        action = PruneAction.WOULD_DELETE
        if delete:
            try:
                await retry_transient(partial(client.images.delete, image.id))
                action = PruneAction.DELETED
            except DaytonaConflictError:
                action = PruneAction.IN_USE
            except DaytonaNotFoundError:
                action = PruneAction.GONE
        pruned.append(PrunedImage(image.name, image.created_at, action))
    return pruned
