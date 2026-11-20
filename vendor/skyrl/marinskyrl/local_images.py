"""Prepare regional OCI mirrors without changing a task's registry image identity."""

from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.verified_copy import verified_copy_prefix
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.image import ImageCache, PreparedImage, RegistryImage, image_cache_path


class LocalImageMirror(BaseModel):
    """A digest-pinned registry image and its regional OCI copy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(pattern=r"^\S+@sha256:[0-9a-f]{64}$")
    uri: str

    @field_validator("image")
    @classmethod
    def registry_reference(cls, value: str) -> str:
        if "://" in value:
            raise ValueError("An image mirror binds a registry reference, without a transport scheme")
        return value


class LocalMachineSettings(BaseModel):
    """Worker placement and immutable images prepared by the LOCAL backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str | None = None
    image_mirrors: tuple[LocalImageMirror, ...] = ()


def local_machine_factory(settings: LocalMachineSettings) -> LocalMachineFactory:
    """Prepare configured regional images before the LOCAL backend accepts tasks."""
    if settings.image_mirrors and settings.region is None:
        raise ValueError("LOCAL image mirrors require an explicit worker region")
    factory = LocalMachineFactory()
    if settings.region is not None:
        for mirror in settings.image_mirrors:
            cached_mirror_image(factory.image_cache, mirror, region=settings.region)
    return factory


def cached_mirror_image(cache: ImageCache, mirror: LocalImageMirror, *, region: str) -> PreparedImage:
    """Verify a same-region OCI mirror and atomically publish it in the digest cache."""
    source = StoragePath(mirror.uri)
    if source.scheme != "s3":
        raise ValueError("LOCAL image mirrors require regional CoreWeave S3")
    bucket = data_buckets()[source.bucket]
    if bucket.store is not StoreType.COREWEAVE or bucket.signing_region is None:
        raise ValueError("LOCAL image mirrors require a declared CoreWeave bucket region")
    if bucket.signing_region.lower() != region.lower():
        raise ValueError("Stage the OCI image in the LOCAL worker region before use")
    image = RegistryImage(mirror.image)
    digest = mirror.image.rsplit("@", 1)[1]
    cache.directory.mkdir(parents=True, exist_ok=True)
    destination = image_cache_path(cache.directory, digest)
    with FileLock(str(cache.directory / f".{destination.name}.lock")):
        if destination.exists():
            return cache.prepare(image)
        with TemporaryDirectory(prefix=".mirror-", dir=cache.directory) as temporary:
            staged_cache = Path(temporary)
            staged = image_cache_path(staged_cache, digest)
            verified_copy_prefix(str(source), str(staged))
            prepared = ImageCache(staged_cache).prepare(image)
            try:
                prepared.layout.rename(destination)
            except FileExistsError:
                # A registry pull can publish the same digest outside the mirror's lock.
                return cache.prepare(image)
        return cache.prepare(image)
