# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conditional reads and writes for one mutable object.

Object storage has no portable compare-and-swap API in fsspec. This module exposes the
small common operation needed by commit pointers: read bytes with an opaque version, then
replace those bytes only if the version is unchanged. Local files use ``flock``, GCS uses
generations, and S3 uses ETags with conditional ``PutObject`` headers.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
from dataclasses import dataclass
from typing import Protocol

import botocore.config
import botocore.session
from botocore.exceptions import ClientError
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from rigging.filesystem import factory
from rigging.filesystem.storage_path import StoragePath

_MAX_READ_ATTEMPTS = 8


class ConditionalWriteError(RuntimeError):
    """The object changed after it was read."""


class UnsupportedConditionalWrite(ValueError):
    """The path's backend cannot provide an atomic conditional write."""


@dataclass(frozen=True)
class VersionedBytes:
    """Bytes read from an object and the opaque backend version that identifies them."""

    data: bytes
    version: str


class ConditionalObject(Protocol):
    """One object supporting compare-and-swap publication."""

    @property
    def path(self) -> str: ...

    def read(self) -> VersionedBytes | None: ...

    def write(self, data: bytes, *, expected_version: str | None) -> str: ...


def _local_version(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


@functools.cache
def _gcs_client() -> storage.Client:
    return storage.Client()


@dataclass(frozen=True)
class LocalConditionalObject:
    """A local object whose readers and writers coordinate through ``flock``."""

    path: str

    def _local_path(self) -> str:
        fs, local_path = factory.url_to_fs(self.path)
        if not getattr(fs, "local_file", False):
            raise UnsupportedConditionalWrite(f"{self.path!r} is not a local path")
        return local_path

    def read(self) -> VersionedBytes | None:
        local_path = self._local_path()
        try:
            with open(local_path, "rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                data = handle.read()
        except FileNotFoundError:
            return None
        return VersionedBytes(data=data, version=_local_version(data))

    def write(self, data: bytes, *, expected_version: str | None) -> str:
        local_path = self._local_path()
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        existed = os.path.exists(local_path)
        with open(local_path, "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            current = handle.read()
            found_version = _local_version(current) if existed or current else None
            if found_version != expected_version:
                raise ConditionalWriteError(
                    f"conditional write failed for {self.path}: expected {expected_version!r}, found {found_version!r}"
                )
            handle.seek(0)
            handle.truncate()
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return _local_version(data)


@dataclass(frozen=True)
class GcsConditionalObject:
    """A GCS object using blob generations as compare-and-swap versions."""

    path: str

    def _bucket_and_key(self):
        parsed = StoragePath(self.path)
        return _gcs_client().bucket(parsed.bucket), parsed.key

    def read(self) -> VersionedBytes | None:
        bucket, key = self._bucket_and_key()
        for _attempt in range(_MAX_READ_ATTEMPTS):
            blob = bucket.get_blob(key)
            if blob is None:
                return None
            assert blob.generation is not None
            generation = blob.generation
            try:
                data = blob.download_as_bytes(if_generation_match=generation)
            except NotFound:
                return None
            except PreconditionFailed:
                continue
            return VersionedBytes(data=data, version=str(generation))
        raise ConditionalWriteError(f"object changed {_MAX_READ_ATTEMPTS} consecutive times while reading {self.path}")

    def write(self, data: bytes, *, expected_version: str | None) -> str:
        bucket, key = self._bucket_and_key()
        blob = bucket.blob(key)
        generation = 0 if expected_version is None else int(expected_version)
        try:
            blob.upload_from_string(data, if_generation_match=generation)
        except PreconditionFailed as exc:
            raise ConditionalWriteError(f"conditional write failed for {self.path}") from exc
        assert blob.generation is not None
        return str(blob.generation)


@dataclass(frozen=True)
class S3ConditionalObject:
    """An S3 object using ETags as compare-and-swap versions."""

    path: str
    endpoint_url: str | None = None

    @staticmethod
    @functools.cache
    def _client(endpoint_url: str | None):
        session = botocore.session.get_session()
        kwargs: dict = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
            kwargs["config"] = botocore.config.Config(s3={"addressing_style": "virtual"})
        return session.create_client("s3", **kwargs)

    def _parts(self) -> tuple[str, str]:
        parsed = StoragePath(self.path)
        return parsed.bucket, parsed.key

    def read(self) -> VersionedBytes | None:
        bucket, key = self._parts()
        try:
            response = self._client(self.endpoint_url).get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        return VersionedBytes(data=response["Body"].read(), version=response["ETag"])

    def write(self, data: bytes, *, expected_version: str | None) -> str:
        client = self._client(self.endpoint_url)
        bucket, key = self._parts()
        condition = {"IfNoneMatch": "*"} if expected_version is None else {"IfMatch": expected_version}
        try:
            response = client.put_object(Bucket=bucket, Key=key, Body=data, **condition)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in (
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "409",
                "412",
            ):
                raise ConditionalWriteError(f"conditional write failed for {self.path}") from exc
            raise
        return response["ETag"]


def conditional_object(path: str) -> ConditionalObject:
    """Return the conditional object implementation for ``path``.

    Generic fsspec schemes are intentionally rejected. A writable commit pointer must not
    silently degrade to last-writer-wins behavior.
    """
    parsed = StoragePath(path)
    if parsed.is_local:
        return LocalConditionalObject(path)
    if parsed.scheme == "gs":
        return GcsConditionalObject(path)
    if parsed.scheme == "s3":
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        return S3ConditionalObject(path, endpoint_url=endpoint_url)
    raise UnsupportedConditionalWrite(f"conditional writes are not supported for {parsed.scheme!r} paths")
