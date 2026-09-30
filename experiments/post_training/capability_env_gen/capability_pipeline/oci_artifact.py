"""Construct OCI metadata only after validating a frozen compressed rootfs.

This module never extracts or executes an archive. Publication, privacy review,
provider reconstruction and task acceptance are separate gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import BinaryIO


class ImageArtifactError(ValueError):
    pass


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True)
class ValidatedLayer:
    compressed_digest: str
    compressed_bytes: int
    diff_id: str
    uncompressed_bytes: int


def validate_layer(
    stream: BinaryIO,
    *,
    expected_digest: str,
    expected_bytes: int,
    max_uncompressed_bytes: int,
) -> ValidatedLayer:
    """Require exact bytes, CRC/trailer and a single complete gzip member.

    Memory stays bounded even for highly compressible input. Concatenated gzip
    members and trailing bytes are rejected instead of silently ignored.
    """
    if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest) is None:
        raise ImageArtifactError("invalid expected layer digest")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ImageArtifactError("expected compressed size must be positive")
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
        raise ImageArtifactError("uncompressed size bound must be positive")
    raw, expanded = hashlib.sha256(), hashlib.sha256()
    raw_size = expanded_size = 0
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        while chunk := stream.read(1 << 20):
            raw_size += len(chunk)
            if raw_size > expected_bytes:
                raise ImageArtifactError("compressed layer exceeds expected size")
            raw.update(chunk)
            if decoder.eof:
                raise ImageArtifactError("trailing bytes after gzip member")
            remaining = chunk
            while remaining:
                output = decoder.decompress(remaining, 1 << 20)
                expanded.update(output)
                expanded_size += len(output)
                if expanded_size > max_uncompressed_bytes:
                    raise ImageArtifactError("uncompressed layer exceeds size bound")
                if decoder.unused_data:
                    raise ImageArtifactError(
                        "trailing bytes or concatenated gzip members"
                    )
                remaining = decoder.unconsumed_tail
    except zlib.error as error:
        raise ImageArtifactError("invalid gzip layer") from error
    if not decoder.eof:
        raise ImageArtifactError("truncated gzip layer")
    if raw_size != expected_bytes:
        raise ImageArtifactError("compressed layer size mismatch")
    raw_digest = "sha256:" + raw.hexdigest()
    if raw_digest != expected_digest:
        raise ImageArtifactError("compressed layer digest mismatch")
    return ValidatedLayer(
        raw_digest, raw_size, "sha256:" + expanded.hexdigest(), expanded_size
    )


def build_oci_metadata(
    layer: ValidatedLayer,
    *,
    image_config: Mapping[str, object],
    architecture: str,
    operating_system: str,
) -> tuple[bytes, bytes]:
    """Preserve reviewed source config verbatim; never invent runtime defaults."""
    # A publisher must obtain these fields from reviewed source provenance. An
    # explicit empty value is different from an unavailable/unexamined field.
    required = {"Env", "WorkingDir", "User", "Entrypoint", "Cmd"}
    if not required.issubset(image_config):
        raise ImageArtifactError("source image config lacks required provenance fields")
    if not isinstance(architecture, str) or not architecture:
        raise ImageArtifactError("source architecture is required")
    if not isinstance(operating_system, str) or not operating_system:
        raise ImageArtifactError("source operating system is required")
    for field in ("Env", "Entrypoint", "Cmd"):
        value = image_config[field]
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ImageArtifactError(f"invalid source config field: {field}")
    for field in ("WorkingDir", "User"):
        if not isinstance(image_config[field], str):
            raise ImageArtifactError(f"invalid source config field: {field}")
    config = canonical_json(
        {
            "architecture": architecture,
            "os": operating_system,
            "config": dict(image_config),
            "rootfs": {"type": "layers", "diff_ids": [layer.diff_id]},
        }
    )
    manifest = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": digest(config),
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": layer.compressed_digest,
                    "size": layer.compressed_bytes,
                }
            ],
        }
    )
    return config, manifest


def verify_blob(data: bytes, *, expected_digest: str, expected_bytes: int) -> None:
    """Validate registry readback, without relying on its digest response header."""
    if len(data) != expected_bytes or digest(data) != expected_digest:
        raise ImageArtifactError("registry blob readback mismatch")
