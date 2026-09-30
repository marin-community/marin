"""Validate published OCI launch metadata and derive provider build recipes.

Daytona does not preserve an ENTRYPOINT inherited by a FROM-only Dockerfile in
its snapshot launch metadata.  The catalog binds reviewed image references to
their exact canonical OCI config and manifest bytes so runtime consumers can
restate that configuration without registry credentials.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .oci_artifact import canonical_json, digest

DEFAULT_CATALOG = Path(__file__).with_name("published_image_metadata.json")
CATALOG_SCHEMA = "capability-published-image-runtime-metadata-v1"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
LAYER_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
_REFERENCE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


class ImageRuntimeMetadataError(ValueError):
    """Published runtime metadata is malformed or cryptographically inconsistent."""


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ImageRuntimeMetadataError("runtime metadata catalog is unreadable") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != CATALOG_SCHEMA
        or document.get("canonical_encoding") != "RFC8259 sorted compact UTF-8"
        or not isinstance(document.get("images"), dict)
    ):
        raise ImageRuntimeMetadataError("runtime metadata catalog has the wrong schema")
    return document["images"]


def _validated_config(reference: str, record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"config", "manifest"}:
        raise ImageRuntimeMetadataError("published image metadata record is malformed")
    config, manifest = record["config"], record["manifest"]
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise ImageRuntimeMetadataError("published OCI metadata must be JSON objects")
    config_bytes, manifest_bytes = canonical_json(config), canonical_json(manifest)
    expected_manifest_digest = reference.rsplit("@", 1)[1]
    if digest(manifest_bytes) != expected_manifest_digest:
        raise ImageRuntimeMetadataError("published manifest digest does not match reference")
    if (
        manifest.get("schemaVersion") != 2
        or manifest.get("mediaType") != MANIFEST_TYPE
        or set(manifest) != {"schemaVersion", "mediaType", "config", "layers"}
    ):
        raise ImageRuntimeMetadataError("published OCI manifest is malformed")
    descriptor = manifest.get("config")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("mediaType") != CONFIG_TYPE
        or descriptor.get("size") != len(config_bytes)
        or descriptor.get("digest") != digest(config_bytes)
    ):
        raise ImageRuntimeMetadataError("published OCI config descriptor disagrees")
    layers = manifest.get("layers")
    if (
        not isinstance(layers, list)
        or len(layers) != 1
        or not isinstance(layers[0], dict)
        or layers[0].get("mediaType") != LAYER_TYPE
        or not isinstance(layers[0].get("size"), int)
        or layers[0]["size"] <= 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", layers[0].get("digest", "")) is None
    ):
        raise ImageRuntimeMetadataError("published OCI layer descriptor is malformed")
    runtime = config.get("config")
    rootfs = config.get("rootfs")
    if (
        config.get("architecture") != "amd64"
        or config.get("os") != "linux"
        or not isinstance(runtime, dict)
        or not isinstance(rootfs, dict)
        or rootfs.get("type") != "layers"
        or not isinstance(rootfs.get("diff_ids"), list)
        or len(rootfs["diff_ids"]) != len(layers)
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value or "") is None
            for value in rootfs["diff_ids"]
        )
    ):
        raise ImageRuntimeMetadataError("published OCI image config is malformed")
    if not {"Entrypoint", "Cmd"}.issubset(runtime):
        raise ImageRuntimeMetadataError("published OCI launch config is incomplete")
    for key in ("Entrypoint", "Cmd"):
        value = runtime[key]
        if value is not None and (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ImageRuntimeMetadataError(f"published OCI {key} is malformed")
    return runtime


def derive_daytona_recipe(
    reference: str, *, catalog_path: Path | None = None
) -> str:
    """Return a Dockerfile that preserves verified launch metadata for Daytona.

    References absent from the catalog retain the legacy FROM-only recipe.
    Cataloged references fail closed on any metadata or digest inconsistency.
    """
    if not isinstance(reference, str) or _REFERENCE.fullmatch(reference) is None:
        raise ImageRuntimeMetadataError("image reference must be canonical and digest-only")
    images = _load_catalog(catalog_path or DEFAULT_CATALOG)
    record = images.get(reference)
    if record is None:
        return f"FROM {reference}\n"
    runtime = _validated_config(reference, record)
    recipe = f"FROM {reference}\n"
    if runtime["Entrypoint"] is not None:
        recipe += "ENTRYPOINT " + json.dumps(runtime["Entrypoint"], separators=(",", ":")) + "\n"
    if runtime["Cmd"] is not None:
        recipe += "CMD " + json.dumps(runtime["Cmd"], separators=(",", ":")) + "\n"
    return recipe
