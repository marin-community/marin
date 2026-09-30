"""Streaming OCI publication transport for trusted workers.

This is an integrity gate, not a rootfs privacy review or task acceptance gate.
Callers must approve captured bytes/configuration before supplying credentials.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

from .oci_artifact import build_oci_metadata, digest, validate_layer

MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"


class RegistryError(RuntimeError):
    """Sanitized transport failure; never includes a signed URL or credential."""


def _https_origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise RegistryError("registry transport requires an HTTPS URL without userinfo")
    return parts.netloc


class _ReadRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = _https_origin(newurl)
        if req.get_method() not in {"GET", "HEAD"}:
            raise RegistryError("registry writes cannot redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and destination != _https_origin(req.full_url):
            redirected.remove_header("Authorization")
        return redirected


def verify_stream(stream: BinaryIO, expected_digest: str, expected_bytes: int) -> None:
    """Hash a complete readback without buffering the layer in memory."""
    total = 0
    checksum = hashlib.sha256()
    while chunk := stream.read(1 << 20):
        total += len(chunk)
        if total > expected_bytes:
            raise RegistryError("registry readback exceeds expected size")
        checksum.update(chunk)
    if total != expected_bytes or "sha256:" + checksum.hexdigest() != expected_digest:
        raise RegistryError("registry readback digest or size mismatch")


class RegistryClient:
    """Small Basic-auth Distribution client with no credential-bearing receipts."""

    def __init__(
        self, host: str, repository: str, user: str, password: str, *, timeout=3600
    ):
        if _https_origin("https://" + host) != host or any(c in host for c in "/?#"):
            raise RegistryError("invalid registry host")
        if (
            re.fullmatch(
                r"[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*(\/[a-z0-9]+((\.|_|__|-+)[a-z0-9]+)*)*",
                repository,
            )
            is None
        ):
            raise RegistryError("invalid repository")
        if not user or ":" in user or not password:
            raise RegistryError("invalid registry credentials")
        if not math.isfinite(timeout) or timeout <= 0:
            raise RegistryError("invalid registry timeout")
        self.host, self.repository, self.timeout = host, repository, timeout
        self.base = "https://" + host + "/v2/" + repository
        self._authorization = (
            "Basic " + base64.b64encode((user + ":" + password).encode()).decode()
        )
        self._opener = urllib.request.build_opener(_ReadRedirects())

    def _request(self, url, method="GET", body=None, *, size=None, media_type=None):
        if _https_origin(url) != self.host:
            raise RegistryError("authenticated request targets a different origin")
        headers = {"Authorization": self._authorization, "Accept": MANIFEST_TYPE}
        if size is not None:
            headers["Content-Length"] = str(size)
        if media_type:
            headers["Content-Type"] = media_type
        request = urllib.request.Request(url, body, headers, method=method)
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise RegistryError(f"registry HTTP failure: {error.code}") from None
        except (OSError, ValueError):
            raise RegistryError("registry transport failed") from None

    def _upload(self, stream: BinaryIO, checksum: str, size: int) -> None:
        with self._request(
            self.base + "/blobs/uploads/", "POST", b"", size=0
        ) as response:
            if response.status != 202 or not response.headers.get("Location"):
                raise RegistryError("registry did not open a blob upload")
            location = urllib.parse.urljoin(
                self.base + "/blobs/uploads/", response.headers["Location"]
            )
        # Never follow a registry-supplied upload location to another origin.
        if _https_origin(location) != self.host:
            raise RegistryError("blob upload location targets a different origin")
        parts = urllib.parse.urlsplit(location)
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if any(key == "digest" for key, _ in query):
            raise RegistryError("blob upload location contains a digest")
        # Preserve opaque registry state exactly; append only our digest.
        location += ("&" if parts.query else "?") + urllib.parse.urlencode(
            {"digest": checksum}
        )
        with self._request(
            location, "PUT", stream, size=size, media_type="application/octet-stream"
        ) as response:
            if (
                response.status != 201
                or response.headers.get("Docker-Content-Digest") != checksum
            ):
                raise RegistryError("registry did not confirm the blob digest")

    def _verify_readback(self, route: str, checksum: str, size: int) -> None:
        try:
            with self._request(self.base + route + checksum) as response:
                if response.status != 200:
                    raise RegistryError("registry readback did not return success")
                if (
                    route == "/manifests/"
                    and response.headers.get("Docker-Content-Digest") != checksum
                ):
                    raise RegistryError("registry manifest readback header mismatch")
                verify_stream(response, checksum, size)
        except OSError:
            raise RegistryError("registry readback transport failed") from None

    def publish_layer(
        self,
        path: Path,
        *,
        expected_digest: str,
        expected_bytes: int,
        max_uncompressed_bytes: int,
        image_config: Mapping[str, object],
        architecture: str,
        operating_system: str,
    ) -> dict[str, object]:
        """Validate, upload and read back one reviewed rootfs image by digest.

        No tag is overwritten. All readbacks must pass before a receipt is
        returned. A failed upload may leave unreferenced blobs in the registry.
        """
        with path.open("rb") as stream:
            layer = validate_layer(
                stream,
                expected_digest=expected_digest,
                expected_bytes=expected_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            config, manifest = build_oci_metadata(
                layer,
                image_config=image_config,
                architecture=architecture,
                operating_system=operating_system,
            )
            stream.seek(0)
            self._upload(stream, layer.compressed_digest, layer.compressed_bytes)
        config_digest, manifest_digest = digest(config), digest(manifest)
        self._upload(io.BytesIO(config), config_digest, len(config))
        # Verify blobs before making the manifest visible, then read it back too.
        self._verify_readback(
            "/blobs/", layer.compressed_digest, layer.compressed_bytes
        )
        self._verify_readback("/blobs/", config_digest, len(config))
        with self._request(
            self.base + "/manifests/" + manifest_digest,
            "PUT",
            manifest,
            size=len(manifest),
            media_type=MANIFEST_TYPE,
        ) as response:
            if (
                response.status != 201
                or response.headers.get("Docker-Content-Digest") != manifest_digest
            ):
                raise RegistryError("registry did not confirm the manifest digest")
        self._verify_readback("/manifests/", manifest_digest, len(manifest))
        return {
            "schema_version": "capability-oci-transport-v1",
            "state": "integrity_verified",
            "image": self.host + "/" + self.repository + "@" + manifest_digest,
            "manifest_digest": manifest_digest,
            "manifest_bytes": len(manifest),
            "config_digest": config_digest,
            "config_bytes": len(config),
            "layer_digest": layer.compressed_digest,
            "layer_bytes": layer.compressed_bytes,
            "diff_id": layer.diff_id,
            "uncompressed_bytes": layer.uncompressed_bytes,
            "task_acceptance": "not_evaluated",
            "privacy_review": "caller_required",
        }
