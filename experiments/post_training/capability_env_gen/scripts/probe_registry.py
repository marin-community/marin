#!/usr/bin/env python3
"""Push and read back a harmless, deterministic OCI fixture. Run on CoreWeave.

Credentials are read from REGISTRY_CREDENTIALS_FILE (JSON: registry/user/password).
Only sanitized evidence is printed; no credentials or presigned URLs are retained.
This fixture has one text file and is not an executable task environment.
"""

import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime


class StripCrossOriginAuth(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        self.cross_origin_redirects = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing non-HTTPS redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
            self.cross_origin_redirects.append(
                {
                    "destination_host": urllib.parse.urlsplit(newurl).netloc,
                    "authorization_present": redirected.has_header("Authorization"),
                }
            )
        return redirected


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def main():
    with open(os.environ["REGISTRY_CREDENTIALS_FILE"]) as stream:
        credentials = json.load(stream)
    host = credentials["registry"]
    repo = os.environ.get("REGISTRY_PROBE_REPOSITORY", "capability-infra/oci-smoke")
    root = "https://" + host
    base = root + "/v2/" + repo
    auth = "Basic " + base64.b64encode((credentials["user"] + ":" + credentials["password"]).encode()).decode()
    redirects = StripCrossOriginAuth()
    opener = urllib.request.build_opener(redirects)

    def request(url, method="GET", body=None, content_type=None, authenticated=True):
        assert urllib.parse.urlsplit(url).scheme == "https"
        headers = {"Accept": "application/vnd.oci.image.manifest.v1+json"}
        if authenticated:
            assert urllib.parse.urlsplit(url).netloc == host
            headers["Authorization"] = auth
        if content_type:
            headers["Content-Type"] = content_type
        return opener.open(urllib.request.Request(url, body, headers, method=method), timeout=120)

    anonymous = {}
    for path in ("/v2/", "/v2/" + repo + "/manifests/v1"):
        try:
            with request(root + path, authenticated=False) as response:
                anonymous[path] = response.status
        except urllib.error.HTTPError as exc:
            anonymous[path] = exc.code
    assert set(anonymous.values()) == {401}
    with request(root + "/v2/") as response:
        authenticated_health = response.status
    assert authenticated_health == 200

    buffer = io.BytesIO()
    payload = b"Capability environment registry transport fixture. No task data.\n"
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo("registry-probe.txt")
        info.size = len(payload)
        info.mode = 0o644
        info.mtime = 0
        tar.addfile(info, io.BytesIO(payload))
    uncompressed = buffer.getvalue()
    layer = gzip.compress(uncompressed, mtime=0)
    config = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {},
            "rootfs": {"type": "layers", "diff_ids": [digest(uncompressed)]},
            "history": [{"created_by": "capability registry transport fixture"}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    def upload(blob):
        with request(base + "/blobs/uploads/", "POST", b"") as response:
            location = urllib.parse.urljoin(root, response.headers["Location"])
        separator = "&" if "?" in location else "?"
        with request(
            location + separator + "digest=" + digest(blob), "PUT", blob, "application/octet-stream"
        ) as response:
            assert response.status == 201

    upload(layer)
    upload(config)
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": digest(config),
                "size": len(config),
            },
            "layers": [
                {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": digest(layer), "size": len(layer)}
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected = digest(manifest)
    with request(base + "/manifests/v1", "PUT", manifest, "application/vnd.oci.image.manifest.v1+json") as response:
        put_digest = response.headers["Docker-Content-Digest"]
        assert response.status == 201 and put_digest == expected
    with request(base + "/manifests/" + expected) as response:
        readback = response.read()
        readback_digest = response.headers["Docker-Content-Digest"]
    assert readback == manifest and readback_digest == expected
    for blob in (layer, config):
        with request(base + "/blobs/" + digest(blob)) as response:
            fetched = response.read()
        assert fetched == blob and digest(fetched) == digest(blob)
    assert digest(gzip.decompress(layer)) == json.loads(config)["rootfs"]["diff_ids"][0]
    assert not any(item["authorization_present"] for item in redirects.cross_origin_redirects)
    print(
        json.dumps(
            {
                "schema": "registry-transport-probe-v1",
                "passed": True,
                "completed_at": datetime.now(UTC).isoformat(),
                "registry": host,
                "repository": repo,
                "anonymous_status": anonymous,
                "authenticated_health_status": authenticated_health,
                "manifest_local_sha256": expected,
                "manifest_put_digest": put_digest,
                "manifest_readback_digest": readback_digest,
                "image": host + "/" + repo + "@" + expected,
                "config_digest": digest(config),
                "config_bytes": len(config),
                "layer_digest": digest(layer),
                "layer_bytes": len(layer),
                "uncompressed_layer_digest": digest(uncompressed),
                "manifest_config_layer_byte_equality": True,
                "cross_origin_redirects": redirects.cross_origin_redirects,
                "fixture_only_not_executable": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - sanitize credential-bearing transport errors
        # URLs may contain storage signatures, so never emit exception text.
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}))
        raise SystemExit(1) from None
