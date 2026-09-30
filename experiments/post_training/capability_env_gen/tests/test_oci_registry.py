import gzip
import io
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from capability_pipeline.oci_artifact import ImageArtifactError, digest
from capability_pipeline.oci_registry import (
    RegistryClient,
    RegistryError,
    _ReadRedirects,
)


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, **headers):
        super().__init__(body)
        self.status, self.headers = status, headers

    def read(self, size=-1):
        assert 0 < size <= 1 << 20, "unbounded registry read"
        return super().read(size)


class Registry:
    def __init__(self, defect=None):
        self.blobs, self.manifests, self.requests = {}, {}, []
        self.defect = defect

    def open(self, request, timeout):
        self.requests.append(request)
        url = urllib.parse.urlsplit(request.full_url)
        method = request.get_method()
        if self.defect == "http_error":
            raise urllib.error.HTTPError(
                "https://secret:password@registry.invalid/?signed=private",
                401,
                "private body",
                {},
                io.BytesIO(b"private body"),
            )
        if method == "POST":
            location = "/v2/tasks/test/blobs/uploads/123?state=abc%2Bdef"
            if self.defect == "foreign_upload":
                location = "https://foreign.invalid/upload?signed=private"
            return Response(status=202, Location=location)
        if method == "PUT":
            body = request.data
            data = body.read() if hasattr(body, "read") else body
            assert int(request.headers["Content-length"]) == len(data)
            checksum = digest(data)
            if "/manifests/" in url.path:
                assert url.path.endswith(checksum)
                self.manifests[checksum] = data
            else:
                assert "state=abc%2Bdef" in url.query
                assert urllib.parse.parse_qs(url.query)["digest"] == [checksum]
                self.blobs[checksum] = data
            header = checksum
            if self.defect == "wrong_put_digest":
                header = "sha256:" + "0" * 64
            return Response(status=201, **{"Docker-Content-Digest": header})
        checksum = url.path.rsplit("/", 1)[-1]
        data = (self.manifests if "/manifests/" in url.path else self.blobs)[checksum]
        if self.defect == "corrupt_readback":
            data = bytes([data[0] ^ 1]) + data[1:]
        if self.defect == "extra_readback":
            data += b"extra"
        if self.defect == "short_readback":
            data = data[:-1]
        return Response(data, **{"Docker-Content-Digest": checksum})


def publish(tmp_path, registry, **overrides):
    payload = gzip.compress(b"frozen root filesystem" * 1000, mtime=0)
    path = tmp_path / "layer.tar.gz"
    path.write_bytes(payload)
    client = RegistryClient("registry.invalid", "tasks/test", "publisher", "private")
    client._opener = registry
    kwargs = {
        "expected_digest": digest(payload),
        "expected_bytes": len(payload),
        "max_uncompressed_bytes": 1 << 20,
        "image_config": {
            "Env": ["PATH=/usr/bin"],
            "WorkingDir": "/workspace",
            "User": "1000",
            "Entrypoint": ["/init"],
            "Cmd": ["--task"],
        },
        "architecture": "amd64",
        "operating_system": "linux",
    }
    kwargs.update(overrides)
    return client.publish_layer(path, **kwargs)


def test_publication_binds_bytes_and_config_without_tag_overwrite(tmp_path):
    registry = Registry()
    result = publish(tmp_path, registry)
    manifest = json.loads(registry.manifests[result["manifest_digest"]])
    config = json.loads(registry.blobs[manifest["config"]["digest"]])
    assert config["config"]["User"] == "1000"
    assert config["config"]["Entrypoint"] == ["/init"]
    assert config["config"]["Cmd"] == ["--task"]
    assert result["image"].endswith("@" + result["manifest_digest"])
    assert result["task_acceptance"] == "not_evaluated"
    assert "private" not in json.dumps(result)
    # Both blob GETs precede manifest publication.
    methods = [r.get_method() for r in registry.requests]
    assert methods == ["POST", "PUT", "POST", "PUT", "GET", "GET", "PUT", "GET"]
    assert publish(tmp_path, Registry())["image"] == result["image"]


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_put_digest",
        "corrupt_readback",
        "extra_readback",
        "short_readback",
    ],
)
def test_no_receipt_or_manifest_after_failed_blob_integrity(tmp_path, defect):
    registry = Registry(defect)
    with pytest.raises(RegistryError):
        publish(tmp_path, registry)
    assert not registry.manifests


def test_upload_location_cannot_receive_credentials_cross_origin(tmp_path):
    registry = Registry("foreign_upload")
    with pytest.raises(RegistryError, match="different origin"):
        publish(tmp_path, registry)
    assert len(registry.requests) == 1


def test_corrupt_source_never_reaches_network(tmp_path):
    registry = Registry()
    with pytest.raises(ImageArtifactError):
        publish(tmp_path, registry, expected_digest="sha256:" + "0" * 64)
    assert not registry.requests


def test_transport_errors_omit_urls_credentials_and_provider_body(tmp_path):
    with pytest.raises(RegistryError) as caught:
        publish(tmp_path, Registry("http_error"))
    assert str(caught.value) == "registry HTTP failure: 401"
    assert caught.value.__suppress_context__


def test_read_redirects_strip_auth_and_reject_plaintext_and_writes():
    handler = _ReadRedirects()
    request = urllib.request.Request(
        "https://registry.invalid/blob", headers={"Authorization": "private"}
    )
    redirected = handler.redirect_request(
        request, None, 307, "", {}, "https://storage.invalid/signed"
    )
    assert not redirected.has_header("Authorization")
    with pytest.raises(RegistryError, match="HTTPS"):
        handler.redirect_request(
            request, None, 302, "", {}, "http://storage.invalid/blob"
        )
    write = urllib.request.Request(
        "https://registry.invalid/upload", b"body", method="PUT"
    )
    with pytest.raises(RegistryError, match="cannot redirect"):
        handler.redirect_request(
            write, None, 307, "", {}, "https://storage.invalid/upload"
        )
