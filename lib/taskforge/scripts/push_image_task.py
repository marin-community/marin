# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The push job of ``build_image_job.py``: runs in a fresh, pinned Python 3.14 container on CoreWeave.

Nothing the build produced runs here. The job downloads the build job's Iris output archive from
the cluster object store, checks it against the sha256 Iris recorded, extracts ``image.tar``,
fetches and sha256-checks ``crane``, and pushes. The registry credential arrives as a base64
docker config in ``REGISTRY_AUTH``, which Iris redacts from job descriptions; it is written to a
private config directory that only ``crane`` reads, and removed afterwards.

Standard library only. Python 3.14 is required for zstd (Iris archives are ``.tar.zst``).
"""

import base64
import datetime
import hashlib
import hmac
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

CHUNK_BYTES = 1 << 20
IMAGE_MEMBER = "image.tar"
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"
SIGNED_HEADERS = "host;x-amz-content-sha256;x-amz-date"
# The variables crane may see; everything else in the task env (object-store keys) stays out.
CRANE_ENV = ("PATH", "HOME")


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def s3_get_request(uri: str, endpoint: str, region: str, key_id: str, secret: str) -> urllib.request.Request:
    """A SigV4-signed, virtual-hosted GET for ``s3://bucket/key`` (CoreWeave accepts no path style)."""
    parsed = urllib.parse.urlparse(uri)
    assert parsed.scheme == "s3", uri
    endpoint_url = urllib.parse.urlparse(endpoint)
    host = f"{parsed.netloc}.{endpoint_url.netloc}"
    path = urllib.parse.quote(parsed.path, safe="/-_.~")
    now = datetime.datetime.now(datetime.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    scope = f"{amz_date[:8]}/{region}/s3/aws4_request"
    canonical = "\n".join(
        (
            "GET",
            path,
            "",
            f"host:{host}\nx-amz-content-sha256:{UNSIGNED_PAYLOAD}\nx-amz-date:{amz_date}\n",
            SIGNED_HEADERS,
            UNSIGNED_PAYLOAD,
        )
    )
    to_sign = "\n".join(("AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()))
    key = _hmac(f"AWS4{secret}".encode(), amz_date[:8])
    for part in (region, "s3", "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return urllib.request.Request(
        f"{endpoint_url.scheme}://{host}{path}",
        headers={
            "x-amz-content-sha256": UNSIGNED_PAYLOAD,
            "x-amz-date": amz_date,
            "Authorization": (
                f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={SIGNED_HEADERS}, Signature={signature}"
            ),
        },
    )


def download(request: urllib.request.Request | str, target: Path) -> str:
    """Stream ``request`` to ``target`` and return the sha256 hex of the bytes."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=300) as response, target.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def extract_member(archive: Path, name: str, target: Path, mode: Literal["r|gz", "r|zst"]) -> None:
    """Stream ``archive`` and copy its regular-file member ``name`` to ``target``."""
    with tarfile.open(archive, mode) as tar:
        for member in tar:
            if member.name != name:
                continue
            if not member.isreg():
                raise ValueError(f"{name} in {archive} is not a regular file")
            source = tar.extractfile(member)
            assert source is not None
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, CHUNK_BYTES)
            return
    raise ValueError(f"{archive} has no member {name}")


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="push-"))
    crane_archive = work / "crane.tgz"
    crane_sha256 = download(os.environ["CRANE_URL"], crane_archive)
    if crane_sha256 != os.environ["CRANE_SHA256"]:
        raise ValueError(f"crane archive sha256 {crane_sha256} != {os.environ['CRANE_SHA256']}")
    crane = work / "crane"
    extract_member(crane_archive, "crane", crane, "r|gz")
    crane.chmod(0o700)

    archive = work / "outputs.tar.zst"
    request = s3_get_request(
        os.environ["ARCHIVE_URI"],
        os.environ["AWS_ENDPOINT_URL"],
        os.environ["AWS_REGION"],
        os.environ["AWS_ACCESS_KEY_ID"],
        os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    archive_sha256 = download(request, archive)
    if archive_sha256 != os.environ["ARCHIVE_SHA256"]:
        raise ValueError(f"build archive sha256 {archive_sha256} != the {os.environ['ARCHIVE_SHA256']} Iris recorded")
    image = work / IMAGE_MEMBER
    extract_member(archive, IMAGE_MEMBER, image, "r|zst")
    archive.unlink()
    print(f"image tarball {image.stat().st_size} bytes", flush=True)

    config_dir = work / "docker"
    config_dir.mkdir(mode=0o700)
    config = config_dir / "config.json"
    config.touch(mode=0o600)
    config.write_bytes(base64.b64decode(os.environ.pop("REGISTRY_AUTH")))
    try:
        env = {name: os.environ[name] for name in CRANE_ENV if name in os.environ} | {"DOCKER_CONFIG": str(config_dir)}
        subprocess.run((str(crane), "push", str(image), os.environ["DESTINATION"]), env=env, check=True)
    finally:
        config.unlink()
    print("pushed", flush=True)


if __name__ == "__main__":
    main()
