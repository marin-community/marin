# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run uvx from wheels whose complete bytes match pinned SHA-256 digests."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_wheel(url: str, expected_sha256: str, directory: Path) -> Path:
    """Download a wheel once and return it only when its full digest matches."""
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("wheel SHA-256 must be 64 lowercase hexadecimal characters")
    filename = Path(unquote(urlsplit(url).path)).name
    if not filename.endswith(".whl"):
        raise ValueError(f"wheel URL does not end in .whl: {url}")

    destination = directory / expected_sha256 / filename
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f"{filename}.", suffix=".tmp", delete=False
    ) as output:
        staging = Path(output.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "marin-tpu-vllm"})
        with urllib.request.urlopen(request, timeout=300) as response, staging.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_sha256 = _sha256(staging)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"wheel SHA-256 mismatch for {filename}: {actual_sha256} != {expected_sha256}")
        staging.replace(destination)
        return destination
    finally:
        staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-wheel", nargs=2, required=True, metavar=("URL", "SHA256"))
    parser.add_argument("--with-wheel", nargs=2, action="append", default=[], metavar=("URL", "SHA256"))
    parser.add_argument("uvx_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    uvx_args = args.uvx_args[1:] if args.uvx_args[:1] == ["--"] else args.uvx_args
    if not uvx_args:
        parser.error("uvx arguments are required after --")

    wheel_cache = Path(os.environ.get("UV_CACHE_DIR", tempfile.gettempdir())) / "marin-verified-wheels"
    from_url, from_sha256 = args.from_wheel
    from_wheel = download_verified_wheel(from_url, from_sha256, wheel_cache)
    command = ["uvx", "--from", str(from_wheel)]
    for wheel_url, wheel_sha256 in args.with_wheel:
        command.extend(("--with", str(download_verified_wheel(wheel_url, wheel_sha256, wheel_cache))))
    command.extend(uvx_args)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
