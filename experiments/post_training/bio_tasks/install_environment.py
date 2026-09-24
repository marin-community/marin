# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install checksum-locked native packages during the image build, before offline solving."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

MICROMAMBA_URL = "https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.3.3-0.tar.bz2"
MICROMAMBA_SHA256 = "e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2"
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_ENVIRONMENT_BYTES = 2 * 1024**3


def download(url: str, destination: Path, algorithm: str, checksum: str, limit: int) -> int:
    """Download one immutable package artifact and reject missing or incorrect bytes."""
    if not url.startswith("https://conda.anaconda.org/"):
        raise ValueError("Package URL must use the public Conda channel host")
    digest = hashlib.new(algorithm)
    size = 0
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(64 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"Package download exceeds byte limit: {destination.name}")
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != checksum:
        raise ValueError(f"Package checksum mismatch: {destination.name}")
    return size


def install(lock: Path, prefix: Path) -> None:
    packages = json.loads(lock.read_text())
    if not packages or len({package["name"] for package in packages}) != len(packages):
        raise ValueError("An environment requires distinct locked packages")
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="bio-install-", dir=prefix.parent) as directory:
        work = Path(directory)
        archive = work / "micromamba.tar.bz2"
        download(MICROMAMBA_URL, archive, "sha256", MICROMAMBA_SHA256, 8 * 1024 * 1024)
        manager = work / "micromamba"
        with tarfile.open(archive, "r:bz2") as source:
            member = source.extractfile("bin/micromamba")
            if member is None:
                raise ValueError("Package manager archive has no executable")
            with member:
                manager.write_bytes(member.read())
        manager.chmod(0o700)
        total = 0
        explicit = ["@EXPLICIT"]
        for package in packages:
            artifact = work / package["url"].rsplit("/", 1)[-1]
            algorithm = "sha256" if package.get("sha256") else "md5"
            if not package.get(algorithm):
                raise ValueError(f"Missing artifact checksum for {package['name']}")
            total += download(
                package["url"],
                artifact,
                algorithm,
                package[algorithm],
                min(MAX_PACKAGE_BYTES, MAX_ENVIRONMENT_BYTES - total),
            )
            if total > MAX_ENVIRONMENT_BYTES:
                raise ValueError("Environment downloads exceed total byte limit")
            explicit.append(artifact.as_uri())
        spec = work / "packages.explicit.txt"
        spec.write_text("\n".join(explicit) + "\n")
        environment = dict(os.environ, MAMBA_ROOT_PREFIX=str(work / "mamba"))
        subprocess.run(
            [str(manager), "create", "--offline", "--no-rc", "-y", "-p", str(prefix), "--file", str(spec)],
            env=environment,
            check=True,
            timeout=600,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    install(args.lock, args.prefix)


if __name__ == "__main__":
    main()
