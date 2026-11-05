# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore the measured runtime in a new worktree without launching experiments."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "8983bec6fdaeb1cd6d0acff041194d2bf53ab679"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise ValueError(f"Destination must not exist: {destination}")
    capsule = ROOT / "reproduction/crossover16k-source.tar.gz"
    manifest = json.loads((ROOT / "reproduction/crossover16k-source-manifest.json").read_text())
    if hashlib.sha256(capsule.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Source capsule checksum differs")
    destination.mkdir(parents=True)
    subprocess.run(
        ["git", "-C", str(ROOT.parents[2]), "worktree", "add", "--detach", str(destination / "marin"), BASE],
        check=True,
    )
    with tarfile.open(capsule) as archive:
        archive.extractall(destination, filter="data")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Restored file checksum differs: {name}")
    with tarfile.open(ROOT / "reproduction/support-source.tar.gz") as archive:
        archive.extractall(destination, filter="data")
    print(destination)


if __name__ == "__main__":
    main()
