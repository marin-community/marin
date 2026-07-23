#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build marin-iris-native wheels and source distributions.

Driven by .github/workflows/iris-native-release-wheels.yaml. The release modes
match the other native packages:

    nightly  -- one patch above the latest stable, with a UTC dev timestamp
    stable   -- the version supplied by an iris-native-v* tag
    manual   -- the declared version plus the current commit, for build smokes

The resolved version is written to the maturin project and both Cargo packages
inside the ephemeral CI checkout before building.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

RUST_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUST_DIR.parent.parent.parent
DIST_DIR = REPO_ROOT / "dist"
VERSION_PATHS = (
    RUST_DIR / "pyproject.toml",
    RUST_DIR / "Cargo.toml",
    RUST_DIR / "pyext" / "Cargo.toml",
)
PYPI_JSON_URL = "https://pypi.org/pypi/marin-iris-native/json"
_VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]+"', re.MULTILINE)


def _emit_github_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(f"{key}={value}\n")


def _read_declared_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', VERSION_PATHS[0].read_text(), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not read version from {VERSION_PATHS[0]}")
    return match.group(1)


def _write_versions(version: str) -> None:
    for path in VERSION_PATHS:
        text, count = _VERSION_RE.subn(rf'\1"{version}"', path.read_text(), count=1)
        if count != 1:
            raise RuntimeError(f"Could not write version to {path}")
        path.write_text(text)


def _parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.split(".")[:3]
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Not a semver triple: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _bump_patch(version: str) -> str:
    major, minor, patch = _parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _latest_pypi_stable() -> str | None:
    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=15) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    return data.get("info", {}).get("version") or None


def _nightly_version() -> str:
    declared = _read_declared_version()
    published = _latest_pypi_stable()
    base = max((declared, published), key=_parse_semver) if published else declared
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M")
    return f"{_bump_patch(base)}-dev.{stamp}"


def resolve_version(mode: str, version: str | None) -> str:
    if version:
        return version
    if mode == "stable":
        raise ValueError("--mode stable requires --version")
    if mode == "nightly":
        return _nightly_version()
    if mode == "manual":
        revision = os.environ.get("GITHUB_SHA", "")[:8] or "local"
        return f"{_read_declared_version()}+{revision}"
    raise ValueError(f"Unknown build mode: {mode}")


def _maturin(*args: str) -> None:
    subprocess.run(
        ["uvx", "--from", "maturin>=1.5,<2.0", "maturin", *args],
        cwd=RUST_DIR,
        check=True,
    )


def build_wheel() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()
    _maturin("build", "--release", "--out", str(DIST_DIR))


def build_sdist() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    _maturin("sdist", "--out", str(DIST_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nightly", "stable", "manual"), required=True)
    parser.add_argument("--version")
    parser.add_argument("--build", choices=("wheel", "sdist"))
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()

    if not args.resolve_only and args.build is None:
        parser.error("--build is required unless --resolve-only is set")

    try:
        version = resolve_version(args.mode, args.version)
    except ValueError as error:
        parser.error(str(error))

    print(f"marin-iris-native version: {version} (mode={args.mode})")
    _emit_github_output("version", version)
    if args.resolve_only:
        return

    _write_versions(version)
    if args.build == "wheel":
        build_wheel()
    else:
        build_sdist()


if __name__ == "__main__":
    main()
