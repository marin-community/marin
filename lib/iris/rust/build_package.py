#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build marin-iris-native wheels and source distributions.

Without a pre-resolved ``--version``, the release modes choose versions as
follows:

    nightly  -- one patch above the latest stable, with a UTC dev timestamp
    stable   -- requires an explicit version
    manual   -- the declared version plus an explicit revision, for build smokes

CI resolves the version once, then passes it back through ``--version`` so every
wheel and source distribution carries the same value.
"""

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from enum import StrEnum
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


class BuildMode(StrEnum):
    NIGHTLY = "nightly"
    STABLE = "stable"
    MANUAL = "manual"


class PackageOperation(StrEnum):
    RESOLVE = "resolve"
    WHEEL = "wheel"
    SDIST = "sdist"


def _emit_github_output(path: Path | None, key: str, value: str) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as output:
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


def resolve_version(mode: BuildMode, version: str | None, revision: str | None) -> str:
    if version:
        return version
    if mode == BuildMode.STABLE:
        raise ValueError("--mode stable requires --version")
    if mode == BuildMode.NIGHTLY:
        return _nightly_version()
    if mode == BuildMode.MANUAL:
        if revision is None:
            raise ValueError("--mode manual requires --revision when --version is omitted")
        return f"{_read_declared_version()}+{revision[:8]}"
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
    parser.add_argument("--mode", type=BuildMode, choices=tuple(BuildMode), required=True)
    parser.add_argument("--version", help="Pre-resolved version; bypasses mode-specific resolution.")
    parser.add_argument("--revision", help="Commit revision used to resolve a manual build version.")
    parser.add_argument("--github-output", type=Path, help="GitHub Actions output file for the resolved version.")
    parser.add_argument("--build", type=PackageOperation, choices=tuple(PackageOperation), required=True)
    args = parser.parse_args()

    try:
        version = resolve_version(args.mode, args.version, args.revision)
    except ValueError as error:
        parser.error(str(error))

    print(f"marin-iris-native version: {version} (mode={args.mode})")
    _emit_github_output(args.github_output, "version", version)
    if args.build == PackageOperation.RESOLVE:
        return

    _write_versions(version)
    if args.build == PackageOperation.WHEEL:
        build_wheel()
    else:
        build_sdist()


if __name__ == "__main__":
    main()
