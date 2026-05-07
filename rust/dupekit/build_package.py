#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, publish, and pin marin-dupekit wheels.

Driven by .github/workflows/dupekit-release-wheels.yaml. The matrix legs build
linux and macos wheels; the release leg builds the sdist (separately, so it
isn't duplicated across legs and clobbered by download-artifact merge), then
publishes dist/ and re-pins the root pyproject.toml.

Usage:
    python rust/dupekit/build_package.py --bump --build linux
    python rust/dupekit/build_package.py --bump --build macos
    python rust/dupekit/build_package.py --bump --build sdist
    python rust/dupekit/build_package.py --set-version 0.1.7 --update-pyproject
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

DUPEKIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DUPEKIT_DIR.parent.parent
MANIFEST_PATH = DUPEKIT_DIR / "Cargo.toml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DIST_DIR = REPO_ROOT / "dist"
TOOLS_DIR = REPO_ROOT / ".tools"

ZIG_VERSION = "0.15.2"
# ziglang.org's own server is very slow (<0.1 MB/s); use a community mirror
# from https://ziglang.org/download/community-mirrors.txt instead.
ZIG_DOWNLOAD_BASE = "https://pkg.earth/zig"

# (rust-triple, manylinux-tag) — manylinux is None for native macOS builds.
LINUX_TARGETS: list[tuple[str, str | None]] = [
    ("x86_64-unknown-linux-gnu", "2_28"),
    ("aarch64-unknown-linux-gnu", "2_28"),
]
MAC_TARGETS: list[tuple[str, str | None]] = [
    ("x86_64-apple-darwin", None),
    ("aarch64-apple-darwin", None),
]


def _emit_github_output(key: str, value: str) -> None:
    """Append `key=value` to $GITHUB_OUTPUT when running under GitHub Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


def _zig_platform_key() -> str:
    machine = platform.machine()
    system = platform.system()
    arch_map = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}
    os_map = {"Darwin": "macos", "Linux": "linux"}
    if machine not in arch_map:
        raise ValueError(f"Unsupported architecture: {machine}")
    if system not in os_map:
        raise ValueError(f"Unsupported platform: {system}")
    # Zig >= 0.13 release artifacts use arch-os ordering (e.g. x86_64-linux).
    return f"{arch_map[machine]}-{os_map[system]}"


def _ensure_zig() -> str:
    """Return path to zig binary, downloading from a community mirror if absent."""
    existing = shutil.which("zig")
    if existing:
        return existing

    plat = _zig_platform_key()
    zig_dir = TOOLS_DIR / f"zig-{plat}-{ZIG_VERSION}"
    zig_bin = zig_dir / "zig"
    if zig_bin.exists():
        return str(zig_bin)

    filename = f"zig-{plat}-{ZIG_VERSION}.tar.xz"
    url = f"{ZIG_DOWNLOAD_BASE}/{filename}"
    print(f"Downloading zig {ZIG_VERSION} for {plat} from {ZIG_DOWNLOAD_BASE}...")

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / filename

    last_report = time.monotonic()

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 10:
            return
        last_report = now
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100 / total_size)
            print(f"  zig download: {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.0f}%)")
        else:
            print(f"  zig download: {downloaded / 1e6:.1f} MB")

    urllib.request.urlretrieve(url, archive_path, reporthook=_report)
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(TOOLS_DIR, filter="data")
    archive_path.unlink()

    if not zig_bin.exists():
        print(f"ERROR: Expected zig binary at {zig_bin} after extraction", file=sys.stderr)
        sys.exit(1)
    print(f"zig {ZIG_VERSION} installed to {zig_bin}")
    return str(zig_bin)


def _ensure_maturin() -> str:
    """Return path to maturin, installing via uv tool if missing."""
    existing = shutil.which("maturin")
    if existing:
        return existing

    print("Installing maturin via uv tool...")
    subprocess.run(["uv", "tool", "install", "maturin"], check=True)
    # uv tool installs to a bin dir that may not be on PATH yet.
    tool_bin = subprocess.run(["uv", "tool", "dir", "--bin"], capture_output=True, text=True, check=True).stdout.strip()
    os.environ["PATH"] = f"{tool_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    path = shutil.which("maturin")
    if path is None:
        print("ERROR: maturin not found after installation", file=sys.stderr)
        sys.exit(1)
    return path


def _maturin(*args: str, env: dict[str, str] | None = None) -> None:
    """Run maturin from REPO_ROOT, always pinned to dupekit's manifest."""
    cmd = [_ensure_maturin(), *args, "--manifest-path", str(MANIFEST_PATH)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _read_cargo_version() -> str:
    text = MANIFEST_PATH.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        print("ERROR: Could not parse version from Cargo.toml", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def _write_cargo_version(new_version: str) -> None:
    text = MANIFEST_PATH.read_text()
    new_text, n = re.subn(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        print(f"ERROR: Failed to rewrite version in {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    MANIFEST_PATH.write_text(new_text)


def _bump_patch(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        print(f"ERROR: Cargo.toml version {version!r} is not a semver triple", file=sys.stderr)
        sys.exit(1)
    major, minor, patch = (int(p) for p in parts)
    return f"{major}.{minor}.{patch + 1}"


def bump_cargo_patch_version() -> str:
    current = _read_cargo_version()
    new = _bump_patch(current)
    _write_cargo_version(new)
    print(f"Bumped marin-dupekit version: {current} -> {new}")
    _emit_github_output("version", new)
    return new


def _list_dist_artifacts(label: str) -> None:
    artifacts = sorted(p for p in DIST_DIR.iterdir() if p.is_file())
    print(f"\nBuilt {len(artifacts)} {label}:")
    for f in artifacts:
        print(f"  {f.name}")


def _build_wheels(targets: list[tuple[str, str | None]], use_zig: bool) -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    triples = [t for t, _ in targets]
    print(f"Installing Rust targets: {', '.join(triples)}")
    subprocess.run(["rustup", "target", "add", *triples], check=True)

    env: dict[str, str] | None = None
    if use_zig:
        zig_dir = str(Path(_ensure_zig()).parent)
        env = {**os.environ, "PATH": f"{zig_dir}{os.pathsep}{os.environ.get('PATH', '')}"}

    for triple, manylinux in targets:
        print(f"\n--- Building wheel for {triple} ---")
        args = ["build", "--release", "--out", str(DIST_DIR), "--target", triple]
        if manylinux is not None:
            args += ["--manylinux", manylinux]
        if use_zig:
            args.append("--zig")
        _maturin(*args, env=env)

    _list_dist_artifacts("wheel(s)")


def build_linux_wheels() -> None:
    _build_wheels(LINUX_TARGETS, use_zig=True)


def build_macos_wheels() -> None:
    if platform.system() != "Darwin":
        print("ERROR: macOS wheels require a macOS host (zig can't cross-compile to macOS)", file=sys.stderr)
        sys.exit(1)
    _build_wheels(MAC_TARGETS, use_zig=False)


def build_sdist() -> None:
    # Adds to dist/ rather than resetting it: the release job downloads wheels
    # via download-artifact before invoking us, and we want them in the same
    # directory so `pypa/gh-action-pypi-publish` uploads everything together.
    DIST_DIR.mkdir(exist_ok=True)
    print("\n--- Building sdist ---")
    _maturin("sdist", "--out", str(DIST_DIR))
    _list_dist_artifacts("sdist(s)")


def update_pyproject(version: str) -> None:
    """Pin marin-dupekit to `version` in root pyproject.toml and re-lock."""
    original = PYPROJECT_PATH.read_text()
    new_text, n = re.subn(
        r'"marin-dupekit\s*>=\s*[^"]*"',
        f'"marin-dupekit >= {version}"',
        original,
    )
    if n == 0:
        print("ERROR: pyproject.toml has no marin-dupekit dependency line to update.", file=sys.stderr)
        sys.exit(1)

    if new_text != original:
        PYPROJECT_PATH.write_text(new_text)
        print(f"\n--- Updated pyproject.toml: marin-dupekit pinned to >= {version} ---")
    else:
        print(f"\n--- pyproject.toml already pinned to marin-dupekit >= {version} ---")

    print("\n--- Running uv lock ---")
    subprocess.run(["uv", "lock"], check=True, cwd=REPO_ROOT)


_BUILDERS = {
    "linux": build_linux_wheels,
    "macos": build_macos_wheels,
    "sdist": build_sdist,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--bump",
        action="store_true",
        help="Bump the patch version in rust/dupekit/Cargo.toml.",
    )
    version_group.add_argument(
        "--set-version",
        metavar="VERSION",
        help=(
            "Write VERSION verbatim into rust/dupekit/Cargo.toml. The release "
            "job uses this in the second-stage step to avoid an implicit "
            "re-bump - the first stage emits its bumped version to $GITHUB_OUTPUT."
        ),
    )
    parser.add_argument(
        "--build",
        choices=sorted(_BUILDERS),
        help="Build linux wheels (zig cross-compile), macos wheels (native), or sdist into dist/.",
    )
    parser.add_argument(
        "--update-pyproject",
        action="store_true",
        help="Update root pyproject.toml dependency pin and run uv lock.",
    )
    args = parser.parse_args()

    if not (args.bump or args.set_version or args.build or args.update_pyproject):
        parser.error("nothing to do; pass --bump, --set-version, --build, or --update-pyproject")

    if args.bump:
        version = bump_cargo_patch_version()
    elif args.set_version:
        _write_cargo_version(args.set_version)
        version = args.set_version
        print(f"Set marin-dupekit version: {version}")
    else:
        version = _read_cargo_version()

    print(f"marin-dupekit version: {version}")

    if args.build:
        _BUILDERS[args.build]()

    if args.update_pyproject:
        update_pyproject(version)


if __name__ == "__main__":
    main()
