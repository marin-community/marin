#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, publish, and pin marin-dupekit wheels for all platforms.

End-to-end release flow (driven by .github/workflows/dupekit-release-wheels.yaml):

1. Bump the patch version in rust/dupekit/Cargo.toml (PyPI disallows re-uploads
   at the same version). The matrix legs each bump independently to the same
   value because the bump is `current + 1`.
2. Build wheels (Linux x86_64 + aarch64 via maturin + zig; macOS x86_64 +
   arm64 natively) into dist/.
3. Build the sdist exactly once in the release job (--sdist-only) to avoid
   duplicate sdists across matrix legs clobbering on download-artifact merge.
4. Publish dist/ to PyPI (handled by pypa/gh-action-pypi-publish in the
   workflow itself; this script does not call PyPI).
5. Update the root pyproject.toml: rewrite the marin-dupekit dependency pin
   to the new version. The dupekit find-links line was removed when we cut
   over from GitHub Releases to PyPI; we no longer rewrite find-links.
6. Re-resolve uv.lock so the auto-PR commits a consistent lockfile.

Usage:
    python rust/dupekit/build_package.py --bump --build-wheels --targets linux
    python rust/dupekit/build_package.py --bump --build-wheels --targets macos
    python rust/dupekit/build_package.py --bump --sdist-only
    python rust/dupekit/build_package.py --bump --update-pyproject

Prerequisites:
    maturin (installed automatically via uv tool if missing)
    zig is downloaded automatically if not found on PATH.
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

# Linux targets cross-compiled via zig
LINUX_TARGETS = [
    ("x86_64-unknown-linux-gnu", "2_28"),
    ("aarch64-unknown-linux-gnu", "2_28"),
]

# Mac targets built natively (zig can't cross-compile to macOS due to SDK requirements).
# Rust can cross-compile between macOS architectures, so both are built on any Mac host.
MAC_TARGETS = [
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
]


def _check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found. Install with: {install_hint}", file=sys.stderr)
        sys.exit(1)


def _zig_platform_key() -> str:
    machine = platform.machine()
    system = platform.system()
    arch_map = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}
    os_map = {"Darwin": "macos", "Linux": "linux"}
    if machine not in arch_map:
        raise ValueError(f"Unsupported architecture: {machine}")
    if system not in os_map:
        raise ValueError(f"Unsupported platform: {system}")
    arch = arch_map[machine]
    os_name = os_map[system]
    # Zig >= 0.13 release artifacts use arch-os ordering (e.g. x86_64-linux, aarch64-macos)
    return f"{arch}-{os_name}"


def _ensure_zig() -> str:
    """Return path to zig binary, downloading it if necessary."""
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


def _uv_tool_bin_dir() -> str:
    """Return the directory where uv installs tool binaries."""
    result = subprocess.run(["uv", "tool", "dir", "--bin"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _ensure_maturin() -> str:
    """Return path to maturin, installing via uv tool if missing."""
    existing = shutil.which("maturin")
    if existing:
        return existing

    print("Installing maturin via uv tool...")
    subprocess.run(["uv", "tool", "install", "maturin"], check=True)

    # uv tool installs to a bin dir that may not be on PATH yet
    tool_bin = _uv_tool_bin_dir()
    os.environ["PATH"] = f"{tool_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    path = shutil.which("maturin")
    if path is None:
        print("ERROR: maturin not found after installation", file=sys.stderr)
        sys.exit(1)
    return path


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
    """Bump the patch component of a semver string (e.g. 0.1.0 -> 0.1.1).

    Pre-release / build metadata are not handled; Cargo.toml is expected to
    hold a clean semver triple.
    """
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        print(f"ERROR: Cargo.toml version {version!r} is not a semver triple", file=sys.stderr)
        sys.exit(1)
    major, minor, patch = (int(p) for p in parts)
    return f"{major}.{minor}.{patch + 1}"


def bump_cargo_patch_version() -> str:
    """Bump rust/dupekit/Cargo.toml patch version and return the new version."""
    current = _read_cargo_version()
    new = _bump_patch(current)
    _write_cargo_version(new)
    print(f"Bumped marin-dupekit version: {current} -> {new}")
    return new


def _ensure_dist_dir() -> None:
    DIST_DIR.mkdir(exist_ok=True)


def build_wheels(targets: str) -> None:
    """Build platform wheels into dist/ (no sdist; see build_sdist).

    Args:
        targets: Which platforms to build for - "all" (default), "linux", or "macos".
            "all" builds Linux via zig and macOS natively (macOS host required for macOS wheels).
            "linux" builds only Linux wheels via zig cross-compilation (works from any host).
            "macos" builds only macOS wheels natively (requires macOS host).
    """
    maturin = _ensure_maturin()

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    system = platform.system()
    build_linux = targets in ("all", "linux")
    build_macos = targets in ("all", "macos")

    if build_linux:
        zig_bin = _ensure_zig()

        linux_triples = [t for t, _ in LINUX_TARGETS]
        print(f"Installing Rust targets: {', '.join(linux_triples)}")
        subprocess.run(["rustup", "target", "add", *linux_triples], check=True)

        # Put zig's directory on PATH so maturin can find it
        zig_dir = str(Path(zig_bin).parent)
        env = {**os.environ, "PATH": f"{zig_dir}{os.pathsep}{os.environ.get('PATH', '')}"}

        for target, manylinux in LINUX_TARGETS:
            print(f"\n--- Building wheel for {target} (zig) ---")
            cmd = [
                maturin,
                "build",
                "--release",
                "--out",
                str(DIST_DIR),
                "--manifest-path",
                str(MANIFEST_PATH),
                "--target",
                target,
                "--manylinux",
                manylinux,
                "--zig",
            ]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    if build_macos:
        if system != "Darwin":
            print("ERROR: macOS wheels require a macOS host (zig can't cross-compile to macOS)", file=sys.stderr)
            sys.exit(1)

        mac_triples = list(MAC_TARGETS)
        print(f"Installing Rust targets: {', '.join(mac_triples)}")
        subprocess.run(["rustup", "target", "add", *mac_triples], check=True)

        for target in MAC_TARGETS:
            print(f"\n--- Building wheel for {target} (native) ---")
            cmd = [
                maturin,
                "build",
                "--release",
                "--out",
                str(DIST_DIR),
                "--manifest-path",
                str(MANIFEST_PATH),
                "--target",
                target,
            ]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    wheels = list(DIST_DIR.glob("*.whl"))
    print(f"\nBuilt {len(wheels)} wheel(s):")
    for f in sorted(wheels):
        print(f"  {f.name}")


def build_sdist() -> None:
    """Build the sdist into dist/ exactly once.

    Lifted out of build_wheels() because each matrix leg uploaded a duplicate
    sdist tarball, and download-artifact's merge-multiple silently clobbers on
    duplicate filenames - whichever leg finished last wins, hiding any
    host-dependent metadata drift.
    """
    maturin = _ensure_maturin()
    _ensure_dist_dir()

    print("\n--- Building sdist ---")
    subprocess.run(
        [maturin, "sdist", "--out", str(DIST_DIR), "--manifest-path", str(MANIFEST_PATH)],
        check=True,
        cwd=REPO_ROOT,
    )

    sdists = list(DIST_DIR.glob("*.tar.gz"))
    print(f"\nBuilt {len(sdists)} sdist(s):")
    for f in sorted(sdists):
        print(f"  {f.name}")


# Marker substring identifying the dupekit find-links line. Matching on the
# repo+package prefix avoids touching the kitoken find-links line that lives
# in the same array and survives the cutover to PyPI.
_DUPEKIT_FIND_LINKS_MARKER = "releases/expanded_assets/dupekit-"


def _remove_dupekit_find_links_line(text: str) -> tuple[str, bool]:
    """Drop the dupekit find-links line from a multi-line find-links array.

    Returns (new_text, removed). `removed` is False if no matching line was
    found - this is fine on subsequent runs once the line has already been
    removed; the caller decides whether that's an error.
    """
    new_lines: list[str] = []
    removed = False
    for line in text.splitlines(keepends=True):
        if _DUPEKIT_FIND_LINKS_MARKER in line and "find-links" not in line:
            # Skip the dupekit URL line entirely. The kitoken line and the
            # surrounding `find-links = [` / `]` brackets are preserved with
            # correct comma/whitespace because we never alter their lines.
            removed = True
            continue
        new_lines.append(line)
    return "".join(new_lines), removed


def update_pyproject(version: str) -> None:
    """Pin marin-dupekit to `version` in root pyproject.toml and re-lock.

    Idempotent: re-running with the same version is a no-op except for the
    uv lock invocation. Re-running with the dupekit find-links line already
    removed is also fine.
    """
    original = PYPROJECT_PATH.read_text()
    text = original

    # Remove the dupekit find-links entry (if still present from prior cutover).
    text, _ = _remove_dupekit_find_links_line(text)

    # Rewrite the dependency pin. Match either the legacy "dupekit ..." or the
    # current "marin-dupekit ..." line so the script keeps working both on a
    # fresh post-cutover repo and on any leftover branches still mid-rename.
    new_text, n = re.subn(
        r'"(?:marin-)?dupekit\s*>=\s*[^"]*"',
        f'"marin-dupekit >= {version}"',
        text,
    )
    if n == 0:
        print(
            "ERROR: pyproject.toml has no marin-dupekit (or dupekit) dependency line to update.",
            file=sys.stderr,
        )
        sys.exit(1)
    text = new_text

    if text != original:
        PYPROJECT_PATH.write_text(text)
        print(f"\n--- Updated pyproject.toml: marin-dupekit pinned to >= {version} ---")
    else:
        print(f"\n--- pyproject.toml already pinned to marin-dupekit >= {version} ---")

    print("\n--- Running uv lock ---")
    subprocess.run(["uv", "lock"], check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bump",
        action="store_true",
        help="Bump the patch version in rust/dupekit/Cargo.toml before any other action.",
    )
    parser.add_argument(
        "--build-wheels",
        action="store_true",
        help="Build platform wheels into dist/ (use --targets to pick a subset).",
    )
    parser.add_argument(
        "--targets",
        choices=["all", "linux", "macos"],
        default="all",
        help="Which platforms to build wheels for: all (default), linux, or macos.",
    )
    parser.add_argument(
        "--sdist-only",
        action="store_true",
        help="Build only the sdist into dist/ (run once in the release job, not per matrix leg).",
    )
    parser.add_argument(
        "--update-pyproject",
        action="store_true",
        help="Update root pyproject.toml dependency pin and run uv lock.",
    )
    args = parser.parse_args()

    if not (args.bump or args.build_wheels or args.sdist_only or args.update_pyproject):
        parser.error("nothing to do; pass at least one of --bump, --build-wheels, --sdist-only, --update-pyproject")

    if args.bump:
        version = bump_cargo_patch_version()
    else:
        version = _read_cargo_version()

    print(f"marin-dupekit version: {version}")

    if args.build_wheels:
        build_wheels(args.targets)

    if args.sdist_only:
        build_sdist()

    if args.update_pyproject:
        update_pyproject(version)


if __name__ == "__main__":
    main()
