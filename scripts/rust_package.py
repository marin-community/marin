#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, publish, and pin dupekit wheels for all platforms.

Builds all wheels from a single machine:
  - Linux x86_64 + aarch64 via maturin + zig cross-compilation (any host OS)
  - macOS x86_64 + arm64 natively (requires macOS host)

Publishes wheels to a GitHub Release and updates pyproject.toml's find-links.

Usage:
    python scripts/rust_package.py              # full flow: build + publish + update pyproject
    python scripts/rust_package.py --skip-build  # reuse existing dist/, publish + update
    python scripts/rust_package.py --skip-publish # build only, don't upload or update pyproject
    python scripts/rust_package.py --targets linux  # build Linux wheels only
    python scripts/rust_package.py --targets macos  # build macOS wheels only (requires macOS host)

Prerequisites:
    maturin (installed automatically via uv tool if missing)
    gh auth login

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
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "rust" / "dupekit" / "Cargo.toml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DIST_DIR = REPO_ROOT / "dist"
TOOLS_DIR = REPO_ROOT / ".tools"

ZIG_VERSION = "0.15.2"

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

REPO = "marin-community/marin"


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

    url = f"https://ziglang.org/download/{ZIG_VERSION}/zig-{plat}-{ZIG_VERSION}.tar.xz"
    print(f"Downloading zig {ZIG_VERSION} for {plat}...")

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / f"zig-{plat}-{ZIG_VERSION}.tar.xz"

    urllib.request.urlretrieve(url, archive_path)

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


def _git_short_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT).strip()


def _host_rust_target() -> str:
    """Return the Rust target triple for the current host, or empty string if unsupported."""
    machine = platform.machine()
    system = platform.system()

    arch = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine)
    if arch is None:
        return ""

    if system == "Darwin":
        return f"{arch}-apple-darwin"
    elif system == "Linux":
        return f"{arch}-unknown-linux-gnu"
    return ""


def build_wheels(targets: str) -> None:
    """Build platform wheels + sdist into dist/.

    Args:
        targets: Which platforms to build for — "all" (default), "linux", or "macos".
            "all" builds Linux via zig and macOS natively (macOS host required for macOS wheels).
            "linux" builds only Linux wheels via zig cross-compilation (works from any host).
            "macos" builds only macOS wheels natively (requires macOS host).
    """
    maturin = _ensure_maturin()
    _check_tool("gh", "https://cli.github.com/")

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

    print("\n--- Building sdist ---")
    subprocess.run(
        [maturin, "sdist", "--out", str(DIST_DIR), "--manifest-path", str(MANIFEST_PATH)],
        check=True,
        cwd=REPO_ROOT,
    )

    wheels = list(DIST_DIR.glob("*.whl"))
    sdists = list(DIST_DIR.glob("*.tar.gz"))
    print(f"\nBuilt {len(wheels)} wheel(s) and {len(sdists)} sdist(s):")
    for f in sorted(wheels + sdists):
        print(f"  {f.name}")


def publish_release(tag: str, title: str) -> None:
    """Create a GitHub Release and upload all artifacts from dist/."""
    _check_tool("gh", "https://cli.github.com/")

    files = sorted(DIST_DIR.glob("*"))
    if not files:
        print("ERROR: No files in dist/ to publish", file=sys.stderr)
        sys.exit(1)

    # Delete existing release if present (idempotent)
    subprocess.run(
        ["gh", "release", "delete", tag, "--yes", "--repo", REPO],
        check=False,
        capture_output=True,
    )

    # Create pinned release
    print(f"\n--- Creating release {tag} ---")
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            *[str(f) for f in files],
            "--repo",
            REPO,
            "--prerelease",
            "--title",
            title,
        ],
        check=True,
    )

    # Update rolling "dupekit-latest" release
    subprocess.run(
        ["gh", "release", "delete", "dupekit-latest", "--yes", "--repo", REPO],
        check=False,
        capture_output=True,
    )
    print("\n--- Updating dupekit-latest rolling release ---")
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            "dupekit-latest",
            *[str(f) for f in files],
            "--repo",
            REPO,
            "--prerelease",
            "--title",
            "dupekit (latest wheels)",
            "--notes",
            f"Rolling release. Currently pointing at {tag}.",
        ],
        check=True,
    )


def update_pyproject(tag: str, version: str) -> None:
    """Update pyproject.toml find-links to point at the new release, then uv lock."""
    original = PYPROJECT_PATH.read_text()
    text = original

    new_url = f"https://github.com/{REPO}/releases/expanded_assets/{tag}"

    # Replace find-links URL
    text = re.sub(
        r'find-links\s*=\s*\["https://github\.com/[^"]*"\]',
        f'find-links = ["{new_url}"]',
        text,
    )

    # Update dupekit version pin
    text = re.sub(
        r'"dupekit\s*>=\s*[^"]*"',
        f'"dupekit >= {version}"',
        text,
    )

    if text == original:
        print("ERROR: pyproject.toml regex substitutions did not match anything.", file=sys.stderr)
        print("The find-links or dupekit dependency format may have changed.", file=sys.stderr)
        sys.exit(1)

    PYPROJECT_PATH.write_text(text)
    print(f"\n--- Updated pyproject.toml find-links to {tag} ---")

    print("\n--- Running uv lock ---")
    subprocess.run(["uv", "lock"], check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing dist/")
    parser.add_argument("--skip-publish", action="store_true", help="Build only, don't upload or update pyproject")
    parser.add_argument(
        "--targets",
        choices=["all", "linux", "macos"],
        default="all",
        help="Which platforms to build: all (default), linux, or macos",
    )
    args = parser.parse_args()

    version = _read_cargo_version()
    short_sha = _git_short_sha()
    tag = f"dupekit-{version}-{short_sha}"
    title = f"dupekit {version} ({short_sha})"

    print(f"dupekit version: {version}")
    print(f"git sha: {short_sha}")
    print(f"release tag: {tag}")

    if not args.skip_build:
        build_wheels(args.targets)

    if not args.skip_publish:
        publish_release(tag, title)
        update_pyproject(tag, version)
        print("\nDone. Run 'uv sync' to install the new wheels.")
    else:
        print(f"\nBuild complete. Wheels in {DIST_DIR}/")


if __name__ == "__main__":
    main()
