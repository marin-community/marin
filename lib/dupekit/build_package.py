#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build (and let CI publish) the marin-dupekit dists.

Driven by .github/workflows/dupekit-release-wheels.yaml. Mirrors
lib/finelog/build_package.py: same nightly/stable/manual mode split and the
same zig-cross-compiled manylinux + native macOS wheel matrix.

Builds both dists of the pair in lockstep at one resolved version (see
lib/dupekit/pyproject.toml for how the pure and native packages relate):
  - marin-dupekit-native: platform wheels from the maturin project in rust/,
    one per target across the CI matrix.
  - marin-dupekit: one pure py3-none-any wheel, built on the linux leg only so
    artifacts never collide across legs.

Modes:
    nightly  -- `<bumped_patch>-dev.<YYYYMMDDhhmm>` (UTC), where
                `<bumped_patch>` is one patch above max(pyproject version,
                latest marin-dupekit stable on PyPI; the native dist follows the
                same value). Sorting above the current stable is what lets
                `marin-dupekit-native >= 0.1.2.dev0`-style floors resolve to the
                latest dev. pyproject never needs to be re-bumped after a stable
                cut.
    stable   -- version supplied via --version (extracted from the tag in CI).
                Both the pure pyproject and rust/Cargo.toml are rewritten on
                disk so the builds carry that version; the change is not
                committed. A stable tag therefore cuts a stable PAIR.
    manual   -- `<pyproject>+<sha>` (PEP 440 local version). Build-only smoke
                for PRs and ad-hoc dev; PyPI rejects local-version identifiers,
                so the publish job declines to run in this mode.

Usage:
    python lib/dupekit/build_package.py --mode nightly --build linux
    python lib/dupekit/build_package.py --mode stable --version 0.1.7 --build sdist
    python lib/dupekit/build_package.py --mode manual --build macos
"""

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DUPEKIT_DIR = Path(__file__).resolve().parent
NATIVE_DIR = DUPEKIT_DIR / "rust"
REPO_ROOT = DUPEKIT_DIR.parent.parent
# The pure dist's pyproject is the canonical version source; the resolved
# version is stamped into it and into rust/Cargo.toml (which maturin reads for
# the native dist's dynamic version) so the wheels agree.
PYPROJECT_PATH = DUPEKIT_DIR / "pyproject.toml"
CARGO_PATH = NATIVE_DIR / "Cargo.toml"
DIST_DIR = REPO_ROOT / "dist"
TOOLS_DIR = REPO_ROOT / ".tools"

PYPI_JSON_URL = "https://pypi.org/pypi/marin-dupekit/json"

ZIG_VERSION = "0.15.2"
# Zig tarballs are large and ziglang.org's own server is slow and rate-limited
# (<0.1 MB/s), so prefer the community mirrors from
# https://ziglang.org/download/community-mirrors.txt and fall back to the
# official server only if every mirror fails. Mirrors intermittently 500 or
# drop the connection (a single hard-coded mirror is a CI flake waiting to
# happen), so we rotate through several with retries. Each mirror serves the
# tarball at `<base>/<filename>`; the official server nests it under
# `/download/<version>/`.
ZIG_MIRRORS = (
    "https://pkg.earth/zig",
    "https://pkg.hexops.org/zig",
    "https://zig.linus.dev/zig",
)
ZIG_OFFICIAL_BASE = "https://ziglang.org/download"
ZIG_DOWNLOAD_ATTEMPTS_PER_SOURCE = 2

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


def _download_zig_archive(filename: str, dest: Path, reporthook) -> None:
    """Fetch the zig tarball into ``dest``, trying mirrors then ziglang.org.

    Tries each community mirror (a couple of attempts apiece, since they
    intermittently 500 or drop the connection) before falling back to the slow,
    rate-limited official server. Raises if every source fails.
    """
    sources = [f"{base}/{filename}" for base in ZIG_MIRRORS]
    sources.append(f"{ZIG_OFFICIAL_BASE}/{ZIG_VERSION}/{filename}")
    last_error: Exception | None = None
    for url in sources:
        for attempt in range(1, ZIG_DOWNLOAD_ATTEMPTS_PER_SOURCE + 1):
            print(f"Downloading zig {ZIG_VERSION} from {url} (attempt {attempt})...")
            try:
                urllib.request.urlretrieve(url, dest, reporthook=reporthook)
                return
            except (urllib.error.URLError, OSError) as e:
                last_error = e
                print(f"  download failed: {e}")
                dest.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download zig {ZIG_VERSION} from any mirror or ziglang.org") from last_error


def _ensure_zig() -> str:
    """Return path to zig binary, downloading it if absent (see _download_zig_archive)."""
    existing = shutil.which("zig")
    if existing:
        return existing

    plat = _zig_platform_key()
    zig_dir = TOOLS_DIR / f"zig-{plat}-{ZIG_VERSION}"
    zig_bin = zig_dir / "zig"
    if zig_bin.exists():
        return str(zig_bin)

    filename = f"zig-{plat}-{ZIG_VERSION}.tar.xz"
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

    _download_zig_archive(filename, archive_path, _report)
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
    """Run maturin from lib/dupekit/rust so it reads the native pyproject.toml.

    The `[tool.maturin] manifest-path` in lib/dupekit/rust/pyproject.toml selects
    the cdylib crate; we deliberately do NOT pass --manifest-path (that would
    make maturin look for a sibling pyproject next to the crate).
    """
    cmd = [_ensure_maturin(), *args]
    subprocess.run(cmd, check=True, cwd=NATIVE_DIR, env=env)


# Match the `[project]`/`[package]` table's `version = "..."` — the first
# line-anchored version key in each file (build-system/table headers above it
# have none, and dependency strings are indented so the anchor skips them).
_VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]+"', re.MULTILINE)


def _read_project_version() -> str:
    text = PYPROJECT_PATH.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        print("ERROR: Could not parse [project] version from pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def _write_project_version(new_version: str) -> None:
    # The pure pyproject carries [project] version; rust/Cargo.toml carries the
    # native dist's [package] version (maturin reads it via dynamic version).
    for path in (PYPROJECT_PATH, CARGO_PATH):
        text = path.read_text()
        new_text, n = _VERSION_RE.subn(rf'\1"{new_version}"', text, count=1)
        if n != 1:
            print(f"ERROR: Failed to rewrite version in {path}", file=sys.stderr)
            sys.exit(1)
        path.write_text(new_text)


def _parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.split(".")[:3]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Not a semver triple: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _max_version(a: str, b: str) -> str:
    return a if _parse_semver(a) >= _parse_semver(b) else b


def _bump_patch(version: str) -> str:
    major, minor, patch = _parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _query_pypi_latest_stable() -> str | None:
    """Latest non-pre-release version on PyPI, or None if the project doesn't exist yet.

    PyPI's `info.version` reports the latest stable (it skips pre-releases per
    its own conventions), which is exactly what we want as the bump base.
    """
    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return data.get("info", {}).get("version") or None


def _resolve_nightly_version() -> str:
    """Build a nightly version that uv/pip will prefer over the current stable.

    Format: `<bumped_patch_above_stable>-dev.<YYYYMMDDhhmm>`.
      - The bump base is `max(pyproject version, latest stable on PyPI)`, so the
        resulting `<bumped>` always sits one patch *above* whatever is currently
        stable. PEP 440 then orders the dev release *above* the stable
        (`0.1.3.dev* > 0.1.2`), which is what lets the
        `marin-dupekit-native >= 0.1.2.dev0` floor resolve to the latest dev.
      - Querying PyPI also means pyproject.toml never has to be re-bumped after
        a stable cut — the script always anticipates the next patch correctly.
      - `<YYYYMMDDhhmm>` (UTC) keeps each dev release unique per minute and
        readable at a glance.
    """
    declared = _read_project_version()
    pypi_stable = _query_pypi_latest_stable()
    base = _max_version(declared, pypi_stable) if pypi_stable else declared
    bumped = _bump_patch(base)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M")
    return f"{bumped}-dev.{stamp}"


def _resolve_manual_version() -> str:
    sha = os.environ.get("GITHUB_SHA", "")[:8] or "local"
    # `+<segment>` is semver build metadata and PEP 440 local-version - both
    # ecosystems treat it as "same release, different build", which is what we
    # want for ad-hoc smoke builds.
    return f"{_read_project_version()}+{sha}"


def resolve_version(mode: str, override: str | None) -> str:
    if mode == "stable":
        if not override:
            print("ERROR: --mode stable requires --version", file=sys.stderr)
            sys.exit(1)
        return override
    if mode == "nightly":
        return _resolve_nightly_version()
    if mode == "manual":
        return _resolve_manual_version()
    raise ValueError(f"unknown mode: {mode}")


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


def _uv_build(*args: str) -> None:
    """Build the pure dist with uv, dropping the .gitignore uv writes into the
    output dir — pypi-publish rejects non-distribution files in packages-dir,
    and both the artifact upload and the publish step glob all of dist/."""
    subprocess.run(["uv", "build", *args, "--out-dir", str(DIST_DIR)], check=True, cwd=DUPEKIT_DIR)
    (DIST_DIR / ".gitignore").unlink(missing_ok=True)


def _build_pure_wheel() -> None:
    print("\n--- Building marin-dupekit (pure) wheel ---")
    _uv_build("--wheel")


def build_linux_wheels() -> None:
    _build_wheels(LINUX_TARGETS, use_zig=True)
    # The pure wheel is platform-independent; build it on this leg only so the
    # merged artifacts never contain two copies of the same filename.
    _build_pure_wheel()
    _list_dist_artifacts("wheel(s)")


def build_macos_wheels() -> None:
    if platform.system() != "Darwin":
        print("ERROR: macOS wheels require a macOS host (zig can't cross-compile to macOS)", file=sys.stderr)
        sys.exit(1)
    _build_wheels(MAC_TARGETS, use_zig=False)
    _list_dist_artifacts("wheel(s)")


def build_sdists() -> None:
    # Adds to dist/ rather than resetting it: the release job downloads wheels
    # via download-artifact before invoking us, and we want them in the same
    # directory so `pypa/gh-action-pypi-publish` uploads everything together.
    DIST_DIR.mkdir(exist_ok=True)
    print("\n--- Building marin-dupekit-native sdist ---")
    _maturin("sdist", "--out", str(DIST_DIR))
    print("\n--- Building marin-dupekit sdist ---")
    _uv_build("--sdist")
    _list_dist_artifacts("sdist(s)")


_BUILDERS = {
    "linux": build_linux_wheels,
    "macos": build_macos_wheels,
    "sdist": build_sdists,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["nightly", "stable", "manual"], required=True)
    parser.add_argument(
        "--version",
        help=(
            "Stable: required, taken verbatim. "
            "Nightly: optional precomputed value (CI computes it once in the resolve "
            "job and passes it to every matrix leg, so all wheels and the sdist agree). "
            "Manual: optional override; otherwise derived from pyproject.toml + GITHUB_SHA."
        ),
    )
    parser.add_argument(
        "--build",
        choices=sorted(_BUILDERS),
        help="Build target. Omit with --resolve-only to just print the version.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print the resolved version to stdout and emit it to $GITHUB_OUTPUT; do not build.",
    )
    args = parser.parse_args()

    if not args.resolve_only and not args.build:
        parser.error("--build is required unless --resolve-only is set")

    version = args.version if args.version else resolve_version(args.mode, args.version)
    print(f"marin-dupekit / marin-dupekit-native version: {version} (mode={args.mode})")
    _emit_github_output("version", version)

    if args.resolve_only:
        return

    # The pure wheel reads [project] version from pyproject.toml; the native
    # wheel reads [package] version from rust/Cargo.toml. Stamp the resolved
    # version into both for the duration of this build; we never commit it back.
    _write_project_version(version)
    _BUILDERS[args.build]()


if __name__ == "__main__":
    main()
