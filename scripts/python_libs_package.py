#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and publish marin-* lib wheels.

Builds the seven pure-Python marin-* lib packages (marin, marin-iris,
marin-fray, marin-haliax, marin-levanter, marin-rigging, marin-zephyr) into
dist/, then optionally publishes them as GitHub Releases.

Four modes:
    nightly  -- version becomes <base>.dev<YYYYMMDD>; overwrites the rolling
                marin-<pkg>-latest tag in place. No dated history is kept;
                reproducibility comes from stable tags, not historical nightlies.
    stable   -- version is taken from --version; creates marin-<pkg>-v<version>
                and overwrites marin-<pkg>-stable.
    manual   -- version becomes <base>+manual.<sha>; build only (no publish).
                Useful for inspecting wheels from a workflow_dispatch run.
    vendor   -- version becomes <base>.dev<YYYYMMDDHHMMSS>; copy wheels to a
                local directory (no GH publish). For local-iteration loops
                where a marin worktree feeds wheels into an experiment repo's
                find-links. The timestamp guarantees rebuilt wheels beat any
                nightly already published earlier the same day.

Usage:
    python scripts/python_libs_package.py --mode nightly
    python scripts/python_libs_package.py --mode stable --version 1.0.0
    python scripts/python_libs_package.py --mode nightly --skip-publish
    python scripts/python_libs_package.py --skip-build --publish-only
    python scripts/python_libs_package.py --mode vendor --vendor ../tiny-tpu/vendor

The build is done from a temporary in-place patch of each package's version
file plus a cross-pin rewrite of every sibling dependency. Mutations are
reverted on exit (success OR failure) so the working tree stays clean.
After building, dist/BUILD_INFO.json records the resolved version so that a
subsequent --publish-only call (typically the publish job in CI) uses the
exact version the build job produced, even if the run straddles midnight UTC.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
REPO = "marin-community/marin"


# Each entry: (dist name, lib subdir, version-file path relative to lib subdir, version-file kind)
# kind = "pyproject" -> patch  version = "..."  in pyproject.toml
# kind = "about_py"  -> patch  __version__ = "..."  in src/<pkg>/__about__.py
PACKAGES: dict[str, dict[str, str]] = {
    "marin": {"path": "lib/marin", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-iris": {"path": "lib/iris", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-fray": {"path": "lib/fray", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-rigging": {"path": "lib/rigging", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-zephyr": {"path": "lib/zephyr", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-levanter": {"path": "lib/levanter", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-haliax": {"path": "lib/haliax", "version_file": "src/haliax/__about__.py", "kind": "about_py"},
}

SIBLING_NAMES = sorted(PACKAGES.keys(), key=len, reverse=True)


# ---------- helpers ----------------------------------------------------------


def _check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found. Install with: {install_hint}", file=sys.stderr)
        sys.exit(1)


def _git_short_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT).strip()


def _read_base_version(pkg: str) -> str:
    info = PACKAGES[pkg]
    path = REPO_ROOT / info["path"] / info["version_file"]
    text = path.read_text()
    if info["kind"] == "pyproject":
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    else:
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not read version from {path}")
    return m.group(1)


def _set_version(text: str, kind: str, new_version: str) -> str:
    if kind == "pyproject":
        new_text, count = re.subn(
            r'^version\s*=\s*"[^"]+"',
            f'version = "{new_version}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        new_text, count = re.subn(
            r'^__version__\s*=\s*"[^"]+"',
            f'__version__ = "{new_version}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if count != 1:
        raise RuntimeError(f"Failed to patch version (kind={kind})")
    return new_text


# Match dependency list items: lines that are indented and start with a quoted
# sibling name. Anchored on `^\s+"` so we never touch metadata lines like
# `name = "marin"` (no leading whitespace) or single-line `gpu = ["..."]`
# entries (no marin siblings appear in those today; verified by grep).
_SIBLING_ALT = "|".join(re.escape(s) for s in sorted(PACKAGES, key=len, reverse=True))
_SIBLING_ITEM_RE = re.compile(
    rf'^(?P<indent>\s+)"(?P<name>{_SIBLING_ALT})(?![-\w])(?P<extras>\[[^\]]*\])?[^"]*"(?P<tail>.*)$',
    re.MULTILINE,
)


def _rewrite_sibling_pins(text: str, version: str) -> str:
    """Pin every sibling marin-* package in dependency list items to ==<version>."""
    return _SIBLING_ITEM_RE.sub(
        lambda m: (f'{m.group("indent")}"{m.group("name")}{m.group("extras") or ""}=={version}"{m.group("tail")}'),
        text,
    )


@contextmanager
def patched_tree(version: str):
    """Patch every package's version file and sibling pins; revert on exit.

    Captures the original text of each path exactly once, before any mutation,
    so the finally block restores the truly-original content even if multiple
    patches touched the same file.
    """
    originals: dict[Path, str] = {}
    try:
        for info in PACKAGES.values():
            pyproject_path = REPO_ROOT / info["path"] / "pyproject.toml"
            version_path = REPO_ROOT / info["path"] / info["version_file"]

            if pyproject_path not in originals:
                originals[pyproject_path] = pyproject_path.read_text()
            if version_path not in originals:
                originals[version_path] = version_path.read_text()

            # Apply version patch first; for haliax this writes __about__.py
            # (separate file from pyproject), for the rest it overwrites the
            # pyproject we just snapshotted above.
            patched_version = _set_version(originals[version_path], info["kind"], version)
            version_path.write_text(patched_version)

            # Then sibling-pin rewrite always targets pyproject.toml. Re-read
            # in case version patch already wrote pyproject.
            current_pyproject = pyproject_path.read_text()
            new_pyproject = _rewrite_sibling_pins(current_pyproject, version)
            if new_pyproject != current_pyproject:
                pyproject_path.write_text(new_pyproject)
        yield
    finally:
        for path, text in originals.items():
            path.write_text(text)


# ---------- build ------------------------------------------------------------


def _highest_base_version() -> str:
    """Return the highest version currently declared across the seven libs.

    All seven packages share one synthetic version per build so cross-pins
    resolve cleanly. Picking the max keeps the synthetic version above the
    most recent stable so uv prefers it.
    """
    bases = [_read_base_version(p) for p in PACKAGES]
    return max(bases, key=lambda v: tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-+]", v)))


def resolve_version(mode: str, explicit: str | None) -> str:
    """Return the build version for the requested mode.

    nightly -> <base>.dev<YYYYMMDD>
    stable  -> <explicit>
    manual  -> <base>+manual.<sha>
    vendor  -> <base>.dev<YYYYMMDDHHMMSS>
    """
    if mode == "stable":
        if not explicit:
            raise SystemExit("--version is required for --mode stable")
        return explicit
    if mode == "nightly":
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{_highest_base_version()}.dev{date}"
    if mode == "manual":
        sha = _git_short_sha()
        return f"{_highest_base_version()}+manual.{sha}"
    if mode == "vendor":
        # Second-precision timestamp guarantees the freshly-built wheel beats
        # any nightly built earlier today, so `uv sync` in the consumer always
        # picks up the local copy without cache games.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{_highest_base_version()}.dev{ts}"
    raise SystemExit(f"Unknown mode: {mode}")


# Path inside dist/ where build_wheels persists the resolved version + mode.
# Publish reads this instead of re-resolving so a workflow that straddles
# midnight UTC can't compute a different date in build vs publish.
BUILD_INFO_PATH = DIST_DIR / "BUILD_INFO.json"


def write_build_info(version: str, mode: str) -> None:
    BUILD_INFO_PATH.write_text(json.dumps({"version": version, "mode": mode}, indent=2))


def read_build_info() -> dict[str, str] | None:
    if not BUILD_INFO_PATH.is_file():
        return None
    return json.loads(BUILD_INFO_PATH.read_text())


def build_wheels(version: str, mode: str) -> None:
    """Build all seven marin-* wheels into DIST_DIR with version patched in.

    Persists BUILD_INFO.json next to the wheels so the publish step (which
    runs in a separate job and downloads dist/ as an artifact) can read the
    exact version this build used instead of re-resolving it. That guarantees
    a workflow run straddling midnight UTC produces a consistent version
    across build and publish.
    """
    _check_tool("uv", "https://docs.astral.sh/uv/")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    with patched_tree(version):
        for name, info in PACKAGES.items():
            pkg_dir = REPO_ROOT / info["path"]
            print(f"\n--- Building {name} ({version}) ---")
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(DIST_DIR), str(pkg_dir)],
                check=True,
                cwd=REPO_ROOT,
            )

    wheels = sorted(DIST_DIR.glob("*.whl"))
    print(f"\nBuilt {len(wheels)} wheel(s):")
    for w in wheels:
        print(f"  {w.name}")
    if len(wheels) != len(PACKAGES):
        raise RuntimeError(f"Expected {len(PACKAGES)} wheels, got {len(wheels)}")

    write_build_info(version, mode)


# ---------- vendor -----------------------------------------------------------


def vendor_copy(target: Path) -> None:
    """Drop freshly-built wheels into target/, replacing any prior marin-* wheels.

    Cleans only files matching marin*-*.whl so unrelated files in the target
    directory (e.g. .gitkeep, README) are left alone. Used by --mode vendor
    to feed local wheels into a downstream experiment's find-links.
    """
    target.mkdir(parents=True, exist_ok=True)
    stale = sorted(target.glob("marin*-*.whl"))
    for s in stale:
        s.unlink()
    if stale:
        print(f"\nRemoved {len(stale)} stale marin-* wheel(s) from {target}")
    print(f"\nCopying wheels to {target}:")
    for wheel in sorted(DIST_DIR.glob("*.whl")):
        dest = target / wheel.name
        shutil.copy2(wheel, dest)
        print(f"  -> {dest.name}")


# ---------- publish ----------------------------------------------------------


def _wheel_for(pkg: str) -> Path:
    """Return the dist/ wheel matching pkg (uv normalises hyphens to underscores)."""
    stem = pkg.replace("-", "_")
    candidates = list(DIST_DIR.glob(f"{stem}-*.whl"))
    if not candidates:
        raise FileNotFoundError(f"No wheel for {pkg} in {DIST_DIR}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple wheels for {pkg}: {[c.name for c in candidates]}")
    return candidates[0]


def _gh_release_replace(tag: str, files: list[Path], title: str, notes: str, prerelease: bool) -> None:
    """Idempotently (re)create a GitHub release with the given assets."""
    subprocess.run(
        ["gh", "release", "delete", tag, "--yes", "--cleanup-tag", "--repo", REPO],
        check=False,
        capture_output=True,
    )
    cmd = [
        "gh",
        "release",
        "create",
        tag,
        *[str(f) for f in files],
        "--repo",
        REPO,
        "--title",
        title,
        "--notes",
        notes,
    ]
    if prerelease:
        cmd.append("--prerelease")
    subprocess.run(cmd, check=True)


def publish_releases(version: str, mode: str) -> None:
    """Per-package GH release.

    nightly -> overwrite the rolling marin-<pkg>-latest tag in place. No dated
               tags are kept; consumers point find-links at the rolling URL
               and always get the most recent build. Reproducibility comes
               from stable tags, not from historical nightlies.
    stable  -> create marin-<pkg>-v<version> and overwrite marin-<pkg>-stable.
    """
    _check_tool("gh", "https://cli.github.com/")

    if mode == "nightly":
        for pkg in PACKAGES:
            wheel = _wheel_for(pkg)
            tag = f"{pkg}-latest"
            print(f"\n--- Publishing {tag} ({version}) ---")
            _gh_release_replace(
                tag=tag,
                files=[wheel],
                title=f"{pkg} (latest)",
                notes=f"Rolling nightly. Currently pointing at {version}.",
                prerelease=True,
            )
        return

    if mode == "stable":
        for pkg in PACKAGES:
            wheel = _wheel_for(pkg)
            for tag, label in ((f"{pkg}-v{version}", "stable"), (f"{pkg}-stable", "rolling stable")):
                print(f"\n--- Publishing {tag} ---")
                _gh_release_replace(
                    tag=tag,
                    files=[wheel],
                    title=f"{pkg} {version}",
                    notes=f"{pkg} {version} ({label})",
                    prerelease=False,
                )
        return

    raise SystemExit(f"publish_releases called with unsupported mode: {mode}")


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["nightly", "stable", "manual", "vendor"], default="nightly")
    parser.add_argument("--version", default=None, help="Required for --mode stable")
    parser.add_argument(
        "--vendor",
        type=Path,
        default=None,
        help="Target directory to drop wheels into (required for --mode vendor)",
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing dist/")
    parser.add_argument("--skip-publish", action="store_true", help="Build only")
    parser.add_argument("--publish-only", action="store_true", help="Same as --skip-build")
    args = parser.parse_args()

    if args.publish_only:
        args.skip_build = True

    if args.mode == "vendor":
        if args.vendor is None:
            raise SystemExit("--vendor PATH is required for --mode vendor")
        version = resolve_version(args.mode, args.version)
        print(f"Mode:        {args.mode}\nVersion:     {version}")
        build_wheels(version, args.mode)
        vendor_copy(args.vendor.expanduser().resolve())
        print("\nDone.")
        return

    if not args.skip_build:
        version = resolve_version(args.mode, args.version)
        print(f"Mode:        {args.mode}\nVersion:     {version}")
        build_wheels(version, args.mode)
    else:
        if not DIST_DIR.exists() or not list(DIST_DIR.glob("*.whl")):
            raise SystemExit(f"No wheels in {DIST_DIR}; remove --skip-build/--publish-only or build first.")
        info = read_build_info()
        if info is not None:
            version = info["version"]
            print(f"Mode:        {args.mode}\nVersion:     {version} (from BUILD_INFO.json)")
        else:
            # Legacy path: dist/ has wheels but no BUILD_INFO.json. Fall back
            # to re-resolving; this is the only branch where the midnight-drift
            # bug could resurface.
            version = resolve_version(args.mode, args.version)
            print(f"Mode:        {args.mode}\nVersion:     {version} (re-resolved; no BUILD_INFO.json)")

    if args.skip_publish or args.mode == "manual":
        print(f"\nBuild complete. Wheels in {DIST_DIR}/")
        return

    publish_releases(version, args.mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
