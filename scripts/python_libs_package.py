#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the marin-* lib wheels for PyPI publication.

Builds the seven pure-Python marin-* lib packages (marin-core, marin-iris,
marin-fray, marin-haliax, marin-levanter, marin-rigging, marin-zephyr) into
dist/. Publication to PyPI is done by the release workflow
(.github/workflows/marin-release-libs-wheels.yaml) via
`pypa/gh-action-pypi-publish` with OIDC trusted publishing -- this script
never uploads anything and never needs a token.

marin-finelog and marin-dupekit are NOT built here: each has its own release
workflow (finelog-release-wheels.yaml, dupekit-release-wheels.yaml) that
publishes a pure-Python wheel together with its native (maturin) companion
(marin-finelog-server, marin-dupekit-native).

Four modes:
    nightly  -- version becomes <dev_base>.dev<YYYYMMDDhhmm> (UTC). <dev_base>
                is one patch above max(declared version, latest stable on
                PyPI), so PEP 440 orders the dev release above the current
                stable and `pip install --pre` / `uv` prefer it.
    stable   -- version is taken verbatim from --version (the release
                workflow extracts it from the marin-libs-v<version> tag).
    manual   -- version becomes <declared>+manual.<sha>; a build-only smoke.
                PyPI rejects local-version identifiers, so the workflow's
                publish job declines to run in this mode.
    vendor   -- version becomes <dev_base>.dev<YYYYMMDDHHMMSS>; copy wheels to
                a local directory (no publish). For local-iteration loops
                where a marin worktree feeds wheels into an experiment repo's
                find-links. The second-precision timestamp guarantees rebuilt
                wheels beat any nightly already published earlier the same day.

Usage:
    python scripts/python_libs_package.py --mode nightly
    python scripts/python_libs_package.py --mode stable --version 0.2.0
    python scripts/python_libs_package.py --mode nightly --resolve-only
    python scripts/python_libs_package.py --mode vendor --vendor ../tiny-tpu/vendor

The build is done from a temporary in-place patch of each package's version
file plus a cross-pin rewrite of every sibling dependency, so the eight wheels
published together always require each other at the exact same version.
Mutations are reverted on exit (success OR failure) so the working tree stays
clean.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


# Each entry: (dist name, lib subdir, version-file path relative to lib subdir, version-file kind)
# kind = "pyproject" -> patch  version = "..."  in pyproject.toml
# kind = "about_py"  -> patch  __version__ = "..."  in src/<pkg>/__about__.py
PACKAGES: dict[str, dict[str, str]] = {
    "marin-core": {"path": "lib/marin", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-iris": {"path": "lib/iris", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-fray": {"path": "lib/fray", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-rigging": {"path": "lib/rigging", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-zephyr": {"path": "lib/zephyr", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-levanter": {"path": "lib/levanter", "version_file": "pyproject.toml", "kind": "pyproject"},
    "marin-haliax": {"path": "lib/haliax", "version_file": "src/haliax/__about__.py", "kind": "about_py"},
}


# ---------- helpers ----------------------------------------------------------


def _check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found. Install with: {install_hint}", file=sys.stderr)
        sys.exit(1)


def _git_short_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT).strip()


def _emit_github_output(key: str, value: str) -> None:
    """Append `key=value` to $GITHUB_OUTPUT when running under GitHub Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


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
# `name = "marin-core"` (no leading whitespace) or single-line `gpu = ["..."]`
# entries (no marin siblings appear in those today; verified by grep).
_SIBLING_ALT = "|".join(re.escape(s) for s in sorted(PACKAGES, key=len, reverse=True))
_SIBLING_ITEM_RE = re.compile(
    rf'^(?P<indent>\s+)"(?P<name>{_SIBLING_ALT})(?![-\w])(?P<extras>\[[^\]]*\])?[^"]*"(?P<tail>.*)$',
    re.MULTILINE,
)


def _rewrite_sibling_pins(text: str, version: str) -> str:
    """Pin every sibling marin-* package in dependency list items to ==<version>."""
    return _SIBLING_ITEM_RE.sub(
        lambda m: f'{m.group("indent")}"{m.group("name")}{m.group("extras") or ""}=={version}"{m.group("tail")}',
        text,
    )


# Match dependency list items that use PEP 440 direct URL form
# (`"pkg @ git+https://..."` or `"pkg @ https://..."`). PyPI rejects these in
# uploaded metadata, so we strip the entire list item from pyproject.toml at
# build time. Local dev installs that consume the workspace tree still see the
# original git pin (patched_tree reverts on exit).
_DIRECT_URL_DEP_RE = re.compile(
    r'^\s+"[^"]+?\s*@\s*(?:git\+|https?://)[^"]+",?[ \t]*\n',
    re.MULTILINE,
)


def _strip_direct_url_deps(text: str) -> str:
    return _DIRECT_URL_DEP_RE.sub("", text)


# Path of the marin-iris build-info module that gets stamped with the build
# date during wheel builds. iris.version reads BUILD_DATE from this file to
# populate LaunchJobRequest.client_revision_date so the controller can reject
# stale clients. Editable installs leave this empty and fall back to git log.
IRIS_BUILD_INFO_PATH = REPO_ROOT / "lib" / "iris" / "src" / "iris" / "_build_info.py"


def _stamp_iris_build_date() -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return f'# Auto-generated by scripts/python_libs_package.py during wheel builds.\n\nBUILD_DATE = "{today}"\n'


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

            # Then sibling-pin rewrite + direct-URL strip on pyproject.toml.
            # Re-read in case the version patch already wrote pyproject.
            # The strip pass keeps PyPI-uploaded metadata PEP 440 compliant by
            # removing entries like `lm-eval @ git+https://...` from optional
            # extras; those extras become empty in the published artifacts.
            current_pyproject = pyproject_path.read_text()
            new_pyproject = _rewrite_sibling_pins(current_pyproject, version)
            new_pyproject = _strip_direct_url_deps(new_pyproject)
            if new_pyproject != current_pyproject:
                pyproject_path.write_text(new_pyproject)

        originals[IRIS_BUILD_INFO_PATH] = IRIS_BUILD_INFO_PATH.read_text()
        IRIS_BUILD_INFO_PATH.write_text(_stamp_iris_build_date())
        yield
    finally:
        for path, text in originals.items():
            path.write_text(text)


# ---------- version resolution -----------------------------------------------


def _version_key(version: str) -> tuple[int, ...]:
    """Sort key for a dotted version; non-numeric segments count as 0."""
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-+]", version))


def _bump_patch(version: str) -> str:
    """Return one patch above `version` (e.g. 0.1.0 -> 0.1.1)."""
    parts = [int(p) for p in re.split(r"[.\-+]", version) if p.isdigit()][:3]
    parts += [0] * (3 - len(parts))
    major, minor, patch = parts
    return f"{major}.{minor}.{patch + 1}"


def _highest_declared_version() -> str:
    """Highest version currently declared across the eight libs.

    All eight share one synthetic version per build so cross-pins resolve
    cleanly; the declared versions are the floor that synthetic value sits on.
    """
    return max((_read_base_version(p) for p in PACKAGES), key=_version_key)


def _latest_pypi_stable(pkg: str) -> str | None:
    """Latest non-prerelease version of `pkg` on PyPI, or None if unregistered.

    PyPI's `info.version` reports the latest stable (it skips pre-releases per
    its own conventions), which is exactly what we want as the bump base.
    """
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return data.get("info", {}).get("version") or None


def _dev_base() -> str:
    """Patch-bumped base shared by nightly and vendor builds.

    One patch above max(highest declared version, highest stable on PyPI
    across the libs). PEP 440 orders `<base>.devN` above the current stable,
    so `pip install --pre` / `uv` resolve a dev build in preference to the
    last release. Querying PyPI means the declared versions never need
    re-bumping after a stable cut -- the script always anticipates the next
    patch correctly.
    """
    base = _highest_declared_version()
    for pkg in PACKAGES:
        stable = _latest_pypi_stable(pkg)
        if stable and _version_key(stable) > _version_key(base):
            base = stable
    return _bump_patch(base)


def resolve_version(mode: str, explicit: str | None) -> str:
    """Return the build version for the requested mode.

    nightly -> <dev_base>.dev<YYYYMMDDhhmm>
    stable  -> <explicit>
    manual  -> <declared>+manual.<sha>
    vendor  -> <dev_base>.dev<YYYYMMDDHHMMSS>
    """
    if mode == "stable":
        if not explicit:
            raise SystemExit("--version is required for --mode stable")
        return explicit
    if mode == "nightly":
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M")
        return f"{_dev_base()}.dev{stamp}"
    if mode == "manual":
        return f"{_highest_declared_version()}+manual.{_git_short_sha()}"
    if mode == "vendor":
        # Second-precision timestamp guarantees the freshly-built wheel beats
        # any nightly built earlier today, so `uv sync` in the consumer always
        # picks up the local copy without cache games.
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{_dev_base()}.dev{stamp}"
    raise SystemExit(f"Unknown mode: {mode}")


# ---------- build ------------------------------------------------------------


def build_wheels(version: str) -> None:
    """Build all marin-* wheels + sdists into DIST_DIR with `version` patched in."""
    _check_tool("uv", "https://docs.astral.sh/uv/")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    with patched_tree(version):
        for name, info in PACKAGES.items():
            pkg_dir = REPO_ROOT / info["path"]
            print(f"\n--- Building {name} ({version}) ---")
            subprocess.run(
                ["uv", "build", "--wheel", "--sdist", "--out-dir", str(DIST_DIR), str(pkg_dir)],
                check=True,
                cwd=REPO_ROOT,
            )

    wheels = sorted(DIST_DIR.glob("*.whl"))
    sdists = sorted(DIST_DIR.glob("*.tar.gz"))
    print(f"\nBuilt {len(wheels)} wheel(s) and {len(sdists)} sdist(s):")
    for f in (*wheels, *sdists):
        print(f"  {f.name}")
    if len(wheels) != len(PACKAGES):
        raise RuntimeError(f"Expected {len(PACKAGES)} wheels, got {len(wheels)}")
    if len(sdists) != len(PACKAGES):
        raise RuntimeError(f"Expected {len(PACKAGES)} sdists, got {len(sdists)}")


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


def lock_consumer(project_dir: Path) -> None:
    """Re-lock the consumer project so it picks up the freshly-vendored wheels.

    uv lock preserves existing resolutions when constraints are already
    satisfied, so a plain `uv lock` after vendoring keeps the old version.
    --upgrade-package for each marin-* package forces re-resolution against
    the new wheels in the vendor find-links directory.
    """
    upgrade_flags: list[str] = []
    for pkg in PACKAGES:
        upgrade_flags += ["--upgrade-package", pkg]
    print(f"\nRe-locking {project_dir} ...")
    subprocess.run(["uv", "lock", *upgrade_flags], check=True, cwd=project_dir)


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["nightly", "stable", "manual", "vendor"], default="nightly")
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Explicit version. Required for --mode stable; for the other modes it "
            "overrides the computed value. CI's resolve job computes the version once "
            "and passes it here so the build job stamps the identical value."
        ),
    )
    parser.add_argument(
        "--vendor",
        type=Path,
        default=None,
        help="Target directory to drop wheels into (required for --mode vendor)",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print the resolved version and emit it to $GITHUB_OUTPUT; do not build.",
    )
    args = parser.parse_args()

    version = args.version if args.version else resolve_version(args.mode, args.version)
    print(f"Mode:    {args.mode}\nVersion: {version}")
    _emit_github_output("version", version)

    if args.resolve_only:
        return

    if args.mode == "vendor":
        if args.vendor is None:
            raise SystemExit("--vendor PATH is required for --mode vendor")
        build_wheels(version)
        vendor_target = args.vendor.expanduser().resolve()
        vendor_copy(vendor_target)
        lock_consumer(vendor_target.parent)
        print("\nDone.")
        return

    build_wheels(version)
    print(f"\nBuild complete. Wheels + sdists in {DIST_DIR}/")


if __name__ == "__main__":
    main()
