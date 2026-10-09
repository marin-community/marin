#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the pure-Python Marin library wheels for PyPI publication.

Builds the Python library family in `scripts/ci/package_release.py` into dist/.
The package release engine passes one exact version to this builder.
Publication is done by `.github/workflows/marin-release-libs-wheels.yaml` via
`pypa/gh-action-pypi-publish` with OIDC trusted publishing. This script never
uploads anything and never needs a token.

Native package families use separate build legs in the same release workflow.

Two modes:
    stable   -- build the exact version supplied by the package release engine.
    vendor   -- version becomes <dev_base>.dev<YYYYMMDDHHMMSS>; copy wheels to
                a local directory (no publish). For local-iteration loops
                where a marin worktree feeds wheels into an experiment repo's
                find-links. The second-precision timestamp guarantees rebuilt
                wheels beat any published development release from earlier
                the same day.

Usage:
    python -m scripts.python_libs_package --mode stable --version 0.2.0
    python -m scripts.python_libs_package --mode vendor --vendor ../tiny-tpu/vendor

The build is done from a temporary in-place patch of each package's version
file plus a cross-pin rewrite of every sibling dependency, so the wheels
published together always require each other at the exact same version.
Mutations are reverted on exit (success OR failure) so the working tree stays
clean.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from scripts.ci.package_release import PACKAGES as RELEASE_PACKAGES
from scripts.ci.package_release import PYTHON_LIBS_FAMILY

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


class VersionFileKind(StrEnum):
    PYPROJECT = "pyproject"
    ABOUT_PY = "about_py"


@dataclass(frozen=True)
class PythonLibrary:
    directory: Path
    version_file: Path
    kind: VersionFileKind


def _python_libraries() -> dict[str, PythonLibrary]:
    family = RELEASE_PACKAGES[PYTHON_LIBS_FAMILY]
    libraries = {}
    for version_file in family.declared_version_paths:
        if version_file.parts[0] != "lib" or len(version_file.parts) < 3:
            raise ValueError(f"Python library version file must be below lib/: {version_file}")
        directory = Path(*version_file.parts[:2])
        manifest = tomllib.loads((REPO_ROOT / directory / "pyproject.toml").read_text())
        name = manifest["project"]["name"]
        if version_file.name == "pyproject.toml":
            kind = VersionFileKind.PYPROJECT
        elif version_file.name == "__about__.py":
            kind = VersionFileKind.ABOUT_PY
        else:
            raise ValueError(f"Unsupported Python library version file: {version_file}")
        libraries[name] = PythonLibrary(directory=directory, version_file=version_file, kind=kind)
    if set(libraries) != set(family.artifacts):
        raise ValueError("Python library version files and release artifacts disagree")
    return libraries


PACKAGES: Mapping[str, PythonLibrary] = MappingProxyType(_python_libraries())


# ---------- helpers ----------------------------------------------------------


def _check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found. Install with: {install_hint}", file=sys.stderr)
        sys.exit(1)


def _read_base_version(pkg: str) -> str:
    info = PACKAGES[pkg]
    path = REPO_ROOT / info.version_file
    text = path.read_text()
    if info.kind is VersionFileKind.PYPROJECT:
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    else:
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not read version from {path}")
    return m.group(1)


def _set_version(text: str, kind: VersionFileKind, new_version: str) -> str:
    if kind is VersionFileKind.PYPROJECT:
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


_SIBLING_REQUIREMENT_RE = re.compile(r"^(?P<name>marin-[\w.-]+)(?P<extras>\[[^]]+\])?(?:\s*[<>=!~].*)?$")


def _rewrite_sibling_pins(text: str, version: str) -> str:
    project = tomllib.loads(text)["project"]
    requirements = list(project.get("dependencies", ()))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    for requirement in requirements:
        package, separator, marker = requirement.partition(";")
        match = _SIBLING_REQUIREMENT_RE.fullmatch(package.strip())
        if match is None or match.group("name") not in PACKAGES:
            continue
        pinned = f'{match.group("name")}{match.group("extras") or ""}=={version}'
        if separator:
            pinned += f";{marker}"
        text = text.replace(f'"{requirement}"', f'"{pinned}"')
    return text


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
            pyproject_path = REPO_ROOT / info.directory / "pyproject.toml"
            version_path = REPO_ROOT / info.version_file

            if pyproject_path not in originals:
                originals[pyproject_path] = pyproject_path.read_text()
            if version_path not in originals:
                originals[version_path] = version_path.read_text()

            # Apply version patch first; for haliax this writes __about__.py
            # (separate file from pyproject), for the rest it overwrites the
            # pyproject we just snapshotted above.
            patched_version = _set_version(originals[version_path], info.kind, version)
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
    """Highest version currently declared across the libs.

    They all share one synthetic version per build so cross-pins resolve
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
    """Return the patch-bumped base for a local vendor build.

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


def vendor_version() -> str:
    """Return a development version that wins local vendor resolution."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{_dev_base()}.dev{stamp}"


# ---------- build ------------------------------------------------------------


def build_wheels(version: str) -> None:
    """Build all library wheels and sdists into DIST_DIR with `version` patched in."""
    _check_tool("uv", "https://docs.astral.sh/uv/")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    with patched_tree(version):
        for name, info in PACKAGES.items():
            pkg_dir = REPO_ROOT / info.directory
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
    """Replace this bundle's wheels in target, leaving unrelated files alone."""
    target.mkdir(parents=True, exist_ok=True)
    stale = sorted(wheel for package in PACKAGES for wheel in target.glob(f"{package.replace('-', '_')}-*.whl"))
    for s in stale:
        s.unlink()
    if stale:
        print(f"\nRemoved {len(stale)} stale library wheel(s) from {target}")
    print(f"\nCopying wheels to {target}:")
    for wheel in sorted(DIST_DIR.glob("*.whl")):
        dest = target / wheel.name
        shutil.copy2(wheel, dest)
        print(f"  -> {dest.name}")


def lock_consumer(project_dir: Path) -> None:
    """Select the vendored bundle's wheel versions in a consumer lock."""
    upgrade_flags: list[str] = []
    for pkg in PACKAGES:
        upgrade_flags += ["--upgrade-package", pkg]
    print(f"\nRe-locking {project_dir} ...")
    subprocess.run(["uv", "lock", *upgrade_flags], check=True, cwd=project_dir)


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["stable", "vendor"], default="stable")
    parser.add_argument(
        "--version",
        default=None,
        help=("Exact version. Required for --mode stable; overrides the generated " "local version in vendor mode."),
    )
    parser.add_argument(
        "--vendor",
        type=Path,
        default=None,
        help="Target directory to drop wheels into (required for --mode vendor)",
    )
    args = parser.parse_args()

    if args.mode == "stable" and not args.version:
        raise SystemExit("--version is required for --mode stable")
    version = args.version or vendor_version()
    print(f"Mode:    {args.mode}\nVersion: {version}")

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
