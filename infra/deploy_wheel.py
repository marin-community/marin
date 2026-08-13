#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a native (Rust) wheel from the current tree and publish it so Iris jobs
can install it by pinning a version — no full PyPI release, no custom task image.

Deploying a Rust change to Iris normally means either cutting a real
marin-dupekit-native release (tag + PyPI, slow) or baking a one-off task image
(annoying). This script closes that gap: it builds the manylinux wheel exactly
the way CI does (reusing ``scripts/ci/package_release.py``), then uploads it as an
asset on a single rolling GitHub *prerelease* (default tag ``dev-wheels``)
alongside a flat PEP 503 "find-links" ``index.html``. A job installs it by
pointing uv at that index and pinning the exact version this script prints.

Because ``iris-task`` runs ``uv sync`` (then ``uv pip install`` of any
``pip_packages``) at task startup and inherits the job's ``env_vars``, a job can
pick up the wheel with no Iris code change:

    env_vars={"UV_FIND_LINKS": "<index-url printed below>"}
    pip_packages=["marin-dupekit-native==<version printed below>"]

The wheel version is a ``.devN`` build whose serial is a content hash of the Rust
source tree, so re-running on the same source is idempotent and uncommitted edits
get a distinct version. Consumers pin it exactly, so it never collides with the
released nightly on PyPI.

Usage:
    uv run infra/deploy_wheel.py dupekit                 # build x86_64 + aarch64, publish
    uv run infra/deploy_wheel.py dupekit --arch x86_64   # host arch only (fast)
    uv run infra/deploy_wheel.py dupekit --dry-run       # build + validate, publish nothing
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# scripts/ci is not an importable package; add it to the path so this infra
# script reuses the exact release build (maturin + zig manylinux, version
# stamping, wheel validation) rather than forking a second copy of that logic.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
import package_release as pr  # noqa: E402  # pyrefly: ignore[missing-import]

DEFAULT_REPO = "marin-community/marin"
DEFAULT_TAG = "dev-wheels"
INDEX_PREFIX = "index-"
ARCH_BY_TRIPLE = {
    "x86_64-unknown-linux-gnu": "x86_64",
    "aarch64-unknown-linux-gnu": "aarch64",
}
HOST_ARCH = "x86_64"
RELEASE_TITLE = "Dev native wheels"
RELEASE_NOTES = (
    "Auto-managed bucket of development native (Rust) wheels published by "
    "`infra/deploy_wheel.py`. Assets are prereleases pinned by exact version; "
    "they are not real releases. Guide: https://echo.oa.dev/wiki/146"
)


def native_families() -> list[str]:
    """Package families in package_release.py that build a native wheel."""
    return sorted(name for name, family in pr.PACKAGES.items() if isinstance(family.build, pr.NativeBuild))


def _native_source_files(build: pr.NativeBuild) -> list[Path]:
    """Every build input under the native crate, excluding the cargo target dir."""
    root = REPO_ROOT / build.native_path
    files = []
    for path in root.rglob("*"):
        if path.is_file() and "target" not in path.relative_to(root).parts:
            files.append(path)
    return sorted(files)


def source_serial(build: pr.NativeBuild) -> int:
    """Deterministic serial hashing the crate's working-tree contents.

    Reads the working tree (not git), so uncommitted edits change the serial and
    an unchanged tree reproduces the same version.
    """
    digest = hashlib.blake2b(digest_size=16)
    for path in _native_source_files(build):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return int(digest.hexdigest()[:12], 16)


def _native_floor(build: pr.NativeBuild) -> str:
    """The ``>=`` version floor the pure package declares for its native companion."""
    text = (REPO_ROOT / build.requirement_path).read_text()
    match = re.search(rf'"{re.escape(build.requirement_distribution)}\s*>=\s*([^"]+)"', text)
    if match is None:
        raise ValueError(f"No {build.requirement_distribution!r} floor in {build.requirement_path}")
    return pr.canonical_version(match.group(1).strip())


def resolve_version(package: str, build: pr.NativeBuild) -> str:
    """Resolve a ``.devN`` version above the current floor, serial from source.

    Bumping the patch above ``max(declared, floor)`` keeps the build sorting
    newer than anything on PyPI, and the content-hash serial makes it unique.
    Consumers pin it exactly, so ordering only matters for readability.
    """
    declared = pr._declared_version(REPO_ROOT, pr.PACKAGES[package])
    base = max((declared, _native_floor(build)), key=pr._version_key)
    major, minor, patch, _, _ = pr._version_key(base)
    return f"{major}.{minor}.{patch + 1}.dev{source_serial(build)}"


def build_wheels(build: pr.NativeBuild, version: str, arches: list[str], dist_dir: Path) -> list[Path]:
    """Build manylinux wheels for the requested arches, restoring stamped files."""
    # Stamping the version dirties several tracked files: the crate pyproject and
    # Cargo.toml, the crate Cargo.lock (cargo rewrites it during the build), and
    # the root uv.lock (uv rewrites the pure package's editable entry). Snapshot
    # and restore all of them so a build leaves the working tree clean.
    restore = [REPO_ROOT / path for path in build.version_paths]
    restore.append(REPO_ROOT / build.native_path / "Cargo.lock")
    restore.append(REPO_ROOT / "uv.lock")
    snapshot = {path: path.read_bytes() for path in restore if path.exists()}
    triples = [(triple, manylinux) for triple, manylinux in pr.LINUX_TARGETS if ARCH_BY_TRIPLE[triple] in arches]
    try:
        pr._stamp_versions(REPO_ROOT, build, pr.cargo_compatible_version(version))
        subprocess.run(["rustup", "target", "add", *(triple for triple, _ in triples)], check=True)
        zig_dir = str(Path(pr._ensure_zig(REPO_ROOT)).parent)
        env = {**os.environ, "PATH": f"{zig_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
        for triple, manylinux in triples:
            args = ["build", "--release", "--out", str(dist_dir), "--target", triple, "--zig"]
            if manylinux:
                args += ["--manylinux", manylinux]
            pr._maturin(REPO_ROOT, build, *args, env=env)
    finally:
        for path, data in snapshot.items():
            path.write_bytes(data)
    prefix = pr._normalized_distribution(build.requirement_distribution)
    return sorted(dist_dir.glob(f"{prefix}-*.whl"))


def release_exists(repo: str, tag: str) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "tagName"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_release(repo: str, tag: str) -> None:
    if release_exists(repo, tag):
        return
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--prerelease",
            "--title",
            RELEASE_TITLE,
            "--notes",
            RELEASE_NOTES,
            "--target",
            "main",
        ],
        check=True,
    )


def upload_assets(repo: str, tag: str, paths: list[Path]) -> None:
    subprocess.run(
        ["gh", "release", "upload", tag, *(str(path) for path in paths), "--repo", repo, "--clobber"],
        check=True,
    )


def asset_url(repo: str, tag: str, name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def index_name(wheel_names: list[str]) -> str:
    """Content-addressed find-links page name for a set of wheels.

    The name hashes the wheel set, so a stable filename never serves a stale
    page. GitHub CDN-caches release-asset URLs, so a fixed ``index.html`` would
    keep serving an old wheel list after a new deploy; a content-addressed name
    yields a fresh URL whenever the listed wheels change.
    """
    digest = hashlib.blake2b("\n".join(sorted(wheel_names)).encode(), digest_size=8).hexdigest()
    return f"{INDEX_PREFIX}{digest}.html"


def render_index(repo: str, tag: str, wheel_names: list[str]) -> str:
    """Flat find-links page with absolute asset URLs.

    Hrefs must be absolute: GitHub redirects an asset download to a short-lived
    signed CDN URL, so a relative href would resolve against that redirected base
    and lose the signature. Absolute github.com download URLs each redirect
    correctly to the signed asset.
    """
    links = "\n".join(f'    <a href="{asset_url(repo, tag, name)}">{name}</a><br>' for name in wheel_names)
    return f"<!DOCTYPE html>\n<html>\n  <body>\n{links}\n  </body>\n</html>\n"


def print_recipe(build: pr.NativeBuild, version: str, url: str) -> None:
    distribution = build.requirement_distribution
    pin = f"{distribution}=={version}"
    print("\n" + "=" * 78)
    print(f"Published {distribution} {version}")
    print(f"  find-links index: {url}")
    print("=" * 78)
    print("\nConsume from an Iris job (fray EnvConfig) — no repo edit, no relock:")
    print("    EnvConfig(")
    print(f'        env_vars={{"UV_FIND_LINKS": "{url}"}},')
    print(f'        pip_packages=["{pin}"],')
    print("    )")
    print("\nOr repo-wide: set the floor in", build.requirement_path, f'to "{pin}",')
    print(f'    add UV_FIND_LINKS="{url}" to the job env, then `uv lock`.')
    print("\nVerify locally:")
    print(f'    uv pip install --find-links "{url}" "{pin}"')
    print(f'    python -c "import {build.import_name}"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", choices=native_families(), help="Native package family to build and publish.")
    parser.add_argument(
        "--arch",
        default="x86_64,aarch64",
        help="Comma-separated linux arches to build (default: x86_64,aarch64).",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo hosting the release.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Rolling prerelease tag to publish under.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate but publish nothing.")
    args = parser.parse_args()

    arches = [arch.strip() for arch in args.arch.split(",") if arch.strip()]
    unknown = set(arches) - set(ARCH_BY_TRIPLE.values())
    if unknown:
        parser.error(f"Unknown arch(es): {sorted(unknown)}. Choose from {sorted(set(ARCH_BY_TRIPLE.values()))}.")

    build = pr.PACKAGES[args.package].build
    version = resolve_version(args.package, build)
    print(f"Building {build.requirement_distribution}=={version} for {arches}")

    with tempfile.TemporaryDirectory(prefix="deploy-wheel-") as tmp:
        dist_dir = Path(tmp)
        wheels = build_wheels(build, version, arches, dist_dir)
        if not wheels:
            raise RuntimeError(f"No wheels produced in {dist_dir}")
        print("Built:", *(wheel.name for wheel in wheels), sep="\n  ")

        if HOST_ARCH in arches:
            pr.validate_native_wheel(args.package, version, dist_dir)
            print(f"Validated import of {build.import_name} from the {HOST_ARCH} wheel")

        names = sorted(wheel.name for wheel in wheels)
        url = asset_url(args.repo, args.tag, index_name(names))

        if args.dry_run:
            print("\n[dry-run] Skipping GitHub release. Would publish:")
            print("  ", "\n   ".join(names))
            print_recipe(build, version, url)
            return

        ensure_release(args.repo, args.tag)
        upload_assets(args.repo, args.tag, wheels)
        index_path = dist_dir / index_name(names)
        index_path.write_text(render_index(args.repo, args.tag, names))
        upload_assets(args.repo, args.tag, [index_path])
        print_recipe(build, version, url)


if __name__ == "__main__":
    main()
