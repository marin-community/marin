#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register Marin package names on PyPI.

Builds each workspace package and uploads it to PyPI to claim the distribution
name. Packages uploaded here may not be installable standalone — the goal is
name registration and anti-squatting.

Requires a PyPI API token with account-wide scope (project-scoped tokens can't
create new projects). Create one at https://pypi.org/manage/account/token/

Usage:
    UV_PUBLISH_TOKEN=pypi-xxxx python scripts/infra/register_pypi_packages.py
    python scripts/infra/register_pypi_packages.py --dry-run    # build only
    python scripts/infra/register_pypi_packages.py marin-iris   # single package
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Packages buildable from this workspace via `uv build --package <name>`.
WORKSPACE_PACKAGES: list[str] = [
    "marin",
    "marin-finelog",
    "marin-fray",
    "marin-haliax",
    "marin-iris",
    "marin-levanter",
    "marin-rigging",
    "marin-zephyr",
]

# Packages not buildable from this workspace. Registered via minimal stubs.
STUB_PACKAGES: dict[str, str] = {
    "marin-dupekit": "Optimized text de-duplication, written in Rust",
    "marin-kitoken": "Tokenizer library for Marin",
}

ALL_PACKAGES: list[str] = sorted(WORKSPACE_PACKAGES + list(STUB_PACKAGES))


def package_exists_on_pypi(name: str) -> bool:
    try:
        urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=10)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def build_workspace_package(name: str, out_dir: Path) -> list[Path]:
    subprocess.run(
        ["uv", "build", "--package", name, "--sdist", "--out-dir", str(out_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
    return sorted(out_dir.glob("*"))


def build_stub_package(name: str, description: str, out_dir: Path) -> list[Path]:
    module_name = name.replace("-", "_")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "pyproject.toml").write_text(
            f"""\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.0.1"
description = "{description}"
requires-python = ">=3.11"
license = {{ file = "LICENSE" }}

[tool.hatch.build.targets.wheel]
packages = ["{module_name}"]
"""
        )
        (tmp_path / "LICENSE").write_text("Apache-2.0\n")
        pkg_dir = tmp_path / module_name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(out_dir)],
            check=True,
            cwd=tmp_path,
        )
    return sorted(out_dir.glob("*"))


def _sdists(dist_dir: Path) -> list[Path]:
    return sorted(dist_dir.glob("*.tar.gz"))


def publish(dist_dir: Path) -> None:
    files = _sdists(dist_dir)
    if not files:
        print("  No artifacts to upload", file=sys.stderr)
        return
    subprocess.run(["uv", "publish", *[str(f) for f in files]], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="*", default=ALL_PACKAGES, help="Package names to register (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Build only, don't upload")
    args = parser.parse_args()

    if not args.dry_run:
        # uv publish reads UV_PUBLISH_TOKEN automatically
        if not os.environ.get("UV_PUBLISH_TOKEN"):
            print("ERROR: Set UV_PUBLISH_TOKEN to a PyPI API token (account-scoped).", file=sys.stderr)
            print("  Create one at https://pypi.org/manage/account/token/", file=sys.stderr)
            sys.exit(1)

    unknown = set(args.packages) - set(ALL_PACKAGES)
    if unknown:
        print(f"ERROR: Unknown packages: {', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"  Known packages: {', '.join(ALL_PACKAGES)}", file=sys.stderr)
        sys.exit(1)

    registered = []
    skipped = []
    failed = []

    for name in args.packages:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        if package_exists_on_pypi(name):
            print("  Already on PyPI — skipping")
            skipped.append(name)
            continue

        dist_dir = Path(tempfile.mkdtemp(prefix=f"marin-pypi-{name}-"))
        try:
            if name in STUB_PACKAGES:
                print("  Building stub...")
                build_stub_package(name, STUB_PACKAGES[name], dist_dir)
            else:
                print("  Building from workspace...")
                build_workspace_package(name, dist_dir)

            for f in _sdists(dist_dir):
                print(f"  Built: {f.name}")

            if args.dry_run:
                print("  [dry-run] Skipping upload")
                registered.append(name)
            else:
                print("  Uploading...")
                publish(dist_dir)
                print("  Registered!")
                registered.append(name)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(name)
        finally:
            shutil.rmtree(dist_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    if skipped:
        print(f"  Already on PyPI: {', '.join(skipped)}")
    if registered:
        print(f"  Registered: {', '.join(registered)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        sys.exit(1)
    if not registered and not failed:
        print("  Nothing to do — all packages already registered.")


if __name__ == "__main__":
    main()
