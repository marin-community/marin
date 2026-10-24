#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["packaging>=25"]
# ///
"""Check every tracked uv lock for Brotli exposure and blocked LiteLLM packages.

Reads all package entries, including inactive extras and platform branches, and
validates the frozen graph with uv. Does not install project dependencies.
"""

import argparse
import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version

# PyPI ranges: https://github.com/advisories/GHSA-9v8h-qqgm-4m35
BROTLI_MINIMUM = {"brotli": Version("1.2.1"), "brotlicffi": Version("1.2.1.0")}


@dataclass
class LockResult:
    path: str
    findings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    packages: int = 0


def check_lock(path: Path, root: Path, uv: str) -> LockResult:
    """Inspect package versions and verify the existing graph without changing it."""
    result = LockResult(path=str(path.relative_to(root)))
    try:
        before = path.read_bytes()
        document = tomllib.loads(before.decode())
        packages = document["package"]
        result.packages = len(packages)
        for package in packages:
            name = canonicalize_name(package["name"])
            if name == "litellm":
                result.findings.append(f"litellm=={package.get('version', '?')}: blocked by Marin policy")
            elif name in BROTLI_MINIMUM:
                version = Version(package["version"])
                if version < BROTLI_MINIMUM[name]:
                    result.findings.append(f"{name}=={version}: CVE-2026-111497; requires >={BROTLI_MINIMUM[name]}")
        command = [
            uv,
            "tree",
            "--frozen",
            "--offline",
            "--universal",
            "--all-groups",
            "--format",
            "json",
            "--no-python-downloads",
            "--project",
            str(path.parent),
        ]
        tree = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if tree.returncode:
            result.errors.append(f"uv tree failed ({tree.returncode}): {tree.stderr.strip()}")
        else:
            json.loads(tree.stdout)
        if path.read_bytes() != before:
            result.errors.append("Lock changed during frozen audit")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        result.errors.append(f"{type(error).__name__}: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "uv.lock", "**/uv.lock"],
        check=True,
        capture_output=True,
        text=True,
    )
    locks = sorted(set(files.stdout.rstrip("\0").split("\0"))) if files.stdout else []
    results = [check_lock(root / path, root, args.uv) for path in locks]
    failed = not results or any(result.findings or result.errors for result in results)
    report = {"passed": not failed, "lock_count": len(results), "locks": [asdict(result) for result in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
