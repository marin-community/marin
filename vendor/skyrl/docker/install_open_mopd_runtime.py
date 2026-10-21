#!/usr/bin/env python3
"""Align the Open-MOPD image with its single runtime package manifest."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from open_mopd_versions import versions_match

PURE_PYTHON_OVERRIDES = (
    "absl-py",
    "appdirs",
    "emoji",
    "fsspec",
    "immutabledict",
    "jsonlines",
    "langdetect",
    "nltk",
    "protobuf",
    "ray",
    "s3fs",
    "syllapy",
    "tempdir",
    "transformers",
    "wget",
)


def expected_packages(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text())
    packages = config["environment"]["packages"]
    missing = set(PURE_PYTHON_OVERRIDES) - packages.keys()
    if missing:
        raise ValueError(f"Open-MOPD runtime manifest is missing packages: {sorted(missing)}")
    return packages


def install_and_verify(config_path: Path) -> None:
    packages = expected_packages(config_path)
    requirements = [f"{name}=={packages[name]}" for name in PURE_PYTHON_OVERRIDES]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *requirements], check=True)
    for distribution, expected in packages.items():
        actual = importlib.metadata.version(distribution)
        if not versions_match(expected, actual):
            raise RuntimeError(f"{distribution}: expected {expected}, found {actual}")
    import s3fs  # noqa: F401


if __name__ == "__main__":
    install_and_verify(Path(sys.argv[1]))
