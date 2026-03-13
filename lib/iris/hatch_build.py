# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hatchling custom build hook for Iris.

Regenerates protobuf files from .proto sources and rebuilds the Vue dashboard
when source files are newer than their generated outputs. This runs automatically
during ``uv sync`` / ``pip install -e .`` / wheel builds, eliminating the need
to check generated files into git or manually run build steps.
"""

import logging
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)

# Glob patterns for source and generated files, relative to the iris package root.
_PROTO_SOURCE_GLOBS = ["src/iris/rpc/*.proto"]
_PROTO_OUTPUT_GLOBS = ["src/iris/rpc/*_pb2.py", "src/iris/rpc/*_pb2.pyi", "src/iris/rpc/*_connect.py"]

_DASHBOARD_SOURCE_GLOBS = ["dashboard/src/**/*", "dashboard/package.json", "dashboard/rsbuild.config.ts"]
_DASHBOARD_OUTPUT_DIR = "dashboard/dist"


def _newest_mtime(root: Path, globs: list[str]) -> float:
    """Return the newest mtime across all files matching the given globs."""
    newest = 0.0
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
    return newest


def _oldest_mtime(root: Path, globs: list[str]) -> float:
    """Return the oldest mtime across all files matching the given globs.

    Returns 0.0 if no files match (meaning outputs don't exist yet).
    """
    oldest = float("inf")
    found = False
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                found = True
                oldest = min(oldest, path.stat().st_mtime)
    return oldest if found else 0.0


def _needs_rebuild(root: Path, source_globs: list[str], output_globs: list[str]) -> bool:
    """Return True if any source file is newer than the oldest output file."""
    source_newest = _newest_mtime(root, source_globs)
    output_oldest = _oldest_mtime(root, output_globs)
    return source_newest > output_oldest


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "iris-build"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        self._maybe_generate_protos(root)
        self._maybe_build_dashboard(root)

    def _maybe_generate_protos(self, root: Path) -> None:
        if not _needs_rebuild(root, _PROTO_SOURCE_GLOBS, _PROTO_OUTPUT_GLOBS):
            logger.info("Protobuf outputs are up-to-date, skipping generation")
            return

        generate_script = root / "scripts" / "generate_protos.py"
        if not generate_script.exists():
            logger.warning("scripts/generate_protos.py not found, skipping protobuf generation")
            return

        logger.info("Regenerating protobuf files from .proto sources...")
        result = subprocess.run(
            [sys.executable, str(generate_script)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Protobuf generation failed:\n{result.stdout}\n{result.stderr}"
            )
        logger.info("Protobuf generation complete")

    def _maybe_build_dashboard(self, root: Path) -> None:
        dashboard_dir = root / "dashboard"
        if not (dashboard_dir / "package.json").exists():
            logger.info("Dashboard source not found, skipping build")
            return

        dist_dir = root / _DASHBOARD_OUTPUT_DIR
        source_newest = _newest_mtime(root, _DASHBOARD_SOURCE_GLOBS)
        if dist_dir.exists() and source_newest > 0:
            output_oldest = _oldest_mtime(root, [f"{_DASHBOARD_OUTPUT_DIR}/**/*"])
            if output_oldest > 0 and source_newest <= output_oldest:
                logger.info("Dashboard assets are up-to-date, skipping build")
                return

        logger.info("Building dashboard assets...")
        result = subprocess.run(["npm", "ci"], cwd=dashboard_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"npm ci failed:\n{result.stdout}\n{result.stderr}")

        result = subprocess.run(["npm", "run", "build"], cwd=dashboard_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Dashboard build failed:\n{result.stdout}\n{result.stderr}")
        logger.info("Dashboard build complete")
