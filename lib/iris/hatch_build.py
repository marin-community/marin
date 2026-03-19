# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hatchling custom build hook for Iris.

Regenerates protobuf files from .proto sources when source files are newer
than their generated outputs. This runs automatically during ``uv sync`` /
``pip install -e .`` / wheel builds, eliminating the need to check generated
files into git or manually run build steps.

Dashboard assets are built separately via ``iris build dashboard`` or
``_ensure_dashboard_dist()`` in the Docker image build pipeline.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)

# Glob patterns for source and generated files, relative to the iris package root.
_PROTO_SOURCE_GLOBS = ["src/iris/rpc/*.proto"]
_PROTO_OUTPUT_GLOBS = ["src/iris/rpc/*_pb2.py", "src/iris/rpc/*_pb2.pyi", "src/iris/rpc/*_connect.py"]


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


def _has_missing_outputs(root: Path, source_globs: list[str]) -> bool:
    """Return True if any .proto source is missing its corresponding _pb2.py."""
    for pattern in source_globs:
        for proto_path in root.glob(pattern):
            pb2_path = proto_path.with_name(proto_path.stem + "_pb2.py")
            if not pb2_path.exists():
                return True
    return False


def _needs_rebuild(root: Path, source_globs: list[str], output_globs: list[str]) -> bool:
    """Return True if any source file is strictly newer than the oldest output file.

    Uses a 60-second tolerance because zip archives (used by task bundles)
    can extract files with slightly different timestamps, causing spurious
    rebuilds.
    """
    source_newest = _newest_mtime(root, source_globs)
    output_oldest = _oldest_mtime(root, output_globs)
    return source_newest > output_oldest + 60.0


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "iris-build"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        self._maybe_generate_protos(root)

    def _maybe_generate_protos(self, root: Path) -> None:
        outputs_complete = not _has_missing_outputs(root, _PROTO_SOURCE_GLOBS)

        if outputs_complete and not _needs_rebuild(root, _PROTO_SOURCE_GLOBS, _PROTO_OUTPUT_GLOBS):
            logger.info("Protobuf outputs are up-to-date, skipping generation")
            return

        generate_script = root / "scripts" / "generate_protos.py"
        if not generate_script.exists():
            if not outputs_complete:
                raise RuntimeError(
                    "Protobuf outputs are missing and scripts/generate_protos.py not found. "
                    "Cannot build iris without generated protobuf files."
                )
            logger.warning("scripts/generate_protos.py not found, using existing protobuf outputs")
            return

        if shutil.which("npx") is None:
            if not outputs_complete:
                raise RuntimeError(
                    "Protobuf outputs are missing and npx is not installed. "
                    "Install Node.js (which provides npx) to generate protobuf files: "
                    "https://nodejs.org/ or run `make install_node`"
                )
            logger.warning("npx not found, using existing (possibly stale) protobuf outputs")
            return

        logger.info("Regenerating protobuf files from .proto sources...")
        result = subprocess.run(
            [sys.executable, str(generate_script)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Protobuf generation failed:\n{result.stdout}\n{result.stderr}")
        logger.info("Protobuf generation complete")
