# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hatchling custom build hook for Iris.

Regenerates protobuf files from .proto sources when source files have changed
since the last generation. Generated files are checked into git, so this hook
only triggers a rebuild when .proto sources are modified. Requires ``npx``
(Node.js) and buf to be available; if they are absent and generated files are
already present, the existing outputs are used as-is.

Dashboard assets are built separately via ``iris build dashboard`` or
``_ensure_dashboard_dist()`` in the Docker image build pipeline.
"""

import hashlib
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


_CHECKSUM_FILE = "build/.proto_checksum"


def _source_digest(root: Path, source_globs: list[str]) -> str:
    """SHA-256 digest of all proto source contents, sorted by path for stability."""
    h = hashlib.sha256()
    paths = sorted(p for pattern in source_globs for p in root.glob(pattern) if p.is_file())
    for p in paths:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _needs_rebuild(root: Path, source_globs: list[str], output_globs: list[str]) -> bool:
    """Return True if proto sources have changed since the last generation.

    Uses a content-hash checksum file rather than mtime comparison, because
    git does not preserve file timestamps — after a pull both source and
    output files get the same mtime, hiding real staleness.

    Falls back to mtime comparison (with a 60-second tolerance for zip
    extraction jitter) when no checksum file exists yet.
    """
    checksum_path = root / _CHECKSUM_FILE
    current_digest = _source_digest(root, source_globs)
    if checksum_path.exists():
        return checksum_path.read_text().strip() != current_digest

    # No checksum file — fall back to mtime comparison for backwards compat
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
        # Write checksum so future builds can detect when sources change
        checksum_path = root / _CHECKSUM_FILE
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(_source_digest(root, _PROTO_SOURCE_GLOBS) + "\n")
        logger.info("Protobuf generation complete")
