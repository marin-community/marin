# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hatchling custom build hook for Iris.

Regenerates protobuf files from .proto sources unconditionally when the
generate script and npx are available. This runs automatically during
``uv sync`` / ``pip install -e .`` / wheel builds.

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

# Glob patterns for source files, relative to the iris package root.
_PROTO_SOURCE_GLOBS = ["src/iris/rpc/*.proto"]


def _has_missing_outputs(root: Path, source_globs: list[str]) -> bool:
    """Return True if any .proto source is missing its corresponding _pb2.py."""
    for pattern in source_globs:
        for proto_path in root.glob(pattern):
            pb2_path = proto_path.with_name(proto_path.stem + "_pb2.py")
            if not pb2_path.exists():
                return True
    return False


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "iris-build"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        self._generate_protos(root)

    def _generate_protos(self, root: Path) -> None:
        outputs_missing = _has_missing_outputs(root, _PROTO_SOURCE_GLOBS)

        generate_script = root / "scripts" / "generate_protos.py"
        if not generate_script.exists():
            if outputs_missing:
                raise RuntimeError(
                    "Protobuf outputs are missing and scripts/generate_protos.py not found. "
                    "Cannot build iris without generated protobuf files."
                )
            logger.warning("scripts/generate_protos.py not found, using existing protobuf outputs")
            return

        if shutil.which("npx") is None:
            if outputs_missing:
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
