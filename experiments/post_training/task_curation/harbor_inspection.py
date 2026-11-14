# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect immutable curation artifacts through Harbor's locked production runtime."""

import json
import subprocess
from pathlib import Path

HARBOR_PROJECT = Path(__file__).resolve().parents[3] / "vendor/harbor"


def harbor_inspection(kind: str, paths: list[str]) -> dict:
    """Validate in Harbor's locked runtime without importing it into the curation environment."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(HARBOR_PROJECT),
            "--frozen",
            "--extra",
            "taskcompendium",
            "python",
            "-m",
            "harbor.cli.curation_inspect",
            kind,
        ],
        input=json.dumps(paths),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)
