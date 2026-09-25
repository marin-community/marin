# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the native structural analysis in the task's pinned environment."""

import json
import subprocess
import sys
from importlib.resources import as_file, files
from pathlib import Path


def solve_heme_pocket(inputs: Path, output: Path) -> list[dict]:
    with as_file(files(__package__).joinpath("heme_pocket_native.py")) as script:
        subprocess.run(
            [sys.executable, str(script), "--inputs", str(inputs), "--output", str(output)],
            check=True,
            timeout=1200,
        )
    return json.loads((output / "answer.json").read_text())
