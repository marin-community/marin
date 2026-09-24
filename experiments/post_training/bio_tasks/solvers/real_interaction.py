# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the complete population-by-stage model from observed task inputs."""

import json
import subprocess
from importlib.resources import as_file, files
from pathlib import Path


def solve_interaction(inputs: Path, output: Path) -> list[dict]:
    answer = output / "answer.json"
    with as_file(files(__package__).joinpath("real_interaction.R")) as script:
        subprocess.run(
            ["Rscript", "--vanilla", str(script), str(inputs), str(output), str(answer)], check=True, timeout=600
        )
    return json.loads(answer.read_text())
