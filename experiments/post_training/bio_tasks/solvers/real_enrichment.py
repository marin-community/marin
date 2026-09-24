# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a fresh count, shrinkage and enrichment analysis from public task inputs."""

import json
import subprocess
from importlib.resources import as_file, files
from pathlib import Path


def solve_enrichment(inputs: Path, output: Path) -> list[dict]:
    with as_file(files(__package__).joinpath("real_enrichment.R")) as script:
        subprocess.run(["Rscript", "--vanilla", str(script), str(inputs), str(output)], check=True, timeout=1500)
    return json.loads((output / "answer.json").read_text())
