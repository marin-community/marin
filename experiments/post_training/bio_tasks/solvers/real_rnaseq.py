# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the pinned R statistical engine from public task observations."""

import json
import subprocess
from functools import partial
from importlib.resources import as_file, files
from pathlib import Path


def solve_rnaseq(inputs: Path, output: Path, operation: str) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    if query["analysis"] != operation:
        raise ValueError("RNA-seq query and requested oracle disagree")
    answer = output / "answer.json"
    with as_file(files(__package__).joinpath("real_rnaseq.R")) as script:
        subprocess.run(
            ["Rscript", "--vanilla", str(script), str(inputs), str(output), str(answer)],
            check=True,
            timeout=180,
        )
    return json.loads(answer.read_text())


OUTPUT_SOLVERS = {
    name: partial(solve_rnaseq, operation=name)
    for name in ("real-rnaseq-differential-expression", "real-rnaseq-go-enrichment")
}
