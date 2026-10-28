# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the private source test program without executing the candidate."""

import json
from pathlib import Path

import numpy as np  # noqa: F401 - fail preparation if the declared test dependency is missing
from code_protocol import test_program


def main() -> None:
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    response = Path("/app/answer.txt").read_text()
    program = test_program(config["problem"], response, config["suite"], config.get("direction", ""))
    # The receipt and exception handler reject early exits, including os._exit(0).
    wrapper = (
        "try:\n"
        f"    exec(compile({program!r}, '<candidate>', 'exec'), {{}})\n"
        "except BaseException:\n"
        "    raise SystemExit(1)\n"
        "open('/app/passed', 'w').close()\n"
    )
    Path("/app/program.py").write_text(wrapper)


if __name__ == "__main__":
    main()
