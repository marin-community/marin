# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run source-prepared programs only within the private grading machine."""

import json
from pathlib import Path

import numpy as np  # noqa: F401
from code_protocol import test_program
from python_execution import DEFAULT_MEMORY_BYTES, PASSED, check_correctness


def main() -> None:
    tests = Path(__file__).parent
    config = json.loads((tests / "config.json").read_text())
    response = Path("/app/answer.txt").read_text()
    program = test_program(config["problem"], response, config["suite"], config.get("direction", ""))
    memory = DEFAULT_MEMORY_BYTES if config["suite"] == "humaneval" else None
    outcome = check_correctness(program, timeout=config["timeout"], maximum_memory_bytes=memory)
    print(float(outcome == PASSED))


if __name__ == "__main__":
    main()
