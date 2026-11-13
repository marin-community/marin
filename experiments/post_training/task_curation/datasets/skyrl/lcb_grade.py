# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a code reply with the vendored SkyRL LiveCodeBench evaluator: 1 when every hidden case passes.

``test_cases`` in ``/tests/config.json`` are the evaluator's canonical cases. The evaluator runs the
program in a child process that shares this script's stdout, so stdout points at stderr while it
runs and carries only the reward.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

import livecodebench


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = Path("/app/answer.txt").read_text()
    stdout = os.dup(1)
    os.dup2(2, 1)
    _, reward = livecodebench.compute_score(answer, config["test_cases"])
    sys.stdout.flush()
    os.dup2(stdout, 1)
    print(reward)


if __name__ == "__main__":
    main()
