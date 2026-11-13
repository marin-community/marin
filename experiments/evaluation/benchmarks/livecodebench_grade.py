# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a LiveCodeBench reply as the official harness does: 1 when the extracted program passes every test.

``extract_code`` is LiveCodeBench's chat-model extraction (``lcb_runner/utils/extraction_utils.py``):
the program between the last two lines that contain a code fence. The vendored scorer runs it against
``/tests/tests.json.zlib`` with the official per-test alarm and the process deadline from
``/tests/config.json``: the official ``(per_test_timeout + 1) * tests + 5`` seconds, capped below the run's
grading timeout, instead of the scorer's 300-second training cap.
The scorer's child shares this script's stdout, so stdout points at stderr while it runs and carries
only the reward.
"""

import json
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, "/tests")

import livecodebench

ANSWER_PATH = Path("/app/answer.txt")
CONFIG_PATH = Path("/tests/config.json")
TESTS_PATH = Path("/tests/tests.json.zlib")


def extract_code(model_output: str) -> str:
    lines = model_output.split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def reward(answer: str, cases: list[dict], per_test_timeout: int, deadline: float) -> float:
    limits = livecodebench.VerifierLimits(
        max_memory_bytes=livecodebench.DEFAULT_LIMITS.max_memory_bytes, total_timeout_seconds=deadline
    )
    passed = livecodebench.lcb_check_correctness(cases, extract_code(answer), timeout=per_test_timeout, limits=limits)
    return 1.0 if passed else 0.0


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    cases = json.loads(zlib.decompress(TESTS_PATH.read_bytes()))
    answer = ANSWER_PATH.read_text()
    stdout = os.dup(1)
    os.dup2(2, 1)
    score = reward(answer, cases, config["per_test_timeout"], config["deadline"])
    sys.stdout.flush()
    os.dup2(stdout, 1)
    print(score)


if __name__ == "__main__":
    main()
