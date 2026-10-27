# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Reasoning Gym reply as the seed tasks do: the text after the last Answer: marker, or the whole stripped reply,
scored by the dataset's own scorer against the instance's entry in /tests/config.json."""

import contextlib
import json
import sys
from pathlib import Path

import reasoning_gym

CONFIG = Path("/tests/config.json")
ANSWER = Path("/app/answer.txt")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    reply = ANSWER.read_text(errors="replace")
    _, marker, after = reply.rpartition("Answer:")
    candidate = after.strip() if marker else reply.strip()
    with contextlib.redirect_stdout(sys.stderr):
        reward = float(reasoning_gym.get_score_answer_fn(config["task"])(candidate, config["entry"]))
    print(reward)


if __name__ == "__main__":
    main()
