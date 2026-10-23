# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a reply with the vendored SkyRL IFEval scorer: the fraction of its constraints the reply satisfies.

``constraints`` in ``/tests/config.json`` are the scorer's normalized constraint descriptors.
"""

import json
import sys
from pathlib import Path

# The scorer imports langdetect only for a language constraint and scores a failed import as an
# unmet constraint; importing it here makes a missing package fail the grader instead.
from langdetect import DetectorFactory

sys.path.insert(0, "/tests")

import ifeval_utils


def main() -> None:
    DetectorFactory.seed = 0  # Language detection samples at random; a fixed seed makes a reply's score repeatable.
    config = json.loads(Path("/tests/config.json").read_text())
    answer = Path("/app/answer.txt").read_text()
    print(ifeval_utils.compute_score(answer, config["constraints"])["score"])


if __name__ == "__main__":
    main()
