# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an AI2 QASPER Verifiable JSON answer after trimming whitespace and folding case."""

import json
from pathlib import Path


def score(path: Path, expected: str) -> int:
    try:
        answer = json.loads(path.read_text())["answer"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return 0
    return int(isinstance(answer, str) and answer.strip().casefold() == expected)


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    print(score(Path(config["answer_path"]), config["expected"]))


if __name__ == "__main__":
    main()
