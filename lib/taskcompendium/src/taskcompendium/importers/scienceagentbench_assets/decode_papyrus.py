# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode task 22's submission only inside the reference-free replay machine.

Pickle loading executes candidate-controlled code. Candidate input must never
be decoded in a machine containing gold data or judge credentials. Its JSON-lines
output is the only artifact that may enter the grader. Trusted preparation may
use the same normalizer for the pinned gold pickle, without candidate input.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def decode_pickle(source: Path, target: Path) -> None:
    """Preserve the upstream evaluator's ordering, rounding, and string conversion."""
    frame = pd.read_pickle(source)
    normalized = frame.sort_values(by=["Activity_ID"]).reset_index().drop(["index"], axis=1).round(3).astype(str)
    with target.open("w", encoding="utf-8") as output:
        for row in normalized.itertuples(index=False, name=None):
            output.write(json.dumps(list(row), ensure_ascii=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = subprocess.run((sys.executable, str(args.program)), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    decode_pickle(args.source, args.target)


if __name__ == "__main__":
    main()
