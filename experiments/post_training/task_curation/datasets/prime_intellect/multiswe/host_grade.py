# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright Prime Intellect. Licensed under Apache-2.0.
# Source: research-environments b41d8759070715dc102000e06fba0d73ebce9955.

"""Run the pinned source report parser on logs copied from the actor's machine."""

import argparse
import json
from pathlib import Path

from .data import restore_row


def source_test_output(test_output: Path, read_exit_code: int, stdout: str, stderr: str) -> str:
    if read_exit_code == 0:
        output = test_output.read_bytes().decode("utf-8")
        if output:
            return output
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


def main() -> None:
    # The original benchmark dependency is installed only in the private grader image.
    from .score import calculate_reward  # noqa: PLC0415 - private grader dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--read-exit-code", type=Path, required=True)
    parser.add_argument("--harness-stdout", type=Path, required=True)
    parser.add_argument("--harness-stderr", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    args = parser.parse_args()
    row = restore_row(json.loads(args.row.read_text()))
    output = source_test_output(
        args.test_output,
        int(args.read_exit_code.read_text()),
        args.harness_stdout.read_bytes().decode("utf-8"),
        args.harness_stderr.read_bytes().decode("utf-8"),
    )
    reward = calculate_reward(output, row)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(str(reward) + "\n")


if __name__ == "__main__":
    main()
