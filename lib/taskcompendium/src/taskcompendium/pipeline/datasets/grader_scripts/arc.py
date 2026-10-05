# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse ARC grids and score the final boxed grid or whole answer."""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from verifyit.grade import Reward, scored


def validated_grid(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("ARC grids must be nonempty and rectangular")
    if any(isinstance(cell, bool) or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("ARC cells must be digits 0-9")
    return grid


def parse_grid(text: str) -> list[list[int]] | None:
    """Preserve the source parser's row boundaries and accepted serializations."""
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if (
        isinstance(value, list)
        and value
        and all(isinstance(row, list) for row in value)
        and all(isinstance(cell, int) for row in value for cell in row)
    ):
        return value
    rows = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip().strip("[](){}'\"")
        if not line:
            continue
        try:
            if re.search(r"[,\s]", line):
                tokens = [token.strip("[](){}'\"") for token in re.split(r"[,\s]+", line)]
                row = [int(token) for token in tokens if token]
            elif re.fullmatch(r"\d+", line):
                row = [int(cell) for cell in line]
            else:
                row = [int(line)]
        except ValueError:
            continue
        rows.append(row)
    return rows or None


def last_boxed_grid(text: str) -> str | None:
    """Keep the source's permissive extraction of an unterminated final box."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return text[brace + 1 :]


def grade_arc_grid(text: str, expected_output: list[list[int]]) -> Reward:
    boxed = last_boxed_grid(text)
    candidates = (boxed, text) if boxed is not None else (text,)
    grid = next((parsed for candidate in candidates if (parsed := parse_grid(candidate)) is not None), None)
    return scored(float(grid == expected_output))


def main() -> None:
    tests = Path(os.environ["VERIFYIT_TESTS_DIR"])
    workspace = Path(os.environ["VERIFYIT_WORKSPACE"])
    config = json.loads((tests / "config.json").read_text())
    answer = (workspace / "answer.txt").read_text()
    verdict = grade_arc_grid(answer, config["expected_output"])
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "verdict.json").write_text(json.dumps(asdict(verdict)))


if __name__ == "__main__":
    main()
