# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade one ARC-AGI-2 output grid with tasktrove_verify's JSON Schema primitive."""

import argparse
import json
from pathlib import Path

from tasktrove_verify.file_ops.read import read_regular_bytes
from tasktrove_verify.grade import DEFAULT_LOGS_DIR, InvalidTask, Reward, infra_error, invalid_task, scored, write_reward
from tasktrove_verify.modes.grade_json_schema import grade_json_schema_candidate


def _is_grid(value: object) -> bool:
    """ARC cells are JSON integers 0-9; Python bool and integral floats are not cells."""
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 30
        and isinstance(value[0], list)
        and 1 <= len(value[0]) <= 30
        and all(
            isinstance(row, list)
            and len(row) == len(value[0])
            and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
            for row in value
        )
    )


def grade_grids(expected: object, candidate: object) -> Reward:
    """Keep the source's per-test-pair binary equality with strict cell types."""
    if not _is_grid(expected):
        raise InvalidTask("ARC-AGI-2 expected grid must be rectangular, 1-30 cells per side, integers 0-9")
    if not _is_grid(candidate):
        return scored(0.0, reason="invalid_candidate_grid")
    return grade_json_schema_candidate({"const": expected}, candidate)


def grade_files(expected_path: Path, candidate_path: Path) -> Reward:
    """Treat a bad protected reference separately from a bad candidate file."""
    try:
        expected = json.loads(read_regular_bytes(expected_path))
    except (FileNotFoundError, UnicodeError, ValueError) as error:
        return invalid_task(f"invalid ARC-AGI-2 expected grid: {error}")
    except OSError as error:
        return infra_error(f"cannot read ARC-AGI-2 expected grid: {error}")
    if not _is_grid(expected):
        return invalid_task("ARC-AGI-2 expected grid must be rectangular, 1-30 cells per side, integers 0-9")
    try:
        candidate = json.loads(read_regular_bytes(candidate_path))
    except FileNotFoundError:
        return scored(0.0, reason="missing_output_file")
    except (UnicodeError, ValueError):
        return scored(0.0, reason="invalid_candidate_json")
    except OSError as error:
        return infra_error(f"cannot read ARC-AGI-2 output: {error}")

    try:
        return grade_grids(expected, candidate)
    except InvalidTask as error:
        return invalid_task(str(error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--logs-dir", type=Path, default=Path(DEFAULT_LOGS_DIR))
    args = parser.parse_args(argv)
    try:
        reward = grade_files(args.expected, args.candidate)
    except Exception as error:
        reward = infra_error(f"{type(error).__name__}: {error}")
    write_reward(args.logs_dir, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
