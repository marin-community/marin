# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor answer-file clients for existing exact and MCQ primitives."""

import argparse
import errno
import json
import re
from pathlib import Path

from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.file_ops.read import read_regular_bytes
from verifyit.grade import DEFAULT_LOGS_DIR, InvalidTask, Reward, infra_error, invalid_task, scored, write_reward
from verifyit.json_objects import unique_object
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.spec import McqSpec

SAT_MARKER = re.compile(r"\[(SAT|UNSAT)\]")
ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
ASCII_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ASCII_SPACE = " \t\r\n\v\f"


def grade_answer(mode: str, expected: str, candidate: str | None) -> Reward:
    """Apply the pinned source extractor, then call exact or MCQ candidate grading."""
    if mode == "aime":
        if not expected.isdecimal():
            raise InvalidTask("AIME expected answer must be a decimal integer")
        return grade_literal_candidate(expected, candidate.strip() if candidate is not None else "")
    if mode == "gaia":
        reference = expected.replace("\n", "").translate(ASCII_LOWER).strip(ASCII_SPACE)
        if not reference:
            raise InvalidTask("GAIA expected answer must not be empty")
        answer = (candidate or "").replace("\n", "").translate(ASCII_LOWER).strip(ASCII_SPACE)
        return grade_literal_candidate(reference, answer)
    if mode == "satbench":
        if not isinstance(expected, str) or expected not in {"SAT", "UNSAT"}:
            raise InvalidTask("SATBench expected answer must be SAT or UNSAT")
        markers = SAT_MARKER.findall(candidate or "")
        return grade_literal_candidate(expected, markers[-1] if markers else "")
    if mode == "gpqa-diamond":
        answer = (candidate or "").translate(ASCII_UPPER)
        answer = "".join(character for character in answer if character not in ASCII_SPACE)
        return grade_mcq_candidate(McqSpec(expected=expected, options=4), answer)
    raise InvalidTask(f"unknown Harbor answer route {mode!r}")


def grade_files(mode: str, expected_path: Path, candidate_path: Path) -> Reward:
    """Read protected task reference and candidate output with distinct failure statuses."""
    try:
        expected = read_regular_bytes(expected_path).decode()
    except FileNotFoundError as error:
        return invalid_task(f"missing expected answer: {error.filename}")
    except (UnicodeError, ValueError) as error:
        return invalid_task(f"invalid expected answer: {error}")
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            return invalid_task(f"invalid expected answer: {error}")
        return infra_error(f"cannot read expected answer: {error}")
    if mode == "satbench":
        try:
            data = json.loads(expected, object_pairs_hook=unique_object)
            label = data["expected_answer"]
        except (ValueError, TypeError, KeyError) as error:
            return invalid_task(f"invalid SATBench ground truth: {error}")
        if not isinstance(label, str):
            return invalid_task("SATBench expected answer must be a string")
        expected = label
    try:
        grade_answer(mode, expected, None)
    except InvalidTask as error:
        return invalid_task(str(error))
    try:
        candidate = read_regular_bytes(candidate_path).decode()
    except FileNotFoundError:
        candidate = None
    except UnicodeError as error:
        return infra_error(f"cannot read candidate answer: {error}")
    except ValueError as error:
        return scored(0.0, reason="invalid_answer_file", error=str(error))
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.ENXIO):
            return scored(0.0, reason="invalid_answer_file", error=str(error))
        return infra_error(f"cannot read candidate answer: {error}")
    try:
        if candidate is None and mode == "gaia":
            return scored(0.0, reason="missing_answer_file")
        return grade_answer(mode, expected, candidate)
    except InvalidTask as error:
        return invalid_task(str(error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("aime", "gaia", "satbench", "gpqa-diamond"))
    parser.add_argument("expected", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--logs-dir", type=Path, default=Path(DEFAULT_LOGS_DIR))
    args = parser.parse_args(argv)
    try:
        reward = grade_files(args.mode, args.expected, args.candidate)
    except Exception as error:
        reward = infra_error(f"{type(error).__name__}: {error}")
    write_reward(args.logs_dir, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
