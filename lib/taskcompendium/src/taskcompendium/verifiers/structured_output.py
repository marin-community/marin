# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared file-backed scoring for structured text submissions."""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from tasktrove_verify.grade import Reward, Status

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.submission import Submission, TextSubmission


def grade_file_output(
    submission: Submission,
    *,
    format_name: str,
    grade_candidate: Callable[[Path, Path], Reward],
) -> GradeResult:
    """Write a text candidate to a temporary output and preserve scorer failures."""
    if not isinstance(submission, TextSubmission):
        raise TypeError(f"{format_name} verifier requires a text submission")
    with TemporaryDirectory() as directory:
        workspace = Path(directory)
        output = workspace / "answer.txt"
        output.write_text(submission.value)
        result = grade_candidate(output, workspace)
    if result.status is not Status.SCORED:
        return GradeResult(Outcome.INFRA_ERROR, None, f"{format_name} source grader did not score the submission")
    return GradeResult(Outcome.GRADED, result.reward)
