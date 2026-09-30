# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade CSV text against required column names."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

from pydantic import model_validator
from tasktrove_verify.grade import Status
from tasktrove_verify.modes.grade_csv import grade as grade_csv
from tasktrove_verify.spec import CsvColumnsSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import Submission, TextSubmission


class CsvColumnsVerifier(Verifier):
    """Require valid CSV with a header and at least one data row."""

    required: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        if not self.required and not self.any_of:
            raise ValueError("CSV column verifier requires names")
        CsvColumnsSpec(required=self.required, any_of=self.any_of)
        return self

    async def grade(
        self, submission: Submission, *, specification: VerifierSpec, attempt: GradingAttempt
    ) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("CSV column verifier requires a text submission")
        contract = CsvColumnsSpec(required=self.required, any_of=self.any_of)
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            output = workspace / "answer.txt"
            output.write_text(submission.value)
            result = grade_csv(replace(contract, output=str(output)), workspace, workspace)
        if result.status is not Status.SCORED:
            return GradeResult(Outcome.INFRA_ERROR, None, "CSV source grader did not score the submission")
        return GradeResult(Outcome.GRADED, result.reward)


def csv_columns_answer(required: tuple[str, ...], any_of: tuple[str, ...]) -> VerifierSpec:
    """Construct a generic CSV column verifier."""
    verifier = CsvColumnsVerifier(required=required, any_of=any_of)
    return VerifierSpec(kind=VerifierKind.CSV_COLUMNS, parameters_json=verifier.model_dump_json())
