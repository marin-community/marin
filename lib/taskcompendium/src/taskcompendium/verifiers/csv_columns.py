# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade CSV text against required column names."""

from dataclasses import replace
from pathlib import Path
from typing import Self

from pydantic import model_validator
from tasktrove_verify.modes.grade_csv import grade as grade_csv
from tasktrove_verify.spec import CsvColumnsSpec

from taskcompendium.grading import GradeResult, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission
from taskcompendium.verifiers.structured_output import grade_file_output


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

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        contract = CsvColumnsSpec(required=self.required, any_of=self.any_of)

        def grade_candidate(output: Path, workspace: Path):
            return grade_csv(replace(contract, output=str(output)), workspace, workspace)

        return grade_file_output(submission, format_name="CSV", grade_candidate=grade_candidate)


def csv_columns_answer(required: tuple[str, ...], any_of: tuple[str, ...]) -> VerifierSpec:
    """Construct a generic CSV column verifier."""
    verifier = CsvColumnsVerifier(required=required, any_of=any_of)
    return VerifierSpec(kind=VerifierKind.CSV_COLUMNS, parameters_json=verifier.model_dump_json())
