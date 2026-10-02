# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-chat delivery for VerifyIT's XML-name and CSV-column contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import model_validator
from verifyit.modes.grade_csv import grade as grade_csv
from verifyit.modes.grade_xml import grade as grade_xml
from verifyit.spec import CsvColumnsSpec, XmlElementsSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.submission import extract_answer


class NamedFieldsVerifier(Verifier):
    mode: Literal["xml-elements", "csv-columns"]
    required: tuple[str, ...]
    any_of: tuple[str, ...]

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        if not self.required and not self.any_of:
            raise ValueError("A named-fields contract needs required or any_of names")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            text = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        # The upstream graders own parsing and name matching. Only delivery changes.
        with TemporaryDirectory(prefix="structured-fields-") as directory:
            workspace = Path(directory)
            (workspace / "answer.txt").write_text(text)
            if self.mode == "xml-elements":
                result = grade_xml(XmlElementsSpec(self.required, self.any_of, "/app/answer.txt"), workspace, workspace)
            else:
                result = grade_csv(CsvColumnsSpec(self.required, self.any_of, "/app/answer.txt"), workspace, workspace)
        return GradeResult(Outcome.GRADED, result.reward, json.dumps(result.detail))
