# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade XML text against required element and attribute names."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

from pydantic import model_validator
from tasktrove_verify.grade import Status
from tasktrove_verify.modes.grade_xml import grade as grade_xml
from tasktrove_verify.spec import XmlElementsSpec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import Submission, TextSubmission


class XmlElementsVerifier(Verifier):
    """Require a well-formed XML answer carrying the configured names."""

    required: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        if not self.required and not self.any_of:
            raise ValueError("XML element verifier requires names")
        XmlElementsSpec(required=self.required, any_of=self.any_of)
        return self

    async def grade(
        self, submission: Submission, *, specification: VerifierSpec, attempt: GradingAttempt
    ) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("XML element verifier requires a text submission")
        contract = XmlElementsSpec(required=self.required, any_of=self.any_of)
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            output = workspace / "answer.txt"
            output.write_text(submission.value)
            result = grade_xml(replace(contract, output=str(output)), workspace, workspace)
        if result.status is not Status.SCORED:
            return GradeResult(Outcome.INFRA_ERROR, None, "XML source grader did not score the submission")
        return GradeResult(Outcome.GRADED, result.reward)


def xml_elements_answer(required: tuple[str, ...], any_of: tuple[str, ...]) -> VerifierSpec:
    """Construct a generic XML-name verifier."""
    verifier = XmlElementsVerifier(required=required, any_of=any_of)
    return VerifierSpec(kind=VerifierKind.XML_ELEMENTS, parameters_json=verifier.model_dump_json())
