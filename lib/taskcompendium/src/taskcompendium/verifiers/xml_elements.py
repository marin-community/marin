# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade XML text against required element and attribute names."""

from dataclasses import replace
from pathlib import Path
from typing import Self

from pydantic import model_validator
from tasktrove_verify.modes.grade_xml import grade as grade_xml
from tasktrove_verify.spec import XmlElementsSpec

from taskcompendium.grading import GradeResult, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import GradingAttempt, Submission
from taskcompendium.verifiers.structured_output import grade_file_output


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

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        contract = XmlElementsSpec(required=self.required, any_of=self.any_of)

        def grade_candidate(output: Path, workspace: Path):
            return grade_xml(replace(contract, output=str(output)), workspace, workspace)

        return grade_file_output(submission, format_name="XML", grade_candidate=grade_candidate)


def xml_elements_answer(required: tuple[str, ...], any_of: tuple[str, ...]) -> VerifierSpec:
    """Construct a generic XML-name verifier."""
    verifier = XmlElementsVerifier(required=required, any_of=any_of)
    return VerifierSpec(kind=VerifierKind.XML_ELEMENTS, parameters_json=verifier.model_dump_json())
