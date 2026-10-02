# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the shared multiple-choice contract during task conversion."""

from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import verifier_descriptor
from taskcompendium.models import VerifierSpec


def multiple_choice_answer(expected: str, options: int) -> VerifierSpec:
    return verifier_descriptor(McqSpec(expected=expected.strip().upper(), options=options))
