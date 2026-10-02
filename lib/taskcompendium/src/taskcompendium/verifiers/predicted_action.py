# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the shared function-call contract during task conversion."""

from tasktrove_verify.spec import FunctionCall as CandidateCall
from tasktrove_verify.spec import PredictedActionSpec

from taskcompendium.grading import verifier_descriptor
from taskcompendium.models import FunctionCall, VerifierSpec


def predicted_action_verifier(expected_calls: tuple[FunctionCall, ...]) -> VerifierSpec:
    return verifier_descriptor(
        PredictedActionSpec(expected_calls=tuple(CandidateCall(call.name, call.arguments) for call in expected_calls))
    )
