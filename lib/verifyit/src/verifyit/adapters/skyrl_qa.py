# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""QA client composition after the source's normalization and token extraction."""

from collections.abc import Sequence, Set

from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.grade import Aggregation, InvalidTask, Reward, aggregate_rewards
from verifyit.modes.grade_exact import grade_collection_f1


def grade_qa_exact(candidate: str, references: Sequence[str]) -> Reward:
    """Compare source-normalized alternatives without adding normalization."""
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)) or not references:
        raise InvalidTask("QA requires nonempty normalized references")
    if any(not isinstance(reference, str) for reference in references):
        raise InvalidTask("QA normalized references must be strings")
    if not isinstance(candidate, str):
        raise InvalidTask("QA requires a string candidate")
    results = [grade_literal_candidate(reference, candidate) for reference in references]
    return aggregate_rewards(results, expected_total=len(references), policy=Aggregation.MAX)


def grade_qa_token_sets(candidate: Set[str], reference: Set[str]) -> Reward:
    """Pass source-prepared token sets to core overlap grading."""
    if not isinstance(reference, Set):
        raise InvalidTask("QA F1 requires token sets")
    if not isinstance(candidate, Set):
        grade_collection_f1(reference, [], multiplicity="set", empty_reference="zero", round_digits=None)
        raise InvalidTask("QA F1 requires token sets")
    return grade_collection_f1(reference, candidate, multiplicity="set", empty_reference="zero", round_digits=None)
