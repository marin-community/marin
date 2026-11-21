# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move training continuation controls near a verified premature-closure boundary.

For the reference ["first", "second"], a prefix such as ["first", "secon
can be closed into valid JSON by appending "] but would lose the final "d".
The correct continuation is d"]. This variant advances each training prefix to
the latest such point, so a model must preserve the remaining reference content
before closing. Validation data stays unchanged.
"""

from dataclasses import replace

from verifyit.closure_scoring import ContinuationOutcome, continuation_outcome

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CONTINUE_REQUIRED,
    ClosureCase,
    Split,
    closure_format,
    score_case,
    syntax_state,
)


def near_boundary_case(case: ClosureCase) -> ClosureCase:
    """Advance only ordinary training controls, retaining known reference content."""
    if case.split != Split.TRAIN or case.control != CONTINUE_REQUIRED:
        return case
    if not case.expected.startswith(case.prefix):
        raise ValueError(f"Prefix is not part of the reference: {case.case_id}")
    for cut in range(len(case.expected) - 1, len(case.prefix) - 1, -1):
        prefix = case.expected[:cut]
        suffix = syntax_state(prefix, case.family).closing_suffix
        if not suffix:
            continue
        outcome = continuation_outcome(prefix + suffix, case.expected, closure_format(case.family))
        if outcome == ContinuationOutcome.REFERENCE_PREFIX_CLOSED_EARLY:
            result = replace(case, prefix=prefix)
            if not score_case(result, case.expected[cut:]).success:
                raise ValueError(f"Reference continuation does not pass: {case.case_id}")
            return result
    raise ValueError(f"No premature-closure boundary: {case.case_id}")
