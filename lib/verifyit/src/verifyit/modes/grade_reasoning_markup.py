# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Credit a reply whose reasoning markup is well formed.

The candidate is the reply text with its reasoning markers retained. A reply with no markers is
well formed. The verdict names the violation (``unclosed``, ``second_span`` or ``stray_end``) when
there is one. The candidate carries no stop reason, so an unclosed span is a violation here even
when the reply was cut off by a length limit.
"""

from pathlib import Path

from verifyit.grade import Reward, empty_output_policy, read_output, scored
from verifyit.reasoning_markup import reasoning_span_violation
from verifyit.spec import ReasoningMarkupSpec

_COMPLETED_STOP_REASON = "stop"


def validate_reasoning_markup(spec: ReasoningMarkupSpec) -> None:
    empty_output_policy(spec)


def grade_reasoning_markup_candidate(spec: ReasoningMarkupSpec, candidate: str) -> Reward:
    validate_reasoning_markup(spec)
    violation = reasoning_span_violation(candidate, _COMPLETED_STOP_REASON)
    if violation is None:
        return scored(1.0, violation=None)
    return scored(0.0, violation=violation.value)


def grade(spec: ReasoningMarkupSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_reasoning_markup(spec)
    reply = read_output(spec, workspace)
    if reply is None:
        return scored(0.0, reason="no_output")
    return grade_reasoning_markup_candidate(spec, reply)
