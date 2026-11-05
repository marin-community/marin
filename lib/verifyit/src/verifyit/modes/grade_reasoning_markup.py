# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Credit a reply whose reasoning markup is well formed.

The candidate is the reply text with its reasoning markers retained, and the reason generation
stopped. A reply with no markers is well formed. The prompt is taken to open no span, so a leading
closing marker is stray. A span left open by a length stop is truncation, not a violation. The file
grader reads no stop reason, so there an unclosed span is always a violation. The verdict names the
violation (``unclosed``, ``second_span`` or ``stray_end``) when there is one.
"""

from pathlib import Path

from verifyit.grade import Reward, empty_output_policy, read_output, scored
from verifyit.reasoning_markup import COMPLETED_STOP_REASON, ReasoningReply, reasoning_span_violation
from verifyit.spec import ReasoningMarkupSpec


def validate_reasoning_markup(spec: ReasoningMarkupSpec) -> None:
    empty_output_policy(spec)


def grade_reasoning_markup_candidate(spec: ReasoningMarkupSpec, candidate: ReasoningReply) -> Reward:
    validate_reasoning_markup(spec)
    violation = reasoning_span_violation(candidate.text, candidate.stop_reason, prompt_opens_span=False)
    if violation is None:
        return scored(1.0, violation=None)
    return scored(0.0, violation=violation.value)


def grade(spec: ReasoningMarkupSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_reasoning_markup(spec)
    reply = read_output(spec, workspace)
    if reply is None:
        return scored(0.0, reason="no_output")
    return grade_reasoning_markup_candidate(spec, ReasoningReply(reply, COMPLETED_STOP_REASON))
