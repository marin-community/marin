# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a continuation of a pinned prefix using the first completed outer region.

Callers with generation stop metadata normalize terminal markers before calling
this text-only mode. Trailing output affects diagnostics, not historical reward.
"""

from dataclasses import asdict
from pathlib import Path

from verifyit.closure_scoring import ClosureFormat, closure_state, score_closure_region
from verifyit.grade import InvalidTask, Reward, empty_output_policy, read_output, scored
from verifyit.spec import StructureClosureSpec
from verifyit.structure_closure import PrefixStatus


def validate_structure_closure(spec: StructureClosureSpec) -> None:
    if not isinstance(spec.format, ClosureFormat):
        raise InvalidTask("structure_closure requires a supported format")
    if not isinstance(spec.prefix, str) or not isinstance(spec.expected, str):
        raise InvalidTask("structure_closure prefix and expected must be text")
    if not isinstance(spec.terminal_markers, tuple) or any(
        not isinstance(marker, str) or not marker for marker in spec.terminal_markers
    ):
        raise InvalidTask("structure_closure terminal_markers must be nonempty strings")
    if not spec.expected.startswith(spec.prefix):
        raise InvalidTask("structure_closure reference must extend the supplied prefix")
    if closure_state(spec.prefix, spec.format).status != PrefixStatus.EXTENDABLE:
        raise InvalidTask("structure_closure prefix must be an unfinished valid structure")
    try:
        reference = score_closure_region(spec.expected, spec.expected, spec.format)
    except ValueError as error:
        raise InvalidTask(f"Invalid structure_closure reference: {error}") from error
    if not reference.whole_response_success:
        raise InvalidTask("structure_closure reference must contain one complete structure")
    empty_output_policy(spec)


def grade_structure_closure_candidate(spec: StructureClosureSpec, candidate: str) -> Reward:
    """Grade only the generated continuation; the fixed prefix is task-owned."""
    validate_structure_closure(spec)
    score = score_closure_region(spec.prefix + candidate, spec.expected, spec.format)
    return scored(float(score.success), **asdict(score))


def grade(spec: StructureClosureSpec, tests_dir: Path, workspace: Path) -> Reward:
    validate_structure_closure(spec)
    continuation = read_output(spec, workspace)
    if continuation is None:
        return scored(0.0, reason="no_output")
    return grade_structure_closure_candidate(spec, continuation)
