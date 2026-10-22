# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normal answers to privately labelled unsolvable tasks always score zero."""

from pathlib import Path

from verifyit.grade import Reward, empty_output_policy, scored
from verifyit.spec import UnsolvableSpec


def validate_unsolvable(spec: UnsolvableSpec) -> None:
    empty_output_policy(spec)


def grade_unsolvable_candidate(spec: UnsolvableSpec, _candidate: str) -> Reward:
    validate_unsolvable(spec)
    return scored(0.0)


def grade(spec: UnsolvableSpec, _tests_dir: Path, _workspace: Path) -> Reward:
    validate_unsolvable(spec)
    return scored(0.0)
