# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nonce-sensitive retrieval and final-line graph node metrics."""

from pathlib import Path

from verifyit.adapters.evalchemy_long_context import grade_graphwalks, grade_nonce_similarity
from verifyit.grade import InvalidTask, Reward, empty_output_policy, read_output
from verifyit.spec import GraphWalksSpec, MRCRSpec


def validate_mrcr(spec: MRCRSpec) -> None:
    empty_output_policy(spec)
    if not isinstance(spec.expected, str) or not isinstance(spec.nonce, str) or not spec.nonce:
        raise InvalidTask("MRCR requires text and a nonempty nonce")


def validate_graphwalks(spec: GraphWalksSpec) -> None:
    empty_output_policy(spec)
    if not isinstance(spec.expected, tuple) or any(not isinstance(node, str) for node in spec.expected):
        raise InvalidTask("GraphWalks requires reference node strings")


def grade_mrcr_candidate(spec: MRCRSpec, candidate: str) -> Reward:
    validate_mrcr(spec)
    return grade_nonce_similarity(spec.expected, spec.nonce, candidate)


def grade_graphwalks_candidate(spec: GraphWalksSpec, candidate: str) -> Reward:
    validate_graphwalks(spec)
    return grade_graphwalks(spec.expected, candidate)


def grade(spec: MRCRSpec | GraphWalksSpec, _tests_dir: Path, workspace: Path) -> Reward:
    candidate = read_output(spec, workspace) or ""
    if isinstance(spec, MRCRSpec):
        return grade_mrcr_candidate(spec, candidate)
    return grade_graphwalks_candidate(spec, candidate)
