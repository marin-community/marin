# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select canonical completion rewards while preserving the retained scorer package."""

from dataclasses import replace

from verifyit.grade import Reward, Status

from .harness_generation import grade_harness_generation_candidate as retained_harness_grade


def grade_harness_generation_candidate(
    profile: str, references: tuple[tuple[str, ...], ...], response: str, question: str
) -> Reward:
    result = retained_harness_grade(profile, references, response, question)
    if profile == "gsm8k" and result.status is Status.SCORED:
        return replace(result, reward=result.detail["metrics"]["exact_match,flexible-extract"])
    return result
