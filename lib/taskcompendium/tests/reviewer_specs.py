# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned reviewer fixtures for offline request and pipeline tests."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from verifyit.modes.grade_judge import JudgeConnection

from taskcompendium.pipeline.reviewer_spec import JudgeTier, ReviewerSpec


@dataclass(frozen=True)
class FixtureJudgeEndpoint:
    @property
    def identity(self) -> dict[str, Any]:
        return {"kind": "hosted", "base_url": "http://fixture/v1"}

    def connect(self, api_key: str | None = None):
        return nullcontext(JudgeConnection("http://fixture/v1", api_key or "fixture"))


def reviewer_spec(model: str = "fixture", model_revision: str = "revision", **settings) -> ReviewerSpec:
    return ReviewerSpec(JudgeTier.HEAVY, model, model_revision, FixtureJudgeEndpoint(), **settings)
