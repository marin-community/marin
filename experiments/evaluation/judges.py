# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judges that hosted APIs serve."""

from dataclasses import dataclass

from evalrecords.records import EvaluationOverrides, JudgeIdentity


@dataclass(frozen=True, kw_only=True)
class ApiJudge:
    """A judge a hosted API serves: the model that grades, its endpoint, and ordered secret references for its key."""

    identity: JudgeIdentity
    endpoint: str
    api_key: tuple[str, ...]

    def run_settings(self) -> EvaluationOverrides:
        """The run settings that send a selection's judge requests to this judge."""
        return EvaluationOverrides(judge=self.identity, judge_endpoint=self.endpoint, judge_api_key=self.api_key)


def api_judge(entry: dict) -> ApiJudge:
    """Read a judge from the run settings that name it in configs/evaluations.yaml."""
    settings = EvaluationOverrides.model_validate(entry)
    assert settings.judge is not None and settings.judge_endpoint is not None, entry
    return ApiJudge(identity=settings.judge, endpoint=settings.judge_endpoint, api_key=settings.judge_api_key)
