# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judges that hosted APIs serve, which evaluations and eval policies share."""

from dataclasses import dataclass
from pathlib import Path

import yaml
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


def _api_judge(entry: dict) -> ApiJudge:
    settings = EvaluationOverrides.model_validate(entry)
    assert settings.judge is not None and settings.judge_endpoint is not None, entry
    return ApiJudge(identity=settings.judge, endpoint=settings.judge_endpoint, api_key=settings.judge_api_key)


_CONFIG = yaml.safe_load((Path(__file__).with_name("configs") / "evaluations.yaml").read_text())
JUDGES: dict[str, ApiJudge] = {name: _api_judge(entry) for name, entry in _CONFIG["judges"].items()}
"""The judges named in configs/evaluations.yaml, whose evaluations reference them by YAML anchor."""
