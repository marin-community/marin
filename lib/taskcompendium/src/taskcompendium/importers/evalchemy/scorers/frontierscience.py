# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score FrontierScience Research with the Appendix B rubric."""

import re
from decimal import Decimal
from typing import Any

from verifyit.grade import GradingInfraError, Reward, scored
from verifyit.modes.grade_judge import judge_reply
from verifyit.spec import JudgeSpec

VERDICT_PREFIX = "VERDICT:"


def _research_score(points: Decimal, **detail: Any) -> Reward:
    passed = float(points >= 7)
    return scored(passed, **detail, metrics={"pass_rate": passed, "rubric_points": float(points)})


def grade_research(judge: JudgeSpec, response: str) -> Reward:
    """Retain decimal rubric points and pass answers earning at least seven out of ten."""
    if not response.strip():
        return _research_score(Decimal(0), reason="no_output")

    prompt = judge.prompt_template.format(question=judge.question, reference=judge.references[0], candidate=response)
    reply = judge_reply(judge, [{"role": "user", "content": prompt}])
    detail = {**reply.detail, "completion": reply.text}
    lines = reply.text.strip().splitlines()
    verdicts = [line.strip() for line in lines if line.strip().startswith(VERDICT_PREFIX)]
    if len(verdicts) != 1 or verdicts[0] != lines[-1].strip():
        raise GradingInfraError("judge must end with a single Research verdict", **detail)
    match = re.fullmatch(r" ([0-9]+(?:\.[0-9]+)?)", verdicts[0].removeprefix(VERDICT_PREFIX))
    if not match or (points := Decimal(match[1])) > 10:
        raise GradingInfraError("judge must return Research rubric points between 0 and 10", **detail)
    return _research_score(points, **detail, verdict=verdicts[0])
