# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade one Micro210 validation response using only this TaskSpec's embedded grader.

The self-contained Micro210 policy version embeds the vendored grading library beside this
driver (``grader_lib/`` in the verifier workspace), so grading never imports an ambient
evalchemy or lm_eval wheel. The flow matches the frozen runtime-v4 driver: parse the
embedded spec, grade the response, resolve the declared metric, and print the normalized
reward for the runtime's stdout collector.
"""

import asyncio
import json
import math
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

GRADER_LIB = str(Path(__file__).resolve().parent / "grader_lib")
if GRADER_LIB not in sys.path:
    sys.path.insert(0, GRADER_LIB)

from aiohttp import ClientSession  # noqa: E402
from eval.contracts.eval_spec import EvalSpec, GraderSpec, JudgePlan, ModelRequest, RequestKind, SourceRef  # noqa: E402
from eval.contracts.failures import MalformedJudgeResponse  # noqa: E402
from eval.graders.answer_equivalence import JudgeConfig, JudgeProtocol  # noqa: E402
from marin.evaluation.metric_selection import declared_sample_metric  # noqa: E402
from micro210_lm_eval_min import native_grade  # noqa: E402
from micro210_registry import GRADERS  # noqa: E402
from micro210_together_judge import TogetherJudgeBackend  # noqa: E402


def source_spec(payload):
    return EvalSpec(
        **{
            **payload,
            "source": SourceRef(**payload["source"]),
            "grader": GraderSpec(**payload["grader"]),
            "requests": tuple(ModelRequest(**{**r, "kind": RequestKind(r["kind"])}) for r in payload["requests"]),
        }
    )


def generated_truthful_choice(spec, response):
    labels = spec.doc["mc2_targets"]["labels"]
    boxed = list(re.finditer(r"\\boxed\{\s*([A-Z])\s*\}", response))
    if boxed:
        answer = boxed[-1].group(1)
    else:
        match = re.fullmatch(
            r"(?:Answer\s*:\s*)?([A-Z])[.)]?", response.strip().splitlines()[-1] if response.strip() else ""
        )
        answer = match.group(1) if match else ""
    index = ord(answer) - ord("A") if answer else -1
    return float(0 <= index < len(labels) and labels[index] == 1)


def native_grade_for(spec, response):
    """Route lm_eval:* graders through the vendored minimal harness."""
    return native_grade(spec, [response] * len(spec.requests))


async def grade_response(config, response):
    spec = source_spec(config["spec"])
    if config["benchmark"] == "truthfulqa":
        return generated_truthful_choice(spec, response)
    if spec.grader.name.startswith("lm_eval:"):
        grade = native_grade_for(spec, response)
    else:
        grade = GRADERS[spec.grader.name].function(spec, [response] * len(spec.requests))
    if isinstance(grade, JudgePlan):
        plan = grade
        judge = JudgeConfig.resolve(
            judge_model=config["judge_model"],
            base_url=os.environ["JUDGE_BASE_URL"],
            api_key=os.environ["JUDGE_API_KEY"],
            protocol=JudgeProtocol.MARIN_POLICY,
        )
        async with ClientSession() as session:
            backend = TogetherJudgeBackend(judge, session, policy=config["judge"])
            for attempt in range(plan.max_attempts):
                judgments = [await backend.execute(request) for request in plan.requests]
                try:
                    grade = GRADERS[plan.grader.name].function(replace(spec, grader=plan.grader), judgments)
                    break
                except MalformedJudgeResponse:
                    if attempt + 1 == plan.max_attempts:
                        raise
    selected = declared_sample_metric(grade.metrics, config.get("metric"))
    if selected is None:
        raise ValueError(f"Grader did not emit the declared metric: {grade.metrics}")
    reward = float(selected[1])
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError(f"Invalid normalized reward: {reward}")
    return reward


def main():
    config = json.loads(Path("/tests/config.json").read_text())
    response = Path("/app/answer.txt").read_text()
    print(asyncio.run(grade_response(config, response)))


if __name__ == "__main__":
    main()
