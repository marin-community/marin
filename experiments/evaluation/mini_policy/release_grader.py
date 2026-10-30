# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade one validation response using the shipped, source-pinned evaluator wheel."""

import asyncio
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import eval as eval_package
from aiohttp import ClientSession
from eval.contracts.eval_spec import EvalSpec, GraderSpec, JudgePlan, ModelRequest, RequestKind, SourceRef
from eval.contracts.failures import MalformedJudgeResponse
from eval.execution.judge import JudgeBackend
from eval.graders.answer_equivalence import JudgeConfig, JudgeProtocol
from eval.graders.specs import GRADERS
from eval.sources.lm_eval import grade_lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from marin.evaluation.metric_selection import declared_sample_metric


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


def native_grade(spec, response):
    # Reconstruct grading from one embedded source document, without downloading a dataset.
    include = Path(eval_package.__file__).parent / "lm_eval_tasks"
    manager = TaskManager(include_path=[str(include)])
    entry = manager.task_index[spec.task]
    config = load_yaml(entry.yaml_path, resolve_func=True, recursive=True)
    with tempfile.TemporaryDirectory(prefix="mini-grade-doc-") as directory:
        document = Path(directory) / "source.jsonl"
        document.write_text(json.dumps(spec.doc) + "\n")
        config.update(
            dataset_path="json",
            dataset_name=None,
            dataset_kwargs={"data_files": {"test": str(document)}},
            training_split=None,
            validation_split=None,
            fewshot_split=None,
            test_split="test",
            num_fewshot=0,
            process_docs=None,
        )
        task = ConfigurableTask(config=config)
        return grade_lm_eval(task, spec, [response] * len(spec.requests))


async def grade_response(config, response):
    spec = source_spec(config["spec"])
    if config["benchmark"] == "truthfulqa":
        return generated_truthful_choice(spec, response)
    if spec.grader.name.startswith("lm_eval:"):
        grade = native_grade(spec, response)
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
            backend = JudgeBackend(judge, session)
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
