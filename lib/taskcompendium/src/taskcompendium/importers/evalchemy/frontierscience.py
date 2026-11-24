# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI FrontierScience source rows and Appendix B grading."""

from pathlib import Path

from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import answer_task, source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

# Appendix B, transcribed by bytedance/axion1_0 at 96ce1830d87399f20577790f405fe2407da34156.
_ASSETS = Path(__file__).with_name("assets")
JUDGE_MODEL = "openai/gpt-oss-120b"
_OLYMPIAD_PROMPT = (_ASSETS / "frontierscience_olympiad_judge.txt").read_text(encoding="utf-8").rstrip("\n")
_RESEARCH_PROMPT = (_ASSETS / "frontierscience_research_judge.txt").read_text(encoding="utf-8").rstrip("\n")


def convert_olympiad(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """Keep the published solver prompt intact and package the reference privately."""
    problem = row.data["problem"]
    spec = JudgeSpec(
        rubric="labels",
        question=problem,
        references=(row.data["answer"],),
        model=JUDGE_MODEL,
        reasoning_effort="high",
        prompt_template=_OLYMPIAD_PROMPT,
        exact_gate=False,
        label_scores={"VERDICT: CORRECT": 1.0, "VERDICT: INCORRECT": 0.0},
        label_scan="lines",
    )
    task = answer_task(row, prompt=problem, spec=spec, evidence=row.data)
    return task.model_copy(update={"tags": ("eval", "benchmark:frontierscience-olympiad")})


def convert_research(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Keep each released row and use its answer field as the private scoring rubric."""
    problem = row.data["problem"]
    spec = JudgeSpec(
        question=problem,
        references=(row.data["answer"],),
        model=JUDGE_MODEL,
        reasoning_effort="high",
        prompt_template=_RESEARCH_PROMPT,
        max_completion_tokens=32768,
        request_timeout=300,
        exact_gate=False,
    )
    environment = required_grader_environment(context)
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=problem),),
        package=source_package(
            "frontierscience_research",
            {},
            judge=spec,
            environment=environment.model_copy(update={"capabilities": (*environment.capabilities, "network")}),
            timeout=60 + max(spec.transport_budget, spec.request_timeout),
        ),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "benchmark:frontierscience-research")})


def frontierscience_item(task: TaskSpec) -> EvaluationItem:
    """Select the visible response and record its source subject for reporting."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"subject": source_evidence(task)["subject"]}}
    )
