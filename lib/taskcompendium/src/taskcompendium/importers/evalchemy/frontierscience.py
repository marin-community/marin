# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI FrontierScience source rows and Appendix B grading."""

import hashlib
from pathlib import Path

from verifyit.grade import VERDICT_JSON
from verifyit.spec import EmptyOutputPolicy, JudgeSpec

from taskcompendium.convert.answers import answer_task, source_evidence
from taskcompendium.convert.script_grader import GRADE_ARGV, grade_script
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.grader import GraderPackage
from taskcompendium.models import (
    FileReward,
    GraderMachine,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

# Appendix B, transcribed by bytedance/axion1_0 at 96ce1830d87399f20577790f405fe2407da34156.
_ASSETS = Path(__file__).with_name("assets")
_RESEARCH_FILES = grade_script(Path(__file__).with_name("scorers") / "frontierscience.py")
RESEARCH_SCORER_REVISION = hashlib.sha256(resource_bytes(_RESEARCH_FILES[0])).hexdigest()[:16]
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
    environment = environment.model_copy(update={"capabilities": (*environment.capabilities, "network")})
    grader = ScriptGrader(
        argv=GRADE_ARGV,
        placement=GraderMachine(environment=environment),
        judge=spec,
        empty_output=EmptyOutputPolicy.GRADE,
        timeout=60 + max(spec.transport_budget, spec.request_timeout),
        reward=FileReward(files=(RewardFile(path=f"/app/{VERDICT_JSON}", format=RewardFileFormat.VERDICT_JSON),)),
    )
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=problem),),
        package=GraderPackage(grader, _RESEARCH_FILES),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "benchmark:frontierscience-research")})


def frontierscience_item(task: TaskSpec) -> EvaluationItem:
    """Select the visible response and record its source subject for reporting."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"subject": source_evidence(task)["subject"]}}
    )
