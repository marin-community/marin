# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JEEBench's last-box extraction, source choice credit, and Python-float tolerance."""

from pathlib import Path

from verifyit.adapters.evalchemy_amc import last_boxed_answer
from verifyit.adapters.evalchemy_jee import JEEPolicy, capture_jee_input, grade_jee_answer, prepare_jee_input
from verifyit.grade import InvalidTask, Reward, read_output, scored
from verifyit.spec import JeeQuestionType, JeeSpec


def validate_jee(spec: JeeSpec) -> None:
    if not isinstance(spec.question_type, JeeQuestionType):
        raise InvalidTask("JEE requires a declared source question type")
    prepare_jee_input(capture_jee_input(spec.expected, "", spec.question_type), JEEPolicy.SOURCE)


def grade_jee_candidate(spec: JeeSpec, response: str) -> Reward:
    validate_jee(spec)
    extracted = last_boxed_answer(response)
    verdict = grade_jee_answer(spec.expected, extracted, spec.question_type)
    return Reward(verdict.reward, verdict.status, {**verdict.detail, "extracted": extracted})


def grade(spec: JeeSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_jee(spec)
    response = read_output(spec, workspace)
    if response is None:
        return scored(0.0, reason="no_output")
    return grade_jee_candidate(spec, response)
