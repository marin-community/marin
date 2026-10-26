# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-judge verifiers: a model judge applies the author's rubric to a reply beside the reference.

``draft.judge_grader`` packages one as a verifyit ``JudgeSpec`` whose context is ``JUDGE_SPEC_PATH``
(``JudgeRubricSpec``: the rubric and the judge strength) in the task's verifier resources.

Judge strength is fixed to "default" and its model is ``DEFAULT_JUDGE_MODEL``; authors cannot select it.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from taskforge.spec.rubric import Rubric

JUDGE_SPEC_PATH = "llm_judge.json"
"""An LLM-as-judge task's rubric spec in its verifier resources."""
JUDGE_SPEC_FORMAT = "taskforge.llm_judge/1"
DEFAULT_JUDGE_MODEL = "glm-5.3"
"""The judge every LLM-as-judge grader names until judge strength is more than a stub."""
JUDGE_QUESTION = (
    "Apply the rubric to the candidate answer, reading the reference answer (which earned full credit) beside it. "
    "Judge meaning, not wording: accept the listed equivalences; a negated, quoted, hypothetical or retracted "
    "mention does not count as asserting something; keyword lists without the corresponding claim earn nothing."
)


class JudgeRubricSpec(BaseModel):
    """The verifier artifact of an LLM-as-judge task (``JUDGE_SPEC_PATH``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["taskforge.llm_judge/1"] = JUDGE_SPEC_FORMAT
    rubric: Rubric
    judge_strength: Literal["default"] = "default"
