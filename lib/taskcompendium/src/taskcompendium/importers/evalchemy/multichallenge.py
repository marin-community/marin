# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MultiChallenge conversations replayed as chat history, with the authors' YES/NO judge.

The solver writes the final assistant turn of a fixed user/assistant history. The judge sees only
that reply and the row's target question, as in the reference evaluator.
"""

from verifyit.spec import EmptyOutputPolicy, JudgeCompletionPolicy, JudgeSpec

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

# JUDGE_PROMPT from github.com/ekwinox117/multi-challenge@5ccefcca6a39020d66c1383c4e6a809cb07afa33
# src/evaluator.py. The reference reads the verdict from structured output; the last sentence is
# added so the verdict can be read from the final line instead.
MULTICHALLENGE_JUDGE_TEMPLATE = (
    "You are tasked with evaluating a model response to see if it meets a specific criteria.\n"
    "The criteria will always be YES/NO evaluation.\n"
    "\n"
    "The model response is as follows:\n"
    "<MODEL_RESPONSE>\n"
    "{candidate}\n"
    "</MODEL_RESPONSE>\n"
    "\n"
    "The criteria that the model response must meet is as follows. Be VERY STRICT!:\n"
    "<CRITERIA>\n"
    "{reference}\n"
    "</CRITERIA>\n"
    "\n"
    'Print your reasoning followed by your verdict, either "YES" or "NO".'
    " Write the verdict alone on the final line."
)


def multichallenge_spec(target_question: str, pass_criteria: str) -> JudgeSpec:
    """Pass when the judge's verdict on ``target_question`` equals ``pass_criteria``."""
    labels = ("YES", "NO")
    if pass_criteria not in labels:
        raise ValueError(f"MultiChallenge pass criteria must be YES or NO: {pass_criteria!r}")
    return JudgeSpec(
        references=(target_question,),
        # Use the judge deployment selected by the evaluation launcher.
        model="",
        rubric="labels",
        prompt_template=MULTICHALLENGE_JUDGE_TEMPLATE,
        exact_gate=False,
        empty_output=EmptyOutputPolicy.GRADE,
        label_scores={label: float(label == pass_criteria) for label in labels},
        label_metrics={label: {"accuracy": float(label == pass_criteria)} for label in labels},
        label_scan="lines",
        label_case="upper",
        max_completion_tokens=4096,
        completion_policy=JudgeCompletionPolicy.CONTENT,
        request_timeout=60,
        transport_attempts=3,
        transport_budget=180,
    )


def convert_multichallenge(row: RawRow, _context: ConversionContext) -> TaskSpec:
    conversation = row.data["conversation"]
    events = tuple(
        TextMessage(role=role, content=content)
        for role, content in zip(conversation["role"], conversation["content"], strict=True)
    )
    if events[-1].role != "user":
        raise ValueError(f"MultiChallenge conversation must end on a user turn: {row.data['question_id']}")
    task = conversation_task(
        row,
        events=events,
        package=verifyit_package(multichallenge_spec(row.data["target_question"], row.data["pass_criteria"])),
        evidence={key: row.data[key] for key in ("question_id", "axis", "target_question", "pass_criteria")},
    )
    return task.model_copy(update={"tags": ("eval", "source:scaleai", "benchmark:MultiChallenge")})


def multichallenge_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer, grouped by the source's evaluation axis."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"axis": source_evidence(task)["axis"]}}
    )
