# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PubMedQA and MedMCQA prompts from lm-eval-harness (MIT license).

PubMedQA follows lm-eval v0.4.12's `pubmedqa` task. MedMCQA follows the `medmcqa` and
`medmcqa_cot` tasks in `lm_eval/tasks/medmcqa/utils_medmcqa.py` at
EleutherAI/lm-evaluation-harness@d6de81643928d653435c431bae19945d41d32520.
"""

from verifyit.spec import McqExtraction, McqSpec

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.grader import verifyit_package
from taskcompendium.importers.fewshot import (
    ChoiceDocument,
    ChoiceTemplate,
    Document,
    FewshotPolicy,
    FewshotSampling,
)
from taskcompendium.models import GraderMachine, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

PUBMEDQA_CHOICES = ("yes", "no", "maybe")


def pubmedqa_document(doc: Document) -> ChoiceDocument:
    """Render a `pubmed_qa_labeled_fold0_source` row; abstract sections are newline-joined."""
    abstract = "\n".join(doc["CONTEXTS"])
    prompt = f"Abstract: {abstract}\nQuestion: {doc['QUESTION']}\nAnswer:"
    return ChoiceDocument(prompt, PUBMEDQA_CHOICES, PUBMEDQA_CHOICES.index(doc["final_decision"]))


def pubmedqa_template() -> ChoiceTemplate:
    return ChoiceTemplate("pubmedqa", pubmedqa_document, FewshotPolicy(0, FewshotSampling.FIRST))


MEDMCQA_LETTERS = ("A", "B", "C", "D")
MEDMCQA_COT_INSTRUCTION = (
    "First, think step-by-step. End your response with exactly this format:\n"
    "Final Answer: X\n(where X is A, B, C, or D)."
)


def medmcqa_question(doc: Document) -> str:
    """Render the question and lettered options shared by both lm-eval MedMCQA tasks."""
    options = (doc["opa"], doc["opb"], doc["opc"], doc["opd"])
    return (
        "Question: "
        + doc["question"]
        + "\nChoices:\n"
        + "".join(f"{letter}. {option}\n" for letter, option in zip(MEDMCQA_LETTERS, options, strict=True))
    )


def medmcqa_document(doc: Document) -> ChoiceDocument:
    """Score the answer letters after `Answer:`; `cop` is the 0-based gold option."""
    return ChoiceDocument(medmcqa_question(doc) + "Answer:", MEDMCQA_LETTERS, doc["cop"])


def medmcqa_template() -> ChoiceTemplate:
    return ChoiceTemplate("medmcqa", medmcqa_document, FewshotPolicy(0, FewshotSampling.FIRST))


def convert_medmcqa_cot(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=medmcqa_question(row.data) + MEDMCQA_COT_INSTRUCTION),),
        package=verifyit_package(
            McqSpec(MEDMCQA_LETTERS[row.data["cop"]], options=4, extraction=McqExtraction.MEDMCQA_COT_REFERENCE),
            placement=GraderMachine(environment=required_grader_environment(context)),
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:lm-eval", "benchmark:MedMCQA")})


def medmcqa_cot_item(task: TaskSpec) -> EvaluationItem:
    item = chat_item(task, response_selection="final_only")
    return item.model_copy(update={"groups": {"subject": source_evidence(task)["subject_name"]}})
