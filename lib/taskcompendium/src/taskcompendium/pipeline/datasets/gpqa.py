# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA Diamond with reproducibly ordered choices and a private option key."""

import hashlib

from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, ImportRejection, IntendedUse, RawRow, ReviewRubric
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    question = row.data.get("Question")
    choices = [
        row.data.get(key) for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
    ]
    if (
        not isinstance(question, str)
        or not question.strip()
        or not all(isinstance(choice, str) and choice.strip() for choice in choices)
    ):
        return ImportRejection(reason="missing_prompt_or_choices", detail="Question and all four choices are required")
    texts = [str(choice).strip() for choice in choices]
    if len(set(texts)) != len(texts):
        return ImportRejection(reason="duplicate_options", detail="Choice text repeats after boundary trimming")
    # The hash order avoids always exposing the source's correct-first arrangement.
    ordered = sorted(enumerate(texts), key=lambda item: hashlib.sha256(f"{row.id}:{item[0]}".encode()).digest())
    expected = next(chr(65 + index) for index, (original, _) in enumerate(ordered) if original == 0)
    options = "\n".join(f"{chr(65 + index)}. {text}" for index, (_, text) in enumerate(ordered))
    prompt = f"{question.strip()}\n\n{options}\n\nChoose one option letter from A through D."
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer(expected, options=len(texts)),
        source=row.source,
    )


recipe = DatasetRecipe(
    name="gpqa",
    version="gpqa-v1",
    source=HFSource("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "gpqa_diamond", "train"),
    normalize=normalize,
    rubric=ReviewRubric(
        id="science-mcqa",
        version="1",
        criteria=(
            "Assess every displayed option, units, assumptions, and scientific directionality before judging the key.",
            "Specialist background knowledge is allowed; an omitted experiment, figure, or passage is missing context.",
            "Flag ties, approximate synonyms, or an indefensible key. Technical difficulty alone is not a defect.",
        ),
    ),
    intended_use=IntendedUse.EVAL,
)
