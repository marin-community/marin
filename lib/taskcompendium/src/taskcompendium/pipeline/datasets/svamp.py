# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic word problems with separate context, question, and answer fields."""

from math import isfinite

from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, ImportRejection, IntendedUse, RawRow, ReviewRubric


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not isinstance(body, str) or not body.strip() or not isinstance(question, str) or not question.strip():
        return ImportRejection(reason="missing_prompt", detail="Body and Question must be nonempty strings")
    if not isinstance(answer, str):
        return ImportRejection(reason="invalid_reference", detail="Answer must be a numeric string")
    try:
        expected = float(answer.strip())
    except ValueError:
        return ImportRejection(reason="invalid_reference", detail="Answer is not numeric")
    if not isfinite(expected):
        return ImportRejection(reason="invalid_reference", detail="Answer is not finite")
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=f"{body.strip()} {question.strip()}"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(expected, tolerance_abs=0.0, tolerance_rel=0.0),
        source=row.source,
    )


recipe = DatasetRecipe(
    name="svamp",
    version="svamp-v1",
    source=HFSource("ChilleD/SVAMP", "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef", "default", "train"),
    normalize=normalize,
    rubric=ReviewRubric(
        id="arithmetic-word-problems",
        version="1",
        criteria=(
            "Identify the quantities and the operation the question actually requests. Check units and directionality.",
            "Ignore irrelevant quantities. Distracting numbers alone do not make a task ambiguous.",
            "Flag contradictions or missing quantities that prevent a unique numeric answer.",
        ),
    ),
    intended_use=IntendedUse.TRAIN,
)
