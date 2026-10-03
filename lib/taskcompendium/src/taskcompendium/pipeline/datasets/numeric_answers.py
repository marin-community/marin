# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned numeric-answer sources with their exact numeric verifiers."""

from math import isfinite

from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat, hub_inputs
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, ImportRejection, IntendedUse, RawRow, ReviewRubric

AIME24_DATASET = "HuggingFaceH4/aime_2024"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
SVAMP_DATASET = "ChilleD/SVAMP"
SVAMP_REVISION = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"


def normalize_aime24(row: RawRow) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("problem"), row.data.get("answer")
    if not isinstance(problem, str) or not problem.strip():
        return ImportRejection(reason="missing_prompt", detail="problem must be a nonempty string")
    if not isinstance(answer, str) or not answer.strip().isdigit() or not 0 <= int(answer) <= 999:
        return ImportRejection(reason="invalid_reference", detail="AIME answers must be integers from 0 through 999")
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=problem.strip()),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(float(answer), tolerance_abs=0.0, tolerance_rel=0.0),
        source=row.source,
    )


AIME24_RECIPE = DatasetRecipe(
    name="aime24",
    version="aime24-v1",
    source=HFSource(AIME24_DATASET, AIME24_REVISION, "default", "train"),
    inputs=hub_inputs(
        AIME24_DATASET,
        AIME24_REVISION,
        SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
    ),
    normalize=normalize_aime24,
    rubric=ReviewRubric(
        id="competition-math",
        version="1",
        criteria=(
            "Preserve LaTeX, domains, quantifiers, and geometric assumptions. Do not demand decimal reformulation.",
            "The source expects one integer from 0 through 999; leading zeros do not change the integer.",
            "Check whether the premises specify a unique answer. Difficulty alone is not a quality defect.",
        ),
    ),
    intended_use=IntendedUse.EVAL,
)


def normalize_svamp(row: RawRow) -> TaskSpec | ImportRejection:
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


SVAMP_RECIPE = DatasetRecipe(
    name="svamp",
    version="svamp-v1",
    source=HFSource(SVAMP_DATASET, SVAMP_REVISION, "default", "train"),
    inputs=hub_inputs(
        SVAMP_DATASET,
        SVAMP_REVISION,
        SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
    ),
    normalize=normalize_svamp,
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


RECIPES: dict[str, DatasetRecipe] = {
    "aime24": AIME24_RECIPE,
    "svamp": SVAMP_RECIPE,
}
