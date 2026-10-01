# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AIME competition problems, retaining their integer-answer contract."""

from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, ImportRejection, IntendedUse, RawRow, ReviewRubric


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
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


recipe = DatasetRecipe(
    name="aime24",
    version="aime24-v1",
    source=HFSource("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "default", "train"),
    normalize=normalize,
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
