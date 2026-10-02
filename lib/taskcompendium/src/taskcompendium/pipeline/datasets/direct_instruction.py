# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve direct SkyRL instruction contracts without substituting TaskTrove checkers."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)
from taskcompendium.verifiers.source_contract import SourceContractVerifier

VERIFIER_REVISION = "bb6494e678ffaa1e6bd3967e221d1a67b038757e"
EVALUATOR = "MarinSkyRL:skyrl_gym.envs.ifeval.utils.compute_score"

CRITERIA = (
    (
        "Identify every public content request and requirement across the complete conversation; do not "
        "discard earlier requests."
    ),
    (
        "Check the private canonical constraint configuration against the public wording and identify "
        "missing, additional, or contradictory constraints."
    ),
    (
        "Formal constraint rewards do not establish factual correctness or useful content; judge the "
        "underlying request separately."
    ),
    (
        "The canonical source rewards the fraction of constraints satisfied; TaskTrove similarly named "
        "checks can differ in counting and punctuation semantics."
    ),
    (
        "Missing requested documents or inputs are defects; an unbound canonical evaluator alone is not a "
        "content defect."
    ),
)


def normalize(row: RawRow, messages_key: str, constraints_key: str) -> TaskSpec | ImportRejection:
    try:
        messages = tuple(TextMessage.model_validate(message) for message in row.data[messages_key])
        context = ConversationInput(events=messages)
        constraints = row.data[constraints_key]
        if constraints_key == "ground_truth":
            constraints = json.loads(constraints)
            if not isinstance(constraints, dict) or not isinstance(constraints.get("func_name"), str):
                raise ValueError("The source ground truth must name its canonical constraint function")
        else:
            names = constraints["instruction_id_list"]
            arguments = constraints["instruction_kwargs"]
            if not names or len(names) != len(arguments):
                raise ValueError("Instruction identifiers and arguments must be nonempty and aligned")
        verifier = SourceContractVerifier(
            evaluator=EVALUATOR,
            source_revision=VERIFIER_REVISION,
            contract={
                "constraints": constraints,
                "aggregation": "fraction_satisfied",
                "source_metadata": {
                    key: value
                    for key, value in row.data.items()
                    if key not in {messages_key, constraints_key, "path", "source_byte_offset"}
                    and not key.startswith("sample_")
                },
                "source_transform": (
                    "infra/rl_data/sources.py:_prepare_nemotron_if"
                    if constraints_key == "args"
                    else "infra/rl_data/sources.py:_prepare_rlvr_ifeval"
                ),
            },
            runtime_requirements=("Pinned SkyRL canonical IFEval functions and source-specific argument normalization",),
        )
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_instruction_contract", detail=str(error))
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.SOURCE_CONTRACT, parameters_json=verifier.model_dump_json()),
    )


def recipe(
    name: str,
    snapshot: Path,
    *,
    dataset: str,
    revision: str,
    split: str,
    rubric: ReviewRubric,
    normalize_row: Callable[[RawRow], TaskSpec | ImportRejection],
) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=SnapshotSource(dataset, revision, "default", split, str(snapshot)),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
    )
