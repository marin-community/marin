# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL instruction normalization and canonical constraint contracts."""

import json

from pydantic import ValidationError

from taskcompendium.datasets.direct_contracts import contract_task
from taskcompendium.models import (
    ConversationInput,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)

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
    ("Missing requested documents or inputs are defects; an unbound canonical evaluator alone is not a content defect."),
)


def _normalize(row: RawRow, messages_key: str, constraints_key: str) -> TaskSpec | ImportRejection:
    try:
        messages = tuple(TextMessage.model_validate(message) for message in row.data[messages_key])
        ConversationInput(events=messages)
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
        contract = {
            "constraints": constraints,
            "aggregation": "fraction_satisfied",
            "source_metadata": {
                key: value for key, value in row.data.items() if key not in {messages_key, constraints_key, "path"}
            },
            "source_transform": (
                "infra/rl_data/sources.py:_prepare_nemotron_if"
                if constraints_key == "args"
                else "infra/rl_data/sources.py:_prepare_rlvr_ifeval"
            ),
        }
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="invalid_instruction_contract", detail=str(error)
        )
    return contract_task(
        row,
        messages,
        EVALUATOR,
        contract,
        ("Pinned SkyRL canonical IFEval functions and source-specific argument normalization",),
    )


def normalize_nemotron_if(row: RawRow) -> TaskSpec | ImportRejection:
    return _normalize(row, "input", "args")


def normalize_rlvr_ifeval(row: RawRow) -> TaskSpec | ImportRejection:
    return _normalize(row, "messages", "ground_truth")


NEMOTRON_IF_RUBRIC = ReviewRubric("nemotron_if-answerability", "1", CRITERIA)


def nemotron_if_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_nemotron_if, rubric=NEMOTRON_IF_RUBRIC)


RLVR_IFEVAL_RUBRIC = ReviewRubric("rlvr_ifeval-answerability", "1", CRITERIA)


def rlvr_ifeval_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_rlvr_ifeval, rubric=RLVR_IFEVAL_RUBRIC)
