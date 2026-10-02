# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct SkyRL instruction sources and their canonical constraint contracts."""

import json

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
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat, hub_inputs
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)
from taskcompendium.verifiers.source_contract import SourceContractVerifier

NEMOTRON_IF_SOURCE = HFSource(
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "ab2a40d258a6a4d9d4c277d702aeea445081766c",
    "default",
    "instruction_following",
)
RLVR_IFEVAL_SOURCE = HFSource("allenai/RLVR-IFeval", "47c03c73621c4aab2b824b7818681117d662770e", "default", "train")

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
    ("Missing requested documents or inputs are defects; an unbound canonical evaluator alone is not a content defect."),
)


def _normalize(row: RawRow, messages_key: str, constraints_key: str) -> TaskSpec | ImportRejection:
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
                    key: value for key, value in row.data.items() if key not in {messages_key, constraints_key, "path"}
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


def normalize_nemotron_if(row: RawRow) -> TaskSpec | ImportRejection:
    return _normalize(row, "input", "args")


def normalize_rlvr_ifeval(row: RawRow) -> TaskSpec | ImportRejection:
    return _normalize(row, "messages", "ground_truth")


RECIPES = {
    "nemotron_if": DatasetRecipe(
        name="nemotron_if",
        version="nemotron_if-v1",
        source=NEMOTRON_IF_SOURCE,
        normalize=normalize_nemotron_if,
        rubric=ReviewRubric("nemotron_if-answerability", "1", CRITERIA),
        intended_use=IntendedUse.TRAIN,
        inputs=hub_inputs(
            NEMOTRON_IF_SOURCE.dataset,
            NEMOTRON_IF_SOURCE.revision,
            SourceFiles(("RL/instruction_following/instruction_following.jsonl",), SourceFormat.JSONL),
        ),
    ),
    "rlvr_ifeval": DatasetRecipe(
        name="rlvr_ifeval",
        version="rlvr_ifeval-v1",
        source=RLVR_IFEVAL_SOURCE,
        normalize=normalize_rlvr_ifeval,
        rubric=ReviewRubric("rlvr_ifeval-answerability", "1", CRITERIA),
        intended_use=IntendedUse.TRAIN,
        inputs=hub_inputs(
            RLVR_IFEVAL_SOURCE.dataset,
            RLVR_IFEVAL_SOURCE.revision,
            SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        ),
    ),
}
