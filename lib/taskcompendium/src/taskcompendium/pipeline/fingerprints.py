# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation code and task-content identities."""

import json

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import ResourceVisibility, TaskSpec, VerifierKind
from taskcompendium.pipeline.models import DatasetRecipe

NORMALIZATION_STAGE_REVISION = "2"
VERIFICATION_STAGE_REVISION = "1"
REVIEW_STAGE_REVISION = "1"


def recipe_code_identity(recipe: DatasetRecipe) -> dict[str, str]:
    """Declare which code revisions can change this recipe's audit."""
    return {
        "normalization_stage": NORMALIZATION_STAGE_REVISION,
        "family": recipe.normalize.__module__,
        "family_revision": recipe.version,
        "verification_stage": VERIFICATION_STAGE_REVISION,
        "review_stage": REVIEW_STAGE_REVISION,
    }


def semantic_digest(task: TaskSpec, include_reference: bool) -> str:
    """Hash public task semantics, optionally including its private reference."""
    content = task.model_dump(mode="json", exclude={"id", "source"})
    # Control scripts are executable witnesses, not task semantics.
    content["resources"] = [
        resource for resource in content["resources"] if resource["visibility"] != ResourceVisibility.CONTROL
    ]
    if include_reference and task.verifier.kind in (VerifierKind.MATH_ANSWER, VerifierKind.MCQ_ANSWER):
        # These graders read only their parameters; derivations and provenance are audit evidence.
        content["resources"] = [
            resource for resource in content["resources"] if resource["visibility"] != ResourceVisibility.VERIFIER
        ]
    if include_reference and task.verifier.kind == VerifierKind.PREFERENCE_EVIDENCE:
        parameters = json.loads(content["verifier"]["parameters_json"])
        parameters.pop("source_metadata")
        content["verifier"]["parameters_json"] = json.dumps(parameters, sort_keys=True)
    if not include_reference:
        content.pop("verifier")
        content["resources"] = [
            resource for resource in content["resources"] if resource["visibility"] == ResourceVisibility.AGENT
        ]
    return canonical_sha256(content)


def deduplication_key(task: TaskSpec) -> str:
    """Opaque evaluator inputs and preference candidates define distinct task records."""
    # Different opaque contracts do not establish conflicting answer keys. Their
    # quality review checks reference agreement; exact copies still deduplicate.
    return semantic_digest(
        task,
        include_reference=task.verifier.kind in {VerifierKind.PREFERENCE_EVIDENCE, VerifierKind.SOURCE_CONTRACT},
    )
