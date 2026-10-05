# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation code and task-content identities."""

from verifyit.spec import Mode

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import SCHEMA_VERSION, TaskSpec
from taskcompendium.pipeline.models import DatasetRecipe

NORMALIZATION_STAGE_REVISION = "4"
VERIFICATION_STAGE_REVISION = "3"
REVIEW_STAGE_REVISION = "2"


def recipe_code_identity(recipe: DatasetRecipe) -> dict[str, str]:
    """Declare which code revisions can change this recipe's audit."""
    return {
        "normalization_stage": NORMALIZATION_STAGE_REVISION,
        "task_schema": SCHEMA_VERSION,
        "family": recipe.pipeline.normalize.__module__,
        "family_revision": recipe.version,
        "verification_stage": VERIFICATION_STAGE_REVISION,
        "review_stage": REVIEW_STAGE_REVISION,
    }


def semantic_digest(task: TaskSpec, include_reference: bool) -> str:
    """Hash public task semantics, optionally including its private reference."""
    content = task.model_dump(mode="json", exclude={"id", "source"})
    # Oracle scripts are executable witnesses, not task semantics.
    content["resources"]["oracle"] = []
    if include_reference and task.verifier.kind in (Mode.MATH, Mode.MCQ):
        # These graders read only their parameters; derivations and provenance are audit evidence.
        content["resources"]["verifier"] = []
    if not include_reference:
        content.pop("verifier")
        content["resources"]["verifier"] = []
    return canonical_sha256(content)


def deduplication_key(task: TaskSpec) -> str:
    """Opaque evaluator inputs and preference candidates define distinct task records."""
    # Different opaque contracts do not establish conflicting answer keys. Their
    # quality review checks reference agreement; exact copies still deduplicate.
    return semantic_digest(
        task,
        include_reference=task.verifier.kind == Mode.SCRIPT,
    )
