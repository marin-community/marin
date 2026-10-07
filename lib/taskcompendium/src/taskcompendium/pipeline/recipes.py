# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct source procedures from explicit pinned inputs and grading policy."""

from taskcompendium.pipeline.inputs import RecipeInputs
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, IntendedUse, TaskPolicy


def hf_recipe(
    *,
    name: str,
    version: str,
    hf_id: str,
    revision: str,
    config: str,
    split: str,
    inputs: RecipeInputs,
    policy: TaskPolicy,
    intended_use: IntendedUse = IntendedUse.TRAIN,
) -> DatasetRecipe:
    """Pair a pinned source selection with its concrete grading procedure."""
    return DatasetRecipe(
        name=name,
        version=version,
        source=HFSource(hf_id, revision, config, split),
        policy=policy,
        intended_use=intended_use,
        inputs=inputs,
    )
