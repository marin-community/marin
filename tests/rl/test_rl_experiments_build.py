# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.training.training import LevanterCheckpoint
from marinskyrl.recipe_schema import SkyRLRecipe

from experiments.post_training.iceball_micro import iceball_rl_spec


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("trainer.placement.policy_num_nodes", None),
        ("trainer.algorithm.use_kl_loss", None),
        ("generator.backend", None),
        ("generator.run_engines_locally", False),
    ),
)
def test_rl_artifact_rejects_missing_or_conflicting_author_policy(path: str, value: str | bool | None) -> None:
    model = ArtifactStep.adopt("models/build-test", "2026.10.04", "s3://test/model", kind=LevanterCheckpoint)
    data = ArtifactStep.adopt("documents/build-test", "2026.10.04", "s3://test/data", kind=Artifact)
    spec = iceball_rl_spec(model, data, version="2026.10.04")
    document = spec.recipe.to_skyrl()
    node = document
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    if value is None:
        del node[parts[-1]]
    else:
        node[parts[-1]] = value
    recipe = SkyRLRecipe.from_document(document)

    with pytest.raises(ValueError, match=path):
        replace(spec, recipe=recipe)
