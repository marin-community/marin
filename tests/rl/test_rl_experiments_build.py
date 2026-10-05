# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.skyrl_recipe import SkyRLRecipe
from marin.training.training import LevanterCheckpoint

from experiments.post_training.iceball_micro import iceball_rl_spec


@pytest.mark.parametrize(
    "path",
    (
        "trainer.placement.policy_num_nodes",
        "trainer.algorithm.use_kl_loss",
        "generator.backend",
    ),
)
def test_rl_artifact_requires_explicit_author_policy_before_materialization(path: str) -> None:
    model = ArtifactStep.adopt("models/build-test", "2026.10.04", "s3://test/model", kind=LevanterCheckpoint)
    data = ArtifactStep.adopt("documents/build-test", "2026.10.04", "s3://test/data", kind=Artifact)
    spec = iceball_rl_spec(model, data, version="2026.10.04")
    document = spec.recipe.to_skyrl()
    node = document
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    recipe = SkyRLRecipe.from_document(document)

    with pytest.raises(ValueError, match=path):
        replace(spec, recipe=recipe)
