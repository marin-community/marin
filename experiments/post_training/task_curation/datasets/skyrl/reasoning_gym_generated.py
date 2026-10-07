# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned reasoning gym generated source declaration."""

from functools import partial

from taskcompendium.datasets.reasoning_gym import generated as reasoning_gym_generated
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.reasoning_gym import binding as reasoning_gym_binding
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline

REASONING_GYM_REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:reasoning_gym",
        runtime_binding=partial(
            bind_private_grader,
            binder=partial(
                reasoning_gym_binding.bind,
                contract=reasoning_gym_binding.ReasoningContract.GENERATED,
                package_path="/opt/reasoning-gym-generated",
            ),
        ),
        name="reasoning_gym_generated",
        version="reasoning-gym-direct-v5",
        inputs=RecipeInputs(
            SourceFiles(
                ("generator.tar.gz",),
                SourceFormat.GENERATED,
                reader=reasoning_gym_generated.GeneratedRows(
                    REASONING_GYM_REVISION,
                    python_hash_seed=0,
                    excluded_generators=(
                        (
                            "composite",
                            "Orchestration constructor requires explicit component DatasetSpec configuration",
                        ),
                    ),
                ),
            ),
            (
                UrlDownload(
                    f"https://api.github.com/repos/open-thought/reasoning-gym/tarball/{REASONING_GYM_REVISION}",
                    "generator.tar.gz",
                ),
            ),
        ),
        intended_use=IntendedUse.TRAIN,
        hf_id="open-thought/reasoning-gym",
        revision=REASONING_GYM_REVISION,
        config="generated",
        split="generated",
        policy=reasoning_gym_generated.policy(),
    )
