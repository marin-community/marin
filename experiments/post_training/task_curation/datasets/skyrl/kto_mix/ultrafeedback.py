# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned kto component ultrafeedback source declaration."""

from taskcompendium.datasets import kto_components
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
        name="kto_component_ultrafeedback",
        version="kto_component_ultrafeedback-v1",
        hf_id="trl-lib/kto-mix-14k",
        revision="4470f033f33364e7d064c9f920c3df54d0cce767",
        config="default",
        split="train",
        inputs=kto_components.component_inputs(
            component="argilla/ultrafeedback-binarized-preferences-cleaned",
            kto_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
            parent_revision="f8869fc91bde5c71a104667292addcbbfd15985d",
        ),
        policy=kto_components.component_policy(),
        intended_use=IntendedUse.TRAIN,
    )
