# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen EOT-rendered Grug A2B agentic SFT records from August 2026."""

from fray.types import ANY_REGION, ResourceConfig
from levanter.data.text.formats import LossWeightTransform, PrebuiltLmDatasetFormat
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import dataset_main
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from rigging.filesystem import prefix_join

from experiments.marin_tokenizer import marin_tokenizer

GRUG_A2B_AGENTIC_SFT_FORMAT = PrebuiltLmDatasetFormat(
    input_ids_key="input_ids",
    loss_weights_key="assistant_mask",
    loss_weight_transform=LossWeightTransform.SHIFT_LEFT,
)
_RENDERED_SOURCE = prefix_join(
    "s3://marin-us-east-02a/marin/users/held/datasets/grug-67b-a2b-agentic-sft-eot-20260805",
    "processed/harbor-sft-eot",
)
_TOKENIZE_RESOURCES = ResourceConfig(cpu=2, ram="32g", regions=[ANY_REGION])


def grug_a2b_agentic_sft_eot_dataset() -> ArtifactStep[TokenizedCache]:
    """Build a Levanter cache from the immutable August EOT rendering."""
    rendered = ArtifactStep.adopt(
        name="datasets/grug-a2b-agentic-sft-eot-rendered",
        version="2026.08.05",
        source=_RENDERED_SOURCE,
        kind=Artifact,
    )

    def build_config(ctx: StepContext) -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(rendered), "*/*.parquet")],
            validation_paths=[],
            cache_path=ctx.output_path,
            tokenizer=marin_tokenizer,
            format=GRUG_A2B_AGENTIC_SFT_FORMAT,
            tags=["sft", "agentic", "grug-a2b"],
            max_workers=8,
            worker_resources=_TOKENIZE_RESOURCES,
        )

    return ArtifactStep(
        name="tokenized/grug-a2b-agentic-sft-eot",
        version="2026.08.05",
        artifact_type=TokenizedCache,
        run=tokenize,
        build_config=build_config,
        deps=(rendered,),
    )


if __name__ == "__main__":
    dataset_main({"grug-a2b-agentic-sft-eot": grug_a2b_agentic_sft_eot_dataset()})
