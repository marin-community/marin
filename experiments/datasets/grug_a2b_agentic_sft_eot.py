# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Corrected consolidated fixed-EOT cache used by the August 2026 OpenCode SFT run."""

from levanter.data.text.formats import LossWeightTransform, PrebuiltLmDatasetFormat
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main
from marin.processing.tokenize.tokenize import TokenizedCache

GRUG_A2B_AGENTIC_SFT_FORMAT = PrebuiltLmDatasetFormat(
    input_ids_key="input_ids",
    loss_weights_key="assistant_mask",
    loss_weight_transform=LossWeightTransform.SHIFT_LEFT,
)
_CACHE_SOURCE = (
    "s3://marin-us-east-02a/marin/tokenized/grug-a2b-agentic-sft-eot/2026.08.05"
)
_RENDERED_SOURCE = (
    "s3://marin-us-east-02a/marin/users/held/datasets/grug-67b-a2b-agentic-sft-eot-20260805/"
    "processed/harbor-sft-eot"
)


def grug_a2b_agentic_sft_eot_dataset() -> ArtifactStep[TokenizedCache]:
    """Adopt the exact corrected cache used by the 1,888-step OpenCode run."""
    return ArtifactStep.adopt(
        name="tokenized/grug-a2b-agentic-sft-eot",
        version="2026.08.05",
        source=_CACHE_SOURCE,
        kind=TokenizedCache,
    )


if __name__ == "__main__":
    dataset_main({"grug-a2b-agentic-sft-eot": grug_a2b_agentic_sft_eot_dataset()})
