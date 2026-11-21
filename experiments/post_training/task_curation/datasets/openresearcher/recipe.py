# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen OpenResearcher intake; final-pin admission is bound separately."""

from taskcompendium.openresearcher import OpenResearcherState

from experiments.post_training.task_curation.datasets.openresearcher.prepare import (
    PreparationConfig,
    ResearchDataset,
    source,
)
from experiments.post_training.task_curation.source import RlDataSource


def sources() -> list[RlDataSource[ResearchDataset]]:
    return [
        source(
            PreparationConfig(
                stage_receipt="s3://marin-us-east-02a/marin/rl-data/openresearcher/staging/2026.11.15.1/inputs-receipt.json",
                stage_receipt_sha256="de0f1c7d7ec381ea365812694f537a7e7005fa106751802ca12a92c7b6d25280",
                question_receipt="s3://marin-us-east-02a/marin/rl-data/openresearcher/questions/2026.11.15.1/question-receipt.json",
                question_receipt_sha256="3f7efecdaae9bfedaa62475c78c27845f0130bf48fd5d2ebdb7583e7400fcb88",
                grounding_receipt="s3://marin-us-east-02a/marin/rl-data/openresearcher/grounding/2026.11.15.12/grounding-receipt.json",
                grounding_receipt_sha256="eb506c7c74320279937c82a541204cea07f335d2dbb1c78032b6efdef0e3c943",
                heldout_hashes="s3://marin-us-east-02a/marin/rl-data/inputs/openresearcher/2026.11.15.1/heldout-question-hashes-v1.json",
                heldout_hashes_sha256="c8a45fd2dbbfd844001559acb14059f375d0e86fd6abcf98d69d5a63b2c9ff18",
                actor_image="ghcr.io/marin-community/rl-data-openresearcher@sha256:ff448d3a683247068454eaa67ee7e25e75b5357ee74e80696b5df18d15e016fa",
                grader_image="ghcr.io/marin-community/rl-data-openresearcher@sha256:6ccb79c34d3c1da34c6bd947b9a0206e852b2e00d79b671efade9af1f853e9cb",
                state=OpenResearcherState(
                    implementation_revision="036cceab31e06844720488fd29e504d479777f05",
                    corpus_revision="8367703babe6e080899a48782f365c284c32b668",
                    index_manifest_sha256="3203f0e6387c2e6a3e7509a5d4261cbce2c8e263abb0b72bf2eb580f1929f177",
                    tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
                    top_k=5,
                    snippet_max_tokens=512,
                    include_get_document=True,
                ),
                workdir="/tmp/openresearcher-native-v1",
                output_path="",
            )
        )
    ]
