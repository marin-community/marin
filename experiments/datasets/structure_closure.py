# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable native-task cohorts for the structure-closure experiment."""

from enum import StrEnum

from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name


class ClosureCohort(StrEnum):
    COPY_V1 = "copy-v1"
    VALUE_CONTROLS_V5 = "value-controls-v5"


def structure_closure_dataset(cohort: ClosureCohort) -> ArtifactStep:
    """Adopt a verified native-task cohort with its immutable content hashes."""
    if cohort == ClosureCohort.COPY_V1:
        return ArtifactStep.adopt(
            user_owned_name("datasets/structure-closure"),
            "2026.10.12.1",
            "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-copy-tasks-v1",
            config={
                "train_sha256": "83f5c3007f065ae6d50ec3892e942c644327f204f3333962b5420748478386d9",
                "validation_sha256": "5813e728fab3ffc17fb290c86260ae29719df744acc2db33b5aff86854a3be1c",
            },
        )
    if cohort == ClosureCohort.VALUE_CONTROLS_V5:
        return ArtifactStep.adopt(
            user_owned_name("datasets/structure-closure-value-controls"),
            "2026.10.12.5",
            "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-value-controls-tasks-v5",
            config={
                "train_sha256": "24fb8a46665a42518bfea4bc6041fde9de2604af37f8f6f85b2938555fd7ca63",
                "validation_sha256": "0a6a8599acd9ff0c76d43f37f8ed6613821ce9f068bb4eb988a3c2acfce3259d",
            },
        )
    raise ValueError(f"Unknown closure cohort: {cohort}")
