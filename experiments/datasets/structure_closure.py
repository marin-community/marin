# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable native-task cohorts for the structure-closure experiment."""

from enum import StrEnum

from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name


class ClosureCohort(StrEnum):
    COPY_V1 = "copy-v1"
    VALUE_CONTROLS_V5 = "value-controls-v5"
    NATIVE_TOOLS_V6 = "native-tools-v6"


def tool_prompt_probe_dataset() -> ArtifactStep:
    """Adopt the paired declaration/serialization diagnostic, separate from policy scores."""
    return ArtifactStep.adopt(
        user_owned_name("datasets/structure-closure-tool-prompt-probe"),
        "2026.10.13.1",
        "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/tool-prompt-probe-tasks-v1",
        config={"validation_sha256": "6077aff237874efa64d371a81a1128a9cc6399b2506abca4f21f98e3f9472579"},
    )


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
    if cohort == ClosureCohort.NATIVE_TOOLS_V6:
        return ArtifactStep.adopt(
            user_owned_name("datasets/structure-closure-native-tools"),
            "2026.10.13.6",
            "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-native-tool-tasks-v6",
            config={
                "train_sha256": "275bc77b9ebf59a0e279e6721023afd4ca5cf9cc4f30467568913c7d61b7c336",
                "validation_sha256": "f67bf84ad1db454e10fcf2eb98a5caf80d60dba135010030b7c5ddb1195285fa",
            },
        )
    raise ValueError(f"Unknown closure cohort: {cohort}")
