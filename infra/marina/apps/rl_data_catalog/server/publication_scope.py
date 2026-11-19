# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-release public-display approvals, independent of training and evaluation use."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

PUBLICATION_SCOPE_VERSION = "2026.11.19.1"
PUBLICATION_RULING = "rhizome://b610073d-d871-4fd6-a41f-c087d2723ec1/d55f01b04a5a71abc068b1bb45ca2c12"


class PublicationScope(StrEnum):
    PUBLIC_DISPLAY = "public_display"


class PublicationScopeDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    dataset_id: str
    dataset_revision: str
    license_id: str
    upstream_name: str
    upstream_url: str
    scope: PublicationScope = PublicationScope.PUBLIC_DISPLAY
    approved_by: str = "user"
    decided_at: str = "2026-11-19T13:30:00-05:00"
    ruling: str = PUBLICATION_RULING


# The user confirmed these existing grants also allow public display. Their
# archived train/eval approvals remain unchanged; bans and private tiers still win.
PUBLICATION_DECISIONS = tuple(
    PublicationScopeDecision.model_validate(row)
    for row in [
        {
            "source_id": "Evalchemy:lambada_openai",
            "dataset_id": "open-athena/evalchemy-lambada-openai-52fc4632-retained-20261113-v1",
            "dataset_revision": "81670cb498bc5118e8ff1b2232b7978bc16da3c1",
            "license_id": "mit",
            "upstream_name": "EleutherAI/lambada_openai",
            "upstream_url": "https://huggingface.co/datasets/EleutherAI/lambada_openai",
        },
        {
            "source_id": "TaskTrove:R2E-Gym__R2E-Gym-V1",
            "dataset_id": "open-athena/task-trove-r2e-gym-r2e-gym-v1-fda7e545-retained-20261114-oracle-v1",
            "dataset_revision": "5197693cd08583c7540ab59d11889962da4f81a3",
            "license_id": "apache-2.0",
            "upstream_name": "R2E-Gym/R2E-Gym-V1",
            "upstream_url": "https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1",
        },
        {
            "source_id": "TaskTrove:SWE-Gym__SWE-Gym",
            "dataset_id": "open-athena/task-trove-swe-gym-swe-gym-34756fc0-retained-20261114-oracle-v1",
            "dataset_revision": "203ad7878edd71b3a836c7cd414be3ff75acb89c",
            "license_id": "mit",
            "upstream_name": "SWE-Gym/SWE-Gym",
            "upstream_url": "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        },
        {
            "source_id": "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__general",
            "dataset_id": "open-athena/task-trove-xiaomimimo-mimo-v2-6-rl-oss-general-17e9cc3c-retained-20261116-v1",
            "dataset_revision": "4f503485b46029df8bd896dafeab1f22f1296b39",
            "license_id": "apache-2.0",
            "upstream_name": "XiaomiMiMo/MiMo-V2.6-RL-oss",
            "upstream_url": "https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
        },
        {
            "source_id": "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__music",
            "dataset_id": "open-athena/task-trove-xiaomimimo-mimo-v2-6-rl-oss-music-f01e9d44-retained-20261116-v2",
            "dataset_revision": "d6269ca12bf83f0c4ebcf1f863c578aca919560c",
            "license_id": "apache-2.0",
            "upstream_name": "XiaomiMiMo/MiMo-V2.6-RL-oss",
            "upstream_url": "https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
        },
        {
            "source_id": "TaskTrove:smartie-calibforge-rebuilt-v1",
            "dataset_id": "open-athena/campaign-release-smartie-calibforge-rebuilt-v1-a179f723-retained-20261116-v1",
            "dataset_revision": "93f73c7fe618dc57a5527116044384515124fa18",
            "license_id": "cc-by-4.0",
            "upstream_name": "ricdomolm/CalibForge-decontaminated",
            "upstream_url": "https://huggingface.co/datasets/ricdomolm/CalibForge-decontaminated",
        },
        {
            "source_id": "TaskTrove:smartie-calibforge-shared-v2",
            "dataset_id": "open-athena/campaign-release-smartie-calibforge-shared-v2-17ae7896-taskcompendium-20261112",
            "dataset_revision": "bcfae8cd9d8a82338122945afdd843d46ca18b7e",
            "license_id": "cc-by-4.0",
            "upstream_name": "ricdomolm/CalibForge-decontaminated",
            "upstream_url": "https://huggingface.co/datasets/ricdomolm/CalibForge-decontaminated",
        },
        {
            "source_id": "TaskTrove:smartie-swesmith-admitted",
            "dataset_id": "open-athena/campaign-release-smartie-swesmith-admitted-41200e0b-taskcompendium-20261112",
            "dataset_revision": "25cdd6a2737be2f500027ecd2c6dbb1fb96ae377",
            "license_id": "mit",
            "upstream_name": "SWE-bench/SWE-smith",
            "upstream_url": "https://huggingface.co/datasets/SWE-bench/SWE-smith",
        },
        {
            "source_id": "TaskTrove:swesmith-verifier-repaired-v5",
            "dataset_id": "open-athena/campaign-release-swesmith-verifier-repaired-v5-0bfcd894-retained-20261116-v1",
            "dataset_revision": "758b28bf4cc0d0e770329f2944c9451fd578566a",
            "license_id": "mit",
            "upstream_name": "SWE-bench/SWE-smith",
            "upstream_url": "https://huggingface.co/datasets/SWE-bench/SWE-smith",
        },
    ]
)


def publication_scope_decision(declaration: dict) -> PublicationScopeDecision | None:
    """Bind an approval only to its named source and exact dataset release."""
    return next(
        (
            row
            for row in PUBLICATION_DECISIONS
            if (row.source_id, row.dataset_id, row.dataset_revision)
            == (declaration["id"], declaration.get("dataset_id"), declaration.get("dataset_revision"))
        ),
        None,
    )
