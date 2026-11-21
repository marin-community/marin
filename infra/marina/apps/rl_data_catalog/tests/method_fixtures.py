# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completed small SFT receipt for method projection behavior checks."""

from datetime import UTC, datetime

from finestore.sft import (
    REQUIRED_SFT_EVAL_FAMILIES,
    DatasetVariantInput,
    SftEvidenceReference,
    SftPreparationReceipt,
    SftStage,
)

from infra.marina.apps.rl_data_catalog.server.methods import DatasetVariant, VariantIdentity


def sft_variant(source_id: str) -> DatasetVariant:
    identity = DatasetVariantInput(
        source_id=source_id, variant_id="sft-v1", repository="fixture/child", revision="a" * 40, data_sha256="b" * 64
    )
    reference = SftEvidenceReference(uri="s3://fixture/preparation/receipt.json", sha256="c" * 64)
    receipt = SftPreparationReceipt(
        input=identity,
        registry_source="fixture",
        implementation_sha256="d" * 64,
        stages={stage: reference for stage in SftStage},
        eval_manifests={family: reference for family in REQUIRED_SFT_EVAL_FAMILIES},
        eval_policy_sha256="e" * 64,
        output_manifest=reference,
        tokenizer=reference,
        chat_template_sha256="f" * 64,
        input_rows=2,
        training_rows=1,
        filtered_rows=1,
        chat_rows=2,
        normalization_counters={"conversion": {}, "normalization": {}},
        stage_files={stage: reference for stage in [SftStage.CHAT, SftStage.NORMALIZE, SftStage.DECONTAM]},
        training_tokens=3,
        supervised_tokens=1,
        training_sha256="0" * 64,
        completed_at=datetime(2026, 11, 20, tzinfo=UTC),
    )
    return DatasetVariant(
        input=VariantIdentity.model_validate(identity.model_dump()),
        tags=frozenset({"sft"}),
        sft_receipt=receipt,
        sft_receipt_reference=reference,
    )
