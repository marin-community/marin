# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load variant metadata and verify each Datakit preparation receipt at ingestion."""

import json
from functools import cache
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_data_catalog.server.methods import DatasetVariant, require_receipt_bytes, variant_projection


@cache
def registered_variants() -> dict[str, list[dict]]:
    records = json.loads(Path(__file__).with_name("datasets").joinpath("method_variants.json").read_text())
    for source_id, variants in records.items():
        for value in variants:
            variant = DatasetVariant.model_validate(value)
            if variant.input.source_id != source_id:
                raise ValueError("Registered dataset variant belongs to another source")
            if variant.sft_receipt_reference is not None:
                require_receipt_bytes(variant, StoragePath(variant.sft_receipt_reference.uri).read_bytes())
    return records


def source_variants(declaration: dict[str, Any]) -> list[dict]:
    return variant_projection(declaration, registered_variants().get(declaration["id"]))["method_variants"]
