# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from infra.marina.apps.rl_data_catalog.server.methods import (
    DatasetVariant,
    require_receipt_bytes,
    rl_variant,
    variant_projection,
)
from infra.marina.apps.rl_data_catalog.tests.method_fixtures import sft_variant

DECLARATION = {"id": "Fixture:source", "dataset_id": "fixture/rl", "dataset_revision": "a" * 40}


def test_method_tags_union_variant_receipts_and_leave_parent_unchanged():
    parent = rl_variant(DECLARATION)
    child = sft_variant(DECLARATION["id"])
    projection = variant_projection(DECLARATION, [parent.model_dump(mode="json"), child.model_dump(mode="json")])
    assert projection["method_tags"] == ["rl", "sft"]
    assert parent.tags == frozenset({"rl"})
    assert projection["method_variants"][1]["input"]["variant_id"] == "sft-v1"
    assert variant_projection(DECLARATION, None)["method_tags"] == ["rl"]
    with pytest.raises(ValueError, match="belong to their source"):
        variant_projection({**DECLARATION, "id": "Fixture:other"}, [child.model_dump(mode="json")])


@pytest.mark.parametrize("defect", ["missing_receipt", "wrong_content", "wrong_variant", "wrong_task_specs"])
def test_sft_tag_refuses_missing_or_other_input_preparation(defect):
    raw = sft_variant(DECLARATION["id"]).model_dump(mode="json")
    if defect == "missing_receipt":
        raw["sft_receipt"] = None
    else:
        field = {"wrong_content": "data_sha256", "wrong_variant": "variant_id", "wrong_task_specs": "task_specs_sha256"}[
            defect
        ]
        raw["input"][field] = "9" * 64
    with pytest.raises(ValueError):
        DatasetVariant.model_validate(raw)


def test_receipt_readback_requires_exact_bytes_and_typed_evidence():
    raw = sft_variant(DECLARATION["id"]).model_dump(mode="json")
    receipt = sft_variant(DECLARATION["id"]).sft_receipt.model_dump_json().encode()
    raw["sft_receipt_reference"]["sha256"] = hashlib.sha256(receipt).hexdigest()
    variant = DatasetVariant.model_validate(raw)
    require_receipt_bytes(variant, receipt)
    with pytest.raises(ValueError, match="digest"):
        require_receipt_bytes(variant, receipt + b"changed")
