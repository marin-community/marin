# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry ingestion checks the conversion producer's own frozen lineage."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.sft import DatasetVariantInput, SftEvidenceReference
from marin.datakit.sft_conversion import ConversionPlan, convert_chat_variant

from experiments.post_training.task_curation.method_variants import read_variants
from infra.marina.apps.rl_data_catalog.server.methods import ConversionLineage, DatasetVariant, VariantIdentity


def test_registered_conversion_reads_its_own_parent_and_output_receipt(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"prompt": ["Question"], "response": ["Answer"]}), source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"uri": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}], "rows": 1})
    )
    parent = DatasetVariantInput(
        source_id="Fixture:source", variant_id="rl", repository="fixture/parent", revision="a" * 40,
        data_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    receipt = convert_chat_variant(
        ConversionPlan(parent=parent, parent_manifest=SftEvidenceReference(uri=str(manifest), sha256=parent.data_sha256),
                       variant_id="chat", repository="fixture/child", revision="b" * 40, prompt_field="prompt",
                       response_field="response", implementation_revision="c" * 40),
        str(tmp_path / "child"),
    )
    path = tmp_path / "child" / "conversion-receipt.json"
    child = DatasetVariant(
        input=VariantIdentity.model_validate(receipt.output.model_dump()),
        conversion=ConversionLineage(parent=parent, output=receipt.output,
            receipt=SftEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())),
    )
    registry = tmp_path / "variants.json"
    registry.write_text(json.dumps({parent.source_id: [child.model_dump(mode="json")]}))
    assert read_variants(registry)[parent.source_id][0]["tags"] == []
    changed = child.model_dump(mode="json")
    changed["input"]["data_sha256"] = "f" * 64
    changed["conversion"]["output"]["data_sha256"] = "f" * 64
    registry.write_text(json.dumps({parent.source_id: [changed]}))
    with pytest.raises(ValueError, match="another parent or output"):
        read_variants(registry)
