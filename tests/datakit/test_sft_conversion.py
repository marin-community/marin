# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.sft import DatasetVariantInput, SftEvidenceReference
from marin.datakit.sft_conversion import ConversionPlan, convert_chat_variant


def test_conversion_creates_child_with_lineage_and_no_sft_tag(tmp_path):
    parent = tmp_path / "parent.parquet"
    pq.write_table(pa.Table.from_pylist([{"prompt": "Question", "answer": "Response"}]), parent)
    before = parent.read_bytes()
    manifest = tmp_path / "parent.json"
    manifest.write_text(
        json.dumps({"files": [{"uri": str(parent), "sha256": hashlib.sha256(before).hexdigest()}], "rows": 1})
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = DatasetVariantInput(
        source_id="Fixture:source", variant_id="rl", repository="fixture/parent", revision="a" * 40, data_sha256=digest
    )
    plan = ConversionPlan(
        parent=identity,
        parent_manifest=SftEvidenceReference(uri=str(manifest), sha256=digest),
        variant_id="chat",
        repository="fixture/child",
        revision="b" * 40,
        prompt_field="prompt",
        response_field="answer",
        implementation_revision="c" * 40,
    )
    result = convert_chat_variant(plan, str(tmp_path / "child"))
    assert parent.read_bytes() == before
    assert result.parent == identity and result.output.variant_id == "chat"
    assert result.output.data_sha256 != identity.data_sha256
    assert "tags" not in result.model_dump()
    assert pq.read_table(result.lineage.uri).to_pylist()[0]["parent_row_sha256"]
    assert len(pq.read_table(result.records.uri).to_pylist()[0]["messages"]) == 2
    with pytest.raises(ValueError, match="new variant"):
        convert_chat_variant(plan.model_copy(update={"variant_id": "rl"}), str(tmp_path / "wrong"))
