# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.sft_new_mix.tool_overlay import equivalent_tokenizer_json, export_key, select_records


def test_one_replacement_per_original_row_with_explicit_deterministic_priority():
    historical = {
        "source": "science",
        "source_shard": "frozen-accepted-science",
        "source_row": 7,
        "status": "verified",
        "tool_calls_modified": True,
        "input_ids_sha256": "original",
        "archive_sha256": "old-archive",
        "record_prefix": "records/7/",
        "diff": {"modified": True, "changes": [{"kind": "renamed"}], "before_sha256": "python", "after_sha256": "bash"},
        "export_manifest": {"files": {"input_ids.i32": "historical-export"}},
    }
    deterministic = {**historical, "export_manifest": {"files": {"input_ids.i32": "state-preserving-export"}}}
    outside = {**historical, "source_shard": "superseded-old-science"}
    withheld = {**historical, "source_row": 8, "record_prefix": "records/8/"}
    selected, audit = select_records(
        [
            (b"\n".join(json.dumps(row).encode() for row in (historical, outside, withheld)), 1),
            (json.dumps(deterministic).encode(), 2),
        ],
        {("science", "frozen-accepted-science")},
        {export_key(withheld)},
    )
    assert len(selected) == 1
    assert selected[0]["source_row"] == 7
    assert selected[0]["export_manifest"]["files"]["input_ids.i32"] == "state-preserving-export"
    assert audit["counts"]["not_in_frozen_candidate_inputs"] == 1
    assert audit["counts"]["explicitly_withheld_or_superseded_export"] == 1
    assert audit["superseded"][0]["selected_priority"] == 2


def test_export_tokenizer_byte_pin_can_differ_only_when_complete_json_is_equivalent(tmp_path):
    config = {"version": "1.0", "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1}}}
    target = tmp_path / "target.json"
    exported = tmp_path / "export.json"
    target.write_text(json.dumps(config, separators=(",", ":")))
    exported.write_text(json.dumps(config, indent=2, sort_keys=True))
    target_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    export_sha256 = hashlib.sha256(exported.read_bytes()).hexdigest()
    equivalent_tokenizer_json(str(target), target_sha256, str(exported), export_sha256)
    config["model"]["vocab"] = {"a": 1, "b": 0}
    exported.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not JSON-equivalent"):
        equivalent_tokenizer_json(
            str(target), target_sha256, str(exported), hashlib.sha256(exported.read_bytes()).hexdigest()
        )
