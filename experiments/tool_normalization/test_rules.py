# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.tool_normalization.rules import approved, rule_record


def test_rule_approval_accepts_cleaned_production_mapping_and_rejects_code_changes():
    data = Path(__file__).resolve().parents[1] / "glm_tool_serving/production_data"
    config = json.loads((data / "config.json").read_text())
    config["rule_approval"]["path"] = str(data / config["rule_approval"]["path"])
    original = [
        {"role": "user", "content": "Print hello"},
        {"role": "assistant", "content": '<tool_call>{"name":"bash","arguments":{"command":"echo hello"}}</tool_call>'},
        {"role": "tool", "content": "hello"},
    ]
    record = rule_record(original, [])
    assert approved(record, config)
    record["implementation_sha256"]["native_seed.py"] = "0" * 64
    with pytest.raises(ValueError, match="Rules changed after their sample audit"):
        approved(record, config)
