# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from infra.marina.apps.rl_data_catalog.server.catalog_text import (
    BINARY_TEXT_MARKER,
    PREFIX,
    decode_catalog_value,
    display_catalog_value,
    encode_catalog_value,
)


@pytest.mark.parametrize(
    "value", ["res_1vs99.rds\n\x1f\x8b\x08\x00binary", "bad\ud800text", b"\xff\x00\x01", PREFIX + "literal"]
)
def test_binary_text_transport_is_lossless_and_json_safe(value):
    original = {"source_id": "Marin:fixture", "task_preview": value, "normal": "Unicode ✓\ntext"}
    encoded = encode_catalog_value(original)
    transported = json.loads(json.dumps(encoded, ensure_ascii=False, allow_nan=False))
    assert decode_catalog_value(transported) == original
    assert transported["source_id"] == original["source_id"]
    assert transported["normal"] == original["normal"]
    assert display_catalog_value(transported)["task_preview"].startswith(BINARY_TEXT_MARKER)
    assert display_catalog_value(transported["task_preview"][:100]).startswith(BINARY_TEXT_MARKER)
