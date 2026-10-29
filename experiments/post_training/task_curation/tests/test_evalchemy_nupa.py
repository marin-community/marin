# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the immutable panel join independently of source iteration order."""

import hashlib
import json

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.importers.evaluation_nupa import read_nupa_panel
from taskcompendium.pipeline.inputs import ConversionContext

SOURCE = {
    "add_Integer_Integer_Integer": {"1": ["Add: 1 + 2 = 3", "Add: 1 + 2 = 3"]},
    "sub_Integer_Integer_Integer": {"1": ["Subtract: 3 - 1 = 2"]},
}
PANEL = [
    {
        "task_name": "sub_Integer_Integer_Integer",
        "digit": 1,
        "sha256": hashlib.sha256(b"Subtract: 3 - 1 = 2").hexdigest(),
    },
    {"task_name": "add_Integer_Integer_Integer", "digit": 1, "sha256": hashlib.sha256(b"Add: 1 + 2 = 3").hexdigest()},
]


@pytest.fixture
def staged_source(tmp_path):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "test.json").write_text(json.dumps(SOURCE))
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "nupa5k_manifest.jsonl").write_text("".join(json.dumps(identity) + "\n" for identity in PANEL))
    return primary, panel


def test_nupa_panel_resolves_manifest_order_and_rejects_missing_source_identity(staged_source):
    primary, panel = staged_source
    path = primary / "test.json"
    context = ConversionContext({"panel": StoragePath(str(panel))}, None)
    records = list(read_nupa_panel(StoragePath(str(path)), context))
    assert [
        {"task_name": row["task_name"], "digit": row["digit"], "sha256": row["source_sha256"]} for row in records
    ] == PANEL
    data = json.loads(path.read_text())
    data.clear()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        list(read_nupa_panel(StoragePath(str(path)), context))


def test_nupa_panel_order_does_not_depend_on_source_task_order(staged_source):
    primary, panel = staged_source
    path = primary / "test.json"
    context = ConversionContext({"panel": StoragePath(str(panel))}, None)
    before = list(read_nupa_panel(StoragePath(str(path)), context))
    path.write_text(json.dumps(dict(reversed(list(SOURCE.items())))))
    after = list(read_nupa_panel(StoragePath(str(path)), context))
    assert after == before
