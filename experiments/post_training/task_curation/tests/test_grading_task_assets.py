# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.grading_task_assets import swe_grading_asset_manifest
from experiments.post_training.task_curation.source import GradingDatasetFile, SweGradingAssets
from experiments.post_training.task_curation.tests.conversion import tasktrove_row


@pytest.mark.parametrize("change", ["metadata", "comments", "unselected_verifier", "selected_verifier", "judge"])
def test_grading_identity_changes_only_for_selected_verifier_inputs(tmp_path, monkeypatch, change):
    keys = [
        {"trajectory_id": name, "step": 1, "turn": 0, "depth": 1, "instance_id": name, "agent_cls": "swe"}
        for name in ("selected", "other")
    ]
    blend = tmp_path / "blend.jsonl"
    blend.write_text(
        "".join(json.dumps({"dataset": "pivot", "trajectory_id": key["trajectory_id"], "info": {}, "metadata": key}) + "\n" for key in keys)
    )
    membership = tmp_path / "membership.parquet"
    pq.write_table(pa.table({"instance_id": ["selected"]}), membership)
    before = []
    after = []
    for key in keys:
        files = {
            "instruction.md": b"Perform the task.",
            "metadata.json": json.dumps(key).encode(),
            "task.toml": b'[metadata]\nnote = "discovery"\n[verifier]\ntimeout_sec = 60\n',
            "tests/verifier.py": b"def score():\n    return 1\n",
            "tests/judge.toml": b'model = "judge-a"\n',
        }
        before.append(tasktrove_row(files, path=key["instance_id"]))
        if change == "metadata":
            files["metadata.json"] = json.dumps({**key, "discovery_note": "updated"}).encode()
            files["task.toml"] = files["task.toml"].replace(b"discovery", b"new description")
        elif change == "comments":
            files["tests/verifier.py"] = b'# New documentation\ndef score():\n    """An explanation."""\n    return 1\n'
        elif (change == "selected_verifier" and key["instance_id"] == "selected") or (
            change == "unselected_verifier" and key["instance_id"] == "other"
        ):
            files["tests/verifier.py"] = b"def score():\n    return 0\n"
        elif change == "judge" and key["instance_id"] == "selected":
            files["tests/judge.toml"] = b'model = "judge-b"\n'
        after.append(tasktrove_row(files, path=key["instance_id"]))
    for version, rows in (("before", before), ("after", after)):
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"{version}.parquet")

    def download(*, filename, revision, **_kwargs):
        return str(tmp_path / (f"{revision}.parquet" if filename == "proxy.parquet" else filename))

    monkeypatch.setattr("experiments.post_training.task_curation.grading_task_assets.hf_hub_download", download)
    selection = SweGradingAssets(
        blend=GradingDatasetFile(str(tmp_path), "before", blend.name),
        proxies=GradingDatasetFile(str(tmp_path), "before", "proxy.parquet"),
        membership=GradingDatasetFile(str(tmp_path), "before", membership.name),
        component="pivot",
        partition="swe_gym",
    )
    old = swe_grading_asset_manifest(selection)
    new = swe_grading_asset_manifest(replace(selection, proxies=replace(selection.proxies, revision="after")))
    assert (old != new) == (change in {"selected_verifier", "judge"})
    assert len(old["verifiers"]) == len(new["verifiers"]) == 1
