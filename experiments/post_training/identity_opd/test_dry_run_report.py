# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import zipfile

import numpy as np
import pytest
from safetensors.numpy import save_file

from experiments.post_training.identity_opd import dry_run_report
from experiments.post_training.mismatch_probe.report import load_archive
from experiments.post_training.mismatch_probe.test_report import _archive


def test_logprob_gate_reads_a_probe_archive(tmp_path):
    _archive(tmp_path / "archive")
    gate = dry_run_report.logprob_gate(load_archive(str(tmp_path / "archive")))
    assert gate["samples"] == 4
    assert gate["trainer_vs_generation"]["abs_max"] == pytest.approx(0.1)
    assert gate["passed"]
    assert gate["trainer_abs_max_vs_update_0"] == {
        "update 1": pytest.approx(0.01),
        "update 2": pytest.approx(0.03),
    }
    assert gate["top1_agreement"].startswith("unavailable")


def _hf_dir(path, tensors):
    path.mkdir()
    weight_map = {}
    for shard, names in (("model-00001.safetensors", ["a", "b"]), ("model-00002.safetensors", ["c"])):
        save_file({name: tensors[name] for name in names}, str(path / shard))
        weight_map.update(dict.fromkeys(names, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(path)


def test_tensor_hashes_find_the_one_changed_tensor(tmp_path):
    tensors = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.ones(4, dtype=np.float16),
        "c": np.zeros(3, dtype=np.float32),
    }
    before = dry_run_report.tensor_hashes(_hf_dir(tmp_path / "before", tensors))
    same = dry_run_report.tensor_hashes(_hf_dir(tmp_path / "same", tensors))
    changed = dry_run_report.tensor_hashes(_hf_dir(tmp_path / "changed", {**tensors, "c": np.full(3, 1e-8, np.float32)}))
    assert dry_run_report.compare_hashes(before, same)["all_equal"]
    result = dry_run_report.compare_hashes(before, changed)
    assert result["differing"] == ["c"]
    assert result["dtypes_before"] == {"F16": 1, "F32": 2}
    assert not result["all_equal"]


def _record(route, text, tokens, stop="stop"):
    return {
        "trajectory": {"environment_extras": {"teacher_route": route}},
        "response": {"text": text, "token_ids": tokens, "stop_reason": stop},
    }


def test_family_measures_from_retained_trajectories(tmp_path):
    records = [
        _record("identity_thinking", "I am Marin Snowball, made by Open Athena.", [1, 2, 128009]),
        _record("identity_nonthinking", "I am an assistant.", [1, 128009]),
        _record("tool_thinking", '<tool_call>\n{"name": "f"}\n</tool_call>', [3, 128009]),
        _record("tool_nonthinking", "I cannot help.", [3, 4, 5], stop="length"),
    ]
    step = tmp_path / "schema_v1" / "archives" / "phase=train" / "step=00000001"
    step.mkdir(parents=True)
    with zipfile.ZipFile(step / "a.zip", "w") as archive:
        for index, record in enumerate(records):
            archive.writestr(f"records/{index}.json.gz", gzip.compress(json.dumps(record).encode()))
        archive.writestr("manifest.json", "{}")
    other = tmp_path / "schema_v1" / "archives" / "phase=train" / "step=00000002"
    other.mkdir(parents=True)
    with zipfile.ZipFile(other / "b.zip", "w") as archive:
        archive.writestr("records/x.json.gz", gzip.compress(json.dumps(records[0]).encode()))

    measures = dry_run_report.family_measures(dry_run_report.trajectory_records(str(tmp_path), 1))
    assert measures["identity"]["samples"] == 2
    assert measures["identity"]["says_name"] == measures["identity"]["says_creator"] == 0.5
    assert measures["tool"]["no_tool_call"] == 0.5
    assert measures["tool"]["truncated"] == 0.5
    assert measures["tool"]["ends_with_end_of_turn"] == 0.5
    assert measures["tool"]["by_route"] == {"tool_nonthinking": 1, "tool_thinking": 1}
