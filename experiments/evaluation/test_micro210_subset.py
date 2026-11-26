# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import Dataset
from marin.execution.lazy import StepContext
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.dataset.tasks import TaskDataset

from experiments.evaluation.micro210 import micro210_step
from experiments.evaluation.micro210_policy import Micro210Model
from experiments.evaluation.micro210_subset import Micro210Subset, prepare_subset, validate_subset
from tests.rl.micro210_fixture import policy_fixture_reference


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    table = pa.table({"benchmark": ["keep", "skip", "keep"], "prompt": ["first", "other", "last"], "seed": [7, 8, 9]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    raw = buffer.getvalue()
    (source / "rollouts.parquet").write_bytes(raw)
    return {
        "prepared_inputs": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "data_relative_path": "rollouts.parquet",
    }


def test_subset_preserves_frozen_rows_and_rejects_modified_selection(tmp_path, inputs):
    subset = Micro210Subset(("keep",))
    output = tmp_path / "selected"
    output.mkdir()
    prepare_subset(**inputs, subset=subset, output=str(output))
    receipt = validate_subset(**inputs, subset=subset, selected_inputs=str(output))
    assert receipt["source_row_indices"] == [0, 2]
    assert pq.read_table(output / "rollouts.parquet").to_pylist() == [
        {"benchmark": "keep", "prompt": "first", "seed": 7},
        {"benchmark": "keep", "prompt": "last", "seed": 9},
    ]
    table = pa.table({"benchmark": ["keep"], "prompt": ["changed"], "seed": [7]})
    pq.write_table(table, output / "rollouts.parquet")
    receipt["selected_sha256"] = hashlib.sha256((output / "rollouts.parquet").read_bytes()).hexdigest()
    (output / "selection.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from the frozen prepared rows"):
        validate_subset(**inputs, subset=subset, selected_inputs=str(output))


def test_subset_rejects_unknown_benchmark_and_changed_source(tmp_path, inputs):
    output = tmp_path / "selected"
    output.mkdir()
    with pytest.raises(ValueError, match=r"Unknown frozen Micro210 benchmarks.*missing"):
        prepare_subset(**inputs, subset=Micro210Subset(("missing",)), output=str(output))
    assert not list(output.iterdir())
    (tmp_path / "prepared" / "rollouts.parquet").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="prepared task bytes changed"):
        prepare_subset(**inputs, subset=Micro210Subset(("keep",)), output=str(output))


def test_recovery_selects_exact_original_rows_and_preserves_uid_mapping(tmp_path, inputs):
    output = tmp_path / "recovery"
    output.mkdir()
    subset = Micro210Subset(row_indices=(2, 1))
    prepare_subset(**inputs, subset=subset, output=str(output))
    receipt = validate_subset(**inputs, subset=subset, selected_inputs=str(output))
    assert receipt["source_row_indices"] == [1, 2]
    assert pq.read_table(output / "rollouts.parquet").to_pylist() == [
        {"benchmark": "skip", "prompt": "other", "seed": 8, "_source_uid": "1"},
        {"benchmark": "keep", "prompt": "last", "seed": 9, "_source_uid": "2"},
    ]
    dataset = TaskDataset.__new__(TaskDataset)
    dataset.dataframe = Dataset.from_parquet(str(output / "rollouts.parquet"))
    assert [dataset.uid(index) for index in range(len(dataset))] == ["1", "2"]
    assert dataset.dataframe["seed"] == [8, 9]
    receipt["source_row_indices"] = [0, 2]
    (output / "selection.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from the frozen prepared rows"):
        validate_subset(**inputs, subset=subset, selected_inputs=str(output))


def test_standard_micro210_graph_binds_subset_without_changing_protocol(tmp_path, monkeypatch, inputs):
    reference = policy_fixture_reference()
    policy = json.loads(StoragePath(reference.uri).read_text())
    policy["prepared_rollouts_sha256"] = inputs["source_sha256"]
    raw = json.dumps(policy).encode()
    path = tmp_path / "policy.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        "experiments.evaluation.micro210_policy.POLICY_REFERENCE",
        replace(reference, uri=str(path), sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)),
    )
    monkeypatch.setattr("experiments.evaluation.micro210.PREPARED_INPUTS", inputs["prepared_inputs"])
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    full = micro210_step(model, name="fixture/full", version="2026.11.20")
    selected = micro210_step(model, name="fixture/selected", version="2026.11.20", subset=Micro210Subset(("keep",)))
    configs = []
    for step in (full, selected):
        context = StepContext.for_run(
            str(tmp_path / step.name), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        config = step.build_config(replace(context, is_fingerprint=True))
        configs.append(yaml.safe_load(config.skyrl.launch_config_yaml)["skyrl"])
        if step is selected:
            assert "fixture/selected/inputs" in config.selection.inputs
            dependency = next(dep for dep in step.deps if dep.name == "fixture/selected/inputs")
            assert dependency.deps[0].name == "evaluation/micro210/prepared"
    assert configs[0]["generator"]["eval_sampling_params"] == configs[1]["generator"]["eval_sampling_params"]
    assert configs[0]["terminal_bench"] == configs[1]["terminal_bench"]
    assert path.read_bytes() == raw


def test_different_evaluations_keep_subset_artifacts_separate(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", policy_fixture_reference())
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", str(tmp_path / "temporary"))
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "fixture/tokenizer", "b" * 40)
    paths = []
    for name, benchmark in (("fixture/a", "swe"), ("fixture/b", "math")):
        step = micro210_step(model, name=name, version="2026.11.20", subset=Micro210Subset((benchmark,)))
        context = StepContext.for_run(
            str(tmp_path / name), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        config = step.build_config(replace(context, is_fingerprint=True))
        paths.append(config.selection.inputs)
    assert paths[0] != paths[1]
