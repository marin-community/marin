# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict

import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from datasets import Dataset
from marin.execution.lazy import StepContext

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl import pool


def test_scheduler_controls_share_fixtures_and_optimizer_semantics():
    sync, sync_eval = async_rl.build_experiment(
        version="2026.09.05.1", cluster="cw-us-east-02a", runner=async_rl.Runner.SYNC, scale=async_rl.Scale.SMOKE
    )
    asynchronous, async_eval = async_rl.build_experiment(
        version="2026.09.05.1", cluster="cw-us-east-02a", runner=async_rl.Runner.ASYNC, scale=async_rl.Scale.SMOKE
    )
    sync_config = sync.build_config(StepContext.for_fingerprint(sync.runtime_args, sync.deps)).request
    async_config = asynchronous.build_config(
        StepContext.for_fingerprint(asynchronous.runtime_args, asynchronous.deps)
    ).request
    sync_yaml = yaml.safe_load(sync_config.config_yaml)
    async_yaml = yaml.safe_load(async_config.config_yaml)
    assert sync_yaml.pop("entrypoint") != async_yaml.pop("entrypoint")
    assert sync_yaml == async_yaml
    assert asdict(sync_config.model) == asdict(async_config.model)
    assert sync_config.train_data == async_config.train_data
    assert sync_config.validation_data == async_config.validation_data
    assert sync_config.topology == async_config.topology
    assert sync_config.run_id != async_config.run_id
    assert sync_eval.name != async_eval.name


@pytest.mark.parametrize("changed", [{"spans": False}, {"staleness": 0}, {"scale": async_rl.Scale.QUALIFICATION}])
def test_diagnostic_changes_cannot_share_a_run_identity(changed):
    kwargs = dict(
        version="2026.09.05.1", cluster="cw-us-east-02a", runner=async_rl.Runner.ASYNC, scale=async_rl.Scale.SMOKE
    )
    baseline, _ = async_rl.build_experiment(**kwargs)
    variant, _ = async_rl.build_experiment(**(kwargs | changed))
    assert baseline.name != variant.name
    assert baseline.fingerprint() != variant.fingerprint()


def test_gsm8k_artifact_preserves_contracts_and_disjoint_ids(tmp_path, monkeypatch):
    def dataset_from_hub(_name, _subset, *, split, revision):
        # Replace the external dataset download; exercise the real curriculum row construction.
        assert revision == async_rl.DATA_REVISION
        count = async_rl.TRAIN_ROWS if split == "train" else async_rl.VALIDATION_ROWS
        return Dataset.from_list(
            [{"question": f"{split} question {i}", "answer": "Work. #### 1,234"} for i in range(count)]
        )

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return {"input_ids": list(range(len(messages[-1]["content"].split())))}

    monkeypatch.setattr(pool, "load_dataset", dataset_from_hub)
    monkeypatch.setattr(pool.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    async_rl.write_gsm8k_subset(async_rl.Gsm8kSubsetConfig(str(tmp_path)))

    manifest = json.loads((tmp_path / "selection.json").read_text())
    assert set(manifest["rows"]["train"]).isdisjoint(manifest["rows"]["test"])
    for filename, split in (("train.parquet", "train"), ("validation.parquet", "test")):
        rows = pq.read_table(tmp_path / filename).to_pylist()
        assert len(rows) == len(manifest["rows"][split])
        for row in rows:
            assert row["env_class"] == "gsm8k"
            assert row["reward_spec"]["ground_truth"] == row["reward_model"]["ground_truth"] == "1234"
            assert row["extra_info"]["split"] == split
            assert "#### <number>" in row["prompt"][-1]["content"]
            assert row["prompt"][0]["role"] == "system"


def test_run_rejects_cross_region_artifact_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.05.1", "--cluster", "cw-rno2a", "--run"])
    assert result.exit_code != 0
    assert "not local to cw-rno2a" in result.output
