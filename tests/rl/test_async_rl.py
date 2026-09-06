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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from experiments.post_training import async_rl, async_snowball
from experiments.post_training.curriculum_rl import pool


@pytest.mark.parametrize("scale", list(async_rl.Scale))
def test_scheduler_controls_share_fixtures_and_optimizer_semantics(scale):
    sync, sync_eval = async_rl.build_experiment(
        version="2026.09.05.1", cluster="cw-us-east-02a", runner=async_rl.Runner.SYNC, scale=scale
    )
    asynchronous, async_eval = async_rl.build_experiment(
        version="2026.09.05.1", cluster="cw-us-east-02a", runner=async_rl.Runner.ASYNC, scale=scale
    )
    sync_training = sync.deps[0]
    async_training = asynchronous.deps[0]
    sync_run_config = sync_training.build_config(
        StepContext.for_fingerprint(sync_training.runtime_args, sync_training.deps)
    )
    async_run_config = async_training.build_config(
        StepContext.for_fingerprint(async_training.runtime_args, async_training.deps)
    )
    sync_config = sync_run_config.request
    async_config = async_run_config.request
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
    assert (
        sync.runtime_args["skyrl_execution"].priority == asynchronous.runtime_args["skyrl_execution"].priority == "batch"
    )


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


@pytest.mark.parametrize(
    ("prefix", "cluster"),
    [
        ("s3://marin-us-east-02a/marin", "cw-rno2a"),
        ("s3://marin-us-west-04a/marin", "cw-us-east-02a"),
        ("s3://marin-na/marin", "cw-us-east-02a"),
    ],
)
def test_run_rejects_cross_region_artifact_prefix(monkeypatch, prefix, cluster):
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.05.1", "--cluster", cluster, "--run"])
    assert result.exit_code != 0
    assert f"not local to {cluster}" in result.output


def test_east_run_accepts_configured_coreweave_bucket(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    async_rl.validate_regional_storage(async_rl.marin_prefix(), "cw-us-east-02a")


def test_qwen_cli_dry_run_previews_training_and_export() -> None:
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.05.1", "--dry-run"])

    assert result.exit_code == 0, result.output
    preview = json.loads(result.output)
    assert preview["training"]["request"]["completion_mode"] == "checkpoint"
    assert preview["export"]["source_runtime_commit"] == "<from-checkpoint-artifact>"


def test_snowball_model_run_submits_graph_without_resolving_checkpoint(monkeypatch) -> None:
    submitted = []
    monkeypatch.setattr(async_snowball, "validate_regional_storage", lambda *_args: None)
    monkeypatch.setattr(async_snowball, "run", lambda step, **_kwargs: submitted.append(step))

    result = CliRunner().invoke(
        async_snowball.main,
        ["--version", "2026.09.05.1", "--completion", "model", "--run"],
    )

    assert result.exit_code == 0, result.output
    preview = json.loads(result.output)
    assert preview["training"]["request"]["completion_mode"] == "checkpoint"
    assert preview["export"]["source_runtime_commit"] == "<from-checkpoint-artifact>"
    assert len(submitted) == 1


@pytest.mark.parametrize("prompt_limit", [2, 1024])
def test_snowball_fixture_checks_actual_export_token_lengths(tmp_path, monkeypatch, prompt_limit):
    tokenizer_backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_think|>\n' }}{% endif %}"
    )
    model_path = tmp_path / "model"
    tokenizer.save_pretrained(model_path)
    dataset = Dataset.from_list([{"question": "How much is one plus one?", "answer": "Work. #### 2"}])
    monkeypatch.setattr(pool, "load_dataset", lambda *_args, **_kwargs: dataset)
    output = tmp_path / "data"
    output.mkdir()
    config = async_snowball.SnowballDataConfig(
        str(output), str(model_path), train_rows=1, validation_rows=1, max_prompt_tokens=prompt_limit
    )
    if prompt_limit == 2:
        with pytest.raises(ValueError, match="prompt exceeds"):
            async_snowball.write_snowball_gsm8k(config)
        assert not (output / "train.parquet").exists()
        return

    async_snowball.write_snowball_gsm8k(config)
    manifest = json.loads((output / "selection.json").read_text())
    assert set(manifest["rows"]["train"]).isdisjoint(manifest["rows"]["test"])
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert rows[0]["reward_spec"]["ground_truth"] == "2"
    encoded = tokenizer.apply_chat_template(rows[0]["prompt"], add_generation_prompt=True, return_dict=True)
    assert manifest["prompt_lengths"]["train"]["max"] == len(encoded["input_ids"]) > 2
