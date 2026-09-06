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


def test_snowball_budget_pair_keeps_evaluation_and_training_controls_matched() -> None:
    previews = []
    for response_tokens in (2048, 4096):
        result = CliRunner().invoke(
            async_snowball.main,
            [
                "--version",
                "2026.09.06.3",
                "--scale",
                "qualification",
                "--completion",
                "metrics",
                "--response-tokens",
                str(response_tokens),
                "--eval-response-tokens",
                "4096",
                "--context-tokens",
                "8192",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        previews.append(json.loads(result.output)["request"])
    assert previews[0]["run_id"] != previews[1]["run_id"]
    configs = [yaml.safe_load(preview["config_yaml"]) for preview in previews]
    for config, response_tokens in zip(configs, (2048, 4096), strict=True):
        assert config["context_budget"].pop("max_new_tokens_per_turn") == response_tokens
        assert config["generator"]["eval_sampling_params"]["max_generate_length"] == 4096
        assert config["context_budget"]["request_window_tokens"] == 8192
    assert configs[0] == configs[1]
    assert previews[0]["model"] == previews[1]["model"]
    assert previews[0]["train_data"] == previews[1]["train_data"]
    assert previews[0]["validation_data"] == previews[1]["validation_data"]
    assert previews[0]["topology"] == previews[1]["topology"]


def test_snowball_rejects_evaluation_budget_that_cannot_fit_before_submission(monkeypatch) -> None:
    submitted = []
    monkeypatch.setattr(async_snowball, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        async_snowball.main,
        ["--version", "2026.09.06.3", "--eval-response-tokens", "4096", "--run"],
    )
    assert result.exit_code != 0
    assert "Context budget must fit" in str(result.exception)
    assert submitted == []


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


def test_snowball_cadence_pair_preserves_objective_and_evaluation_contract() -> None:
    requests = []
    for interval in (1, 2):
        result = CliRunner().invoke(
            async_snowball.main,
            [
                "--version",
                "2026.09.06.4",
                "--scale",
                "cadence-gate",
                "--completion",
                "metrics",
                "--weight-sync-interval",
                str(interval),
                "--max-staleness-steps",
                "1",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        requests.append(json.loads(result.output)["request"])
    assert requests[0]["run_id"] != requests[1]["run_id"]
    configs = [yaml.safe_load(request["config_yaml"]) for request in requests]
    for config, interval in zip(configs, (1, 2), strict=True):
        trainer = config["trainer"]
        assert trainer["fully_async"].pop("weight_sync_interval") == interval
        assert trainer["fully_async"]["max_staleness_steps"] == 1
        assert trainer["max_steps"] == trainer["eval_interval"] == 5
        assert trainer["eval_before_train"]
    assert configs[0] == configs[1]


@pytest.mark.parametrize("completion", ["metrics", "checkpoint", "model"])
def test_snowball_sync_control_changes_only_scheduler(completion) -> None:
    requests = []
    for runner in ("sync", "async"):
        result = CliRunner().invoke(
            async_snowball.main,
            [
                "--version",
                "2026.09.06.7",
                "--runner",
                runner,
                "--scale",
                "qualification",
                "--completion",
                completion,
                "--response-tokens",
                "4096",
                "--eval-response-tokens",
                "4096",
                "--context-tokens",
                "8192",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        preview = json.loads(result.output)
        requests.append(preview["training"]["request"] if completion == "model" else preview["request"])
    configs = [yaml.safe_load(request["config_yaml"]) for request in requests]
    assert [config.pop("entrypoint") for config in configs] == ["standard", "fully_async"]
    assert configs[0] == configs[1]
    assert requests[0]["run_id"] != requests[1]["run_id"]
    for field in ("model", "train_data", "validation_data", "topology", "runtime", "seed", "completion_mode"):
        assert requests[0][field] == requests[1][field]


def test_snowball_sync_rejects_unsupported_publication_cadence_before_submission(monkeypatch) -> None:
    submitted = []
    monkeypatch.setattr(async_snowball, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        async_snowball.main,
        ["--version", "2026.09.06.7", "--runner", "sync", "--weight-sync-interval", "2", "--run"],
    )
    assert result.exit_code != 0
    assert "synchronous runner publishes every update" in str(result.exception)
    assert submitted == []


def test_snowball_replica_pair_preserves_independent_ep8_geometry() -> None:
    requests = []
    for replicas in (1, 2):
        result = CliRunner().invoke(
            async_snowball.main,
            [
                "--version",
                "2026.09.06.8",
                "--completion",
                "metrics",
                "--inference-replicas",
                str(replicas),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        request = json.loads(result.output)["request"]
        requests.append(request)
        topology = request["topology"]
        assert topology["num_nodes"] == 4 + replicas
        assert topology["gpus_per_node"] == 8
        assert topology["role_plan"]["policy_num_nodes"] == 4
        assert topology["role_plan"]["num_inference_engines"] == replicas
    configs = [yaml.safe_load(request["config_yaml"]) for request in requests]
    for config, replicas in zip(configs, (1, 2), strict=True):
        generator = config["generator"]
        assert generator.pop("num_inference_engines") == replicas
        assert generator["inference_engine_data_parallel_size"] == 8
        assert generator["inference_engine_expert_parallel_size"] == 8
        assert generator["inference_engine_node_local"]
    assert configs[0] == configs[1]
    assert requests[0]["run_id"] != requests[1]["run_id"]
    for field in ("model", "train_data", "validation_data", "runtime", "seed", "completion_mode"):
        assert requests[0][field] == requests[1][field]


def test_snowball_rejects_cadence_that_cannot_admit_rollouts_before_submission(monkeypatch) -> None:
    submitted = []
    monkeypatch.setattr(async_snowball, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        async_snowball.main,
        ["--version", "2026.09.06.4", "--weight-sync-interval", "2", "--max-staleness-steps", "0", "--run"],
    )
    assert result.exit_code != 0
    assert "at most max_staleness_steps + 1" in str(result.exception)
    assert submitted == []


def qwen_metrics_request(**changes):
    step, evaluation = async_rl.build_experiment(
        **(
            dict(
                version="2026.09.06.10",
                cluster="cw-us-east-02a",
                runner=async_rl.Runner.ASYNC,
                scale=async_rl.Scale.SCREENING,
                completion="metrics",
                kl_loss=False,
            )
            | changes
        )
    )
    assert evaluation is None
    return step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request


def test_qwen_screening_arms_change_only_requested_scheduler_controls():
    arms = [
        {"runner": async_rl.Runner.SYNC},
        {},
        {"weight_sync_interval": 2},
        {"staleness": 3},
    ]
    requests = [qwen_metrics_request(**arm) for arm in arms]
    configs = [yaml.safe_load(request.config_yaml) for request in requests]
    assert len({request.run_id for request in requests}) == 4
    assert [config.pop("entrypoint") for config in configs] == ["standard"] + ["fully_async"] * 3
    for config, cadence, age in zip(configs, (1, 1, 2, 1), (1, 1, 1, 3), strict=True):
        trainer = config["trainer"]
        assert trainer["fully_async"].pop("weight_sync_interval", 1) == cadence
        assert trainer["fully_async"].pop("max_staleness_steps") == age
        assert trainer["max_steps"] == trainer["eval_interval"] == 25
        assert trainer["eval_before_train"]
        assert not trainer["algorithm"]["use_kl_loss"]
        assert not trainer["algorithm"]["use_kl_in_reward"]
        assert trainer["algorithm"]["policy_loss_type"] == "behavior_clip"
        assert not trainer["algorithm"]["use_tis"]
    assert all(config == configs[0] for config in configs)
    for request in requests:
        assert request.completion_mode == "metrics"
        for field in ("model", "train_data", "validation_data", "topology", "runtime", "seed"):
            assert getattr(request, field) == getattr(requests[0], field)


def test_qwen_screening_long_confirmation_keeps_fixture_and_changes_only_schedule():
    short = qwen_metrics_request()
    long = qwen_metrics_request(screening_steps=100)
    assert short.train_data == long.train_data
    assert short.validation_data == long.validation_data
    assert short.run_id != long.run_id
    configs = [yaml.safe_load(request.config_yaml) for request in (short, long)]
    for config, steps in zip(configs, (25, 100), strict=True):
        trainer = config["trainer"]
        assert trainer.pop("max_steps") == steps
        assert trainer.pop("eval_interval") == steps
        assert trainer.pop("ckpt_interval") == steps
        assert trainer["eval_before_train"]
    assert configs[0] == configs[1]


@pytest.mark.parametrize("change", [{"seed": 18}, {"kl_loss": True}, {"correction": async_rl.Correction.REGULAR_TIS}])
def test_qwen_objective_and_seed_changes_have_distinct_identity(change):
    baseline = qwen_metrics_request()
    variant = qwen_metrics_request(**change)
    assert baseline.run_id != variant.run_id
    assert baseline.model == variant.model
    assert baseline.train_data == variant.train_data
    assert baseline.validation_data == variant.validation_data
    configs = [yaml.safe_load(request.config_yaml) for request in (baseline, variant)]
    if "seed" in change:
        assert (baseline.seed, variant.seed) == (17, 18)
    elif "kl_loss" in change:
        assert not configs[0]["trainer"]["algorithm"].pop("use_kl_loss")
        assert not configs[0]["trainer"]["algorithm"].pop("use_kl_in_reward")
        assert configs[1]["trainer"]["algorithm"].pop("use_kl_loss")
    else:
        algorithm = configs[1]["trainer"]["algorithm"]
        assert algorithm.pop("tis_imp_ratio_cap") == 2.0
        assert algorithm.pop("require_rollout_logprobs") is True
        assert algorithm["policy_loss_type"] == "regular"
        assert algorithm["use_tis"]
        algorithm.update(policy_loss_type="behavior_clip", use_tis=False)
        assert configs[1]["generator"]["sampling_params"]["logprobs"] == 0
    assert configs[0] == configs[1]


def test_qwen_replica_count_reserves_complete_inference_nodes():
    requests = [qwen_metrics_request(inference_replicas=replicas) for replicas in (8, 16)]
    configs = [yaml.safe_load(request.config_yaml) for request in requests]
    assert requests[0].run_id != requests[1].run_id
    for request, config, replicas in zip(requests, configs, (8, 16), strict=True):
        assert request.topology.num_nodes * request.topology.gpus_per_node == 8 + replicas
        plan = request.topology.role_plan
        assert plan.num_inference_engines == replicas
        assert plan.policy_num_nodes == 1
        assert plan.policy_num_gpus_per_node == 8
        assert plan.inference_engine_tensor_parallel_size == 1
        assert plan.train_batch_size == plan.policy_mini_batch_size == 64
        assert config["generator"].pop("num_inference_engines") == replicas
    assert configs[0] == configs[1]


def test_qwen_training_budget_does_not_change_explicit_evaluation_budget():
    requests = [
        qwen_metrics_request(response_tokens=tokens, eval_response_tokens=1024, context_tokens=4096)
        for tokens in (1024, 2048)
    ]
    configs = [yaml.safe_load(request.config_yaml) for request in requests]
    assert requests[0].run_id != requests[1].run_id
    for config, tokens in zip(configs, (1024, 2048), strict=True):
        assert config["context_budget"].pop("max_new_tokens_per_turn") == tokens
        assert config["generator"]["eval_sampling_params"]["max_generate_length"] == 1024
    assert configs[0] == configs[1]


@pytest.mark.parametrize(
    "options, message",
    [
        (["--stage", "evaluation"], "Metrics completion requires --stage rl"),
        (["--runner", "sync", "--weight-sync-interval", "2"], "synchronous runner publishes every update"),
        (["--weight-sync-interval", "3", "--staleness", "1"], "at most max_staleness_steps + 1"),
        (["--eval-response-tokens", "2048"], "Context budget must fit"),
        (["--inference-replicas", "4"], "Invalid value"),
        (["--scale", "smoke", "--screening-steps", "100"], "only supported by the screening scale"),
    ],
)
def test_qwen_invalid_screening_configuration_rejected_before_submission(monkeypatch, options, message):
    submitted = []
    monkeypatch.setattr(async_rl, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        async_rl.main,
        ["--version", "2026.09.06.10", "--completion", "metrics", "--stage", "rl", "--run", *options],
    )
    assert result.exit_code != 0
    assert message in result.output + str(result.exception)
    assert submitted == []


def test_qwen_metrics_preview_and_submission_have_no_export_stage(monkeypatch):
    submitted = []
    monkeypatch.setattr(async_rl, "validate_regional_storage", lambda *_args: None)
    monkeypatch.setattr(async_rl, "run", lambda step, **kwargs: submitted.append(step))
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            "2026.09.06.10",
            "--completion",
            "metrics",
            "--stage",
            "rl",
            "--scale",
            "screening",
            "--no-kl-loss",
            "--max-staleness-steps",
            "3",
            "--seed",
            "18",
            "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    preview = json.loads(result.output)
    assert "export" not in preview
    assert preview["request"]["completion_mode"] == "metrics"
    assert preview["request"]["seed"] == 18
    config = yaml.safe_load(preview["request"]["config_yaml"])
    assert config["trainer"]["fully_async"]["max_staleness_steps"] == 3
    assert not config["trainer"]["algorithm"]["use_kl_loss"]
    assert len(submitted) == 1
    root = submitted[0]
    assert root.artifact_type is async_rl.SkyRLTrainingResult
    assert {dep.artifact_type for dep in root.deps} == {async_rl.LevanterCheckpoint, async_rl.Artifact}
