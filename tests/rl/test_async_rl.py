# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace

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
from experiments.post_training.math_eval.launcher import BATTERY_PATH, POOL_ARGUMENT


def test_optimizer_precision_cli_keeps_controls_distinct_and_reproducible():
    arguments = [
        "--version",
        "2026.09.06.17",
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "metrics",
        "--screening-steps",
        "5",
        "--eval-interval",
        "5",
    ]
    requests = []
    for precision in async_rl.OptimizerPrecision:
        result = CliRunner().invoke(
            async_rl.main, [*arguments, "--optimizer-precision", precision.value, "--optimizer-state-metrics"]
        )
        assert result.exit_code == 0, result.output
        request = json.loads(result.output)["request"]
        built, _ = async_rl.build_experiment(
            version="2026.09.06.17",
            cluster="cw-us-east-02a",
            runner=async_rl.Runner.ASYNC,
            scale=async_rl.Scale.SCREENING,
            completion="metrics",
            screening_steps=5,
            eval_interval=5,
            optimizer_precision=precision,
            optimizer_state_metrics=True,
        )
        direct = built.build_config(StepContext.for_fingerprint(built.runtime_args, built.deps)).request
        assert request["run_id"] == direct.run_id
        assert request["config_yaml"] == direct.config_yaml
        requests.append(request)
    assert len({request["run_id"] for request in requests}) == len(async_rl.OptimizerPrecision)
    for field in ("model", "train_data", "validation_data", "runtime", "topology", "seed"):
        assert all(request[field] == requests[0][field] for request in requests)
    configs = [yaml.safe_load(request["config_yaml"]) for request in requests]
    for config in configs:
        assert config["trainer"]["optimizer_state_metrics"] is True
        policy = config["trainer"]["policy"]["megatron_config"]
        policy.pop("optimizer_config_kwargs", None)
        policy.pop("ddp_config", None)
        assert config == configs[0]


def test_optimizer_precision_defaults_preserve_existing_recipe_and_require_observation():
    base = dict(spans=True, staleness=1)
    native = async_rl.training_config(async_rl.Runner.ASYNC, async_rl.Scale.SCREENING, **base)
    explicit = async_rl.training_config(
        async_rl.Runner.ASYNC,
        async_rl.Scale.SCREENING,
        **base,
        optimizer_precision=async_rl.OptimizerPrecision.NATIVE,
        optimizer_state_metrics=False,
    )
    assert explicit == native
    assert "optimizer_state_metrics" not in yaml.safe_load(native)["trainer"]
    with pytest.raises(ValueError, match="require policy training spans"):
        async_rl.training_config(
            async_rl.Runner.ASYNC,
            async_rl.Scale.SCREENING,
            spans=False,
            staleness=1,
            optimizer_precision=async_rl.OptimizerPrecision.BF16_BOTH,
        )
    with pytest.raises(ValueError, match="screening scale"):
        async_rl.training_config(
            async_rl.Runner.ASYNC,
            async_rl.Scale.QUALIFICATION,
            **base,
            optimizer_state_metrics=True,
        )


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("runner", list(async_rl.Runner))
def test_epoch_shuffle_cli_and_builder_share_reproducible_variant_identity(recipe, runner):
    version = "2026.09.06.10"
    options = dict(version=version, runner=runner, completion="metrics", epoch_seeded_shuffle=True)
    arguments = ["--version", version, "--runner", runner.value, "--completion", "metrics"]
    if recipe is async_rl:
        options.update(cluster="cw-us-east-02a", scale=recipe.Scale.SCREENING)
        arguments += ["--scale", "screening", "--stage", "rl"]
        built, _ = recipe.build_experiment(**options)
    else:
        options.update(scale=recipe.Scale.QUALIFICATION, timeout_seconds=3600)
        arguments += ["--scale", "qualification"]
        built = recipe.build_experiment(**options)
    built_request = built.build_config(StepContext.for_fingerprint(built.runtime_args, built.deps)).request
    requests = []
    for flags in ([], ["--no-epoch-seeded-shuffle"], ["--epoch-seeded-shuffle"], ["--epoch-seeded-shuffle"]):
        result = CliRunner().invoke(recipe.main, arguments + flags)
        assert result.exit_code == 0, result.output
        requests.append(json.loads(result.output)["request"])
    assert requests[0]["config_yaml"] == requests[1]["config_yaml"]
    assert requests[0]["run_id"] == requests[1]["run_id"] != requests[2]["run_id"]
    assert requests[2]["run_id"] == requests[3]["run_id"] == built_request.run_id
    assert requests[2]["config_yaml"] == built_request.config_yaml
    baseline, enabled = [yaml.safe_load(requests[i]["config_yaml"]) for i in (0, 2)]
    assert "epoch_seeded_shuffle" not in baseline["data"]
    assert enabled["data"].pop("epoch_seeded_shuffle") is True
    assert enabled == baseline
    for field in ("model", "train_data", "validation_data", "runtime", "topology", "seed"):
        assert requests[2][field] == requests[0][field]


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_epoch_shuffle_builder_rejects_non_boolean_values(recipe, value):
    with pytest.raises(ValueError, match="epoch_seeded_shuffle must be a boolean"):
        if recipe is async_rl:
            qwen_metrics_request(epoch_seeded_shuffle=value)
        else:
            recipe.build_experiment(
                version="2026.09.06.10",
                scale=recipe.Scale.QUALIFICATION,
                timeout_seconds=3600,
                epoch_seeded_shuffle=value,
            )


@pytest.mark.parametrize(
    ("recipe", "schedule"),
    [
        (async_rl, ["--scale", "screening", "--screening-steps", "100", "--stage", "rl"]),
        (async_snowball, ["--scale", "qualification", "--study-steps", "100"]),
    ],
)
def test_study_preview_locks_schedule_seed_and_validation_identity(recipe, schedule):
    args = ["--version", "2026.09.06.10", "--completion", "metrics", *schedule]

    def preview(extra):
        result = CliRunner().invoke(recipe.main, args + extra)
        assert result.exit_code == 0, result.output
        return json.loads(result.output)["request"]

    baseline = preview([])
    options = ["--eval-interval", "25", "--seed", "29", "--validation-offset", "128", "--validation-rows", "1191"]
    locked = preview(options)
    repeated = preview(options)
    assert locked["run_id"] == repeated["run_id"] != baseline["run_id"]
    assert locked["validation_data"] == repeated["validation_data"] != baseline["validation_data"]
    assert locked["seed"] == 29
    config = yaml.safe_load(locked["config_yaml"])
    assert config["trainer"]["max_steps"] == 100
    assert config["trainer"]["eval_interval"] == 25
    assert config["trainer"]["eval_before_train"]
    # Evaluation batches are independent of the selected set size; the final
    # short batch must remain eligible in the runtime (drop_last=False).
    assert config["trainer"]["eval_batch_size"] == 128
    for change in [["--seed", "43"], ["--eval-interval", "20"], ["--validation-rows", "1000"]]:
        assert preview(options + change)["run_id"] != locked["run_id"]


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("offset,rows", [(0, 1191), (127, 3), (128, 1192), (1319, 1)])
def test_study_rejects_overlapping_or_out_of_bounds_windows_before_submission(monkeypatch, recipe, offset, rows):
    submitted = []
    monkeypatch.setattr(recipe, "run", lambda step, **_kwargs: submitted.append(step))
    monkeypatch.setattr(recipe, "validate_regional_storage", lambda *_args: None)
    args = [
        "--version",
        "2026.09.06.10",
        "--completion",
        "metrics",
        "--validation-offset",
        str(offset),
        "--validation-rows",
        str(rows),
        "--run",
    ]
    if recipe is async_rl:
        args += ["--stage", "rl"]
    result = CliRunner().invoke(recipe.main, args)
    assert isinstance(result.exception, ValueError)
    assert "test[128:1319]" in str(result.exception)
    assert submitted == []


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("interval", [0, 7, 101])
def test_study_requires_a_complete_declared_evaluation_schedule(recipe, interval):
    if recipe is async_rl:
        with pytest.raises(ValueError, match="eval_interval"):
            recipe.training_config(
                recipe.Runner.ASYNC,
                recipe.Scale.SCREENING,
                spans=True,
                staleness=1,
                screening_steps=100,
                eval_interval=interval,
            )
    else:
        with pytest.raises(ValueError, match="eval_interval"):
            recipe.training_config(recipe.Scale.QUALIFICATION, study_steps=100, eval_interval=interval)


def test_explicit_eval_cadence_preserves_comparison_startup_contract():
    baseline = qwen_metrics_request(scale=async_rl.Scale.COMPARISON)
    variant = qwen_metrics_request(scale=async_rl.Scale.COMPARISON, eval_interval=10)
    before, after = [yaml.safe_load(request.config_yaml) for request in (baseline, variant)]
    assert before["trainer"]["eval_interval"] == 20
    assert after["trainer"]["eval_interval"] == 10
    assert not after["trainer"]["eval_before_train"]
    assert before["trainer"]["max_steps"] == after["trainer"]["max_steps"] == 20
    assert baseline.run_id != variant.run_id


@pytest.mark.parametrize("scale", [async_snowball.Scale.GATE, async_snowball.Scale.CADENCE_GATE])
def test_snowball_study_cannot_silently_repurpose_correctness_gate(scale):
    with pytest.raises(ValueError, match="qualification"):
        async_snowball.training_config(scale, study_steps=100)


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("oversized", [False, True])
def test_locked_window_preserves_source_indices_and_rejects_partial_filtering(tmp_path, monkeypatch, recipe, oversized):
    tokenizer_backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_think|>\n' }}{% endif %}"
    )
    model_path = tmp_path / "model"
    tokenizer.save_pretrained(model_path)
    original_load = pool.AutoTokenizer.from_pretrained
    monkeypatch.setattr(pool.AutoTokenizer, "from_pretrained", lambda *_args, **kwargs: original_load(model_path))

    def dataset_from_hub(_name, _subset, *, split, revision):
        records = [
            {"question": f"question {i}", "answer": "Work. #### 2"} for i in range(1024 if split == "train" else 131)
        ]
        if oversized and split == "test":
            records[129]["question"] = "word " * 1100
        return Dataset.from_list(records)

    monkeypatch.setattr(pool, "load_dataset", dataset_from_hub)
    options = dict(version="2026.09.06.10", completion="metrics", validation_offset=128, validation_rows=3)
    if recipe is async_rl:
        training, _ = recipe.build_experiment(
            **options, cluster="cw-us-east-02a", runner=recipe.Runner.ASYNC, scale=recipe.Scale.SCREENING
        )
        writer = recipe.write_gsm8k_window
    else:
        training = recipe.build_experiment(**options, scale=recipe.Scale.QUALIFICATION, timeout_seconds=3600)
        writer = recipe.write_snowball_window
    data = next(dep for dep in training.deps if "/documents/" in dep.name)
    manifests = []
    for suffix in ("first", "repeat"):
        output = tmp_path / suffix
        output.mkdir()
        context = replace(StepContext.for_fingerprint(data.runtime_args, data.deps), output_path=str(output))
        config = data.build_config(context)
        if recipe is async_snowball:
            # Substitute only the external model artifact with a real local tokenizer.
            config = replace(config, subset=replace(config.subset, model_path=str(model_path)))
        if oversized:
            with pytest.raises(ValueError, match=r"truncated|prompt exceeds"):
                writer(config)
            assert not (output / "selection.json").exists()
            return
        writer(config)
        manifest = json.loads((output / "selection.json").read_text())
        rows = pq.read_table(output / "validation.parquet").to_pylist()
        assert [row["extra_info"]["index"] for row in rows] == [128, 129, 130]
        assert manifest["rows"]["test"] == ["test/128", "test/129", "test/130"]
        assert manifest["validation_window"] == {
            "purpose": "locked_holdout",
            "split": "test",
            "offset": 128,
            "count": 3,
            "excluded_development_rows": [0, 128],
        }
        assert not set(manifest["rows"]["test"]) & {f"test/{i}" for i in range(128)}
        manifests.append(manifest)
    assert manifests[0] == manifests[1]


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


@pytest.mark.parametrize("scale", list(async_snowball.Scale))
def test_snowball_correction_package_preserves_default_and_changes_only_objective(scale):
    args = ["--version", "2026.09.06.13", "--scale", scale.value, "--completion", "metrics"]
    requests = []
    for extra in ([], ["--correction", "behavior_clip"], ["--correction", "regular_tis"]):
        result = CliRunner().invoke(async_snowball.main, args + extra)
        assert result.exit_code == 0, result.output
        requests.append(json.loads(result.output)["request"])
    baseline, explicit, tis = requests
    assert baseline == explicit
    assert baseline["run_id"] != tis["run_id"]
    configs = [yaml.safe_load(request["config_yaml"]) for request in (baseline, tis)]
    algorithm = configs[1]["trainer"]["algorithm"]
    assert algorithm.pop("tis_imp_ratio_cap") == 2.0
    assert algorithm.pop("require_rollout_logprobs") is True
    assert algorithm["policy_loss_type"] == "regular"
    assert algorithm["use_tis"] is True
    assert algorithm["use_kl_loss"] is False
    assert algorithm["use_kl_in_reward"] is False
    algorithm.update(policy_loss_type="behavior_clip", use_tis=False)
    assert configs[0] == configs[1]
    assert configs[1]["generator"]["sampling_params"]["logprobs"] == 0
    for field in ("model", "train_data", "validation_data", "topology", "runtime", "seed", "completion_mode"):
        assert baseline[field] == tis[field]
    step = async_snowball.build_experiment(
        version="2026.09.06.13",
        scale=scale,
        timeout_seconds=3600,
        completion="metrics",
        correction=async_rl.Correction.REGULAR_TIS,
    )
    built = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request
    assert built.config_yaml == tis["config_yaml"]
    assert built.run_id == tis["run_id"]


def test_snowball_rejects_unknown_correction_before_submission(monkeypatch):
    submitted = []
    monkeypatch.setattr(async_snowball, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        async_snowball.main,
        ["--version", "2026.09.06.13", "--correction", "unknown", "--run"],
    )
    assert result.exit_code != 0
    assert submitted == []
    with pytest.raises(ValueError, match="Unknown correction mode"):
        async_snowball.training_config(async_snowball.Scale.CADENCE_GATE, correction="unknown")


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


@pytest.mark.parametrize(
    "change",
    [{"seed": 18}, {"kl_loss": True}, {"correction": async_rl.Correction.REGULAR_TIS}, {"correction": "regular_no_tis"}],
)
def test_qwen_objective_and_seed_changes_have_distinct_identity(change):
    baseline = qwen_metrics_request()
    variant = qwen_metrics_request(
        **(change | {"correction": async_rl.Correction(change["correction"])} if "correction" in change else change)
    )
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
        use_tis = change["correction"] == async_rl.Correction.REGULAR_TIS
        if use_tis:
            assert algorithm.pop("tis_imp_ratio_cap") == 2.0
        assert algorithm.pop("require_rollout_logprobs") is True
        assert algorithm["policy_loss_type"] == "regular"
        assert algorithm["use_tis"] is use_tis
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


@pytest.mark.parametrize(
    "recipe,arguments,steps",
    [
        (async_rl, ["--stage", "rl", "--scale", "screening"], 25),
        (async_snowball, ["--scale", "cadence-gate"], 5),
    ],
)
def test_observation_cli_preview_preserves_defaults_and_isolates_enabled_variants(recipe, arguments, steps):
    requests = []
    for options in (
        [],
        ["--initial-eval-repeat-count", "1", "--no-weight-change-probe"],
        ["--initial-eval-repeat-count", "3"],
        ["--weight-change-probe"],
        ["--initial-eval-repeat-count", "3", "--weight-change-probe"],
    ):
        result = CliRunner().invoke(
            recipe.main,
            ["--version", "2026.09.06.12", "--completion", "metrics", "--dry-run", *arguments, *options],
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        requests.append(json.loads(result.output)["request"])
    assert requests[0]["config_yaml"] == requests[1]["config_yaml"]
    assert requests[0]["run_id"] == requests[1]["run_id"]
    assert len({request["run_id"] for request in requests}) == 4
    baseline = yaml.safe_load(requests[0]["config_yaml"])
    assert "initial_eval_repeat_count" not in baseline["trainer"]
    assert "weight_change_probe" not in baseline["trainer"]
    for request, repeats, probe in zip(requests[2:], (3, 1, 3), (False, True, True), strict=True):
        config = yaml.safe_load(request["config_yaml"])
        trainer = config["trainer"]
        assert trainer.pop("initial_eval_repeat_count", 1) == repeats
        assert trainer.pop("weight_change_probe", False) == probe
        assert trainer["max_steps"] == trainer["eval_interval"] == steps
        assert trainer["eval_before_train"]
        assert trainer["strategy"] == "megatron"
        assert not trainer["placement"]["colocate_all"]
        assert config["generator"]["run_engines_locally"]
        assert not config["generator"].get("fuse_weights", False)
        assert config == baseline
        for key in ("model", "train_data", "validation_data", "topology", "runtime", "seed", "completion_mode"):
            assert request[key] == requests[0][key]


@pytest.mark.parametrize(
    "recipe,arguments",
    [
        (async_rl, ["--stage", "rl", "--scale", "comparison"]),
        (async_snowball, ["--scale", "gate"]),
    ],
)
def test_observation_repeats_cannot_silently_enable_disabled_evaluation(recipe, arguments, monkeypatch):
    submitted = []
    monkeypatch.setattr(recipe, "run", lambda *args, **kwargs: submitted.append(args))
    result = CliRunner().invoke(
        recipe.main,
        [
            "--version",
            "2026.09.06.12",
            "--completion",
            "metrics",
            "--run",
            *arguments,
            "--initial-eval-repeat-count",
            "3",
        ],
    )
    assert result.exit_code != 0
    assert "require a schedule" in str(result.exception)
    assert submitted == []


@pytest.mark.parametrize("recipe", [async_rl, async_snowball])
@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_observation_cli_rejects_invalid_repeat_count(recipe, value):
    result = CliRunner().invoke(recipe.main, ["--version", "2026.09.06.12", "--initial-eval-repeat-count", value])
    assert result.exit_code != 0
    assert "Invalid value for '--initial-eval-repeat-count'" in result.output


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_observation_builder_rejects_invalid_repeat_count(value):
    with pytest.raises(ValueError, match="positive integer"):
        qwen_metrics_request(initial_eval_repeat_count=value)


@pytest.mark.parametrize("value", [1, None, "true"])
def test_observation_builder_requires_boolean_probe(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        qwen_metrics_request(weight_change_probe=value)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("trainer", "strategy", "fsdp2"),
        ("placement", "colocate_all", True),
        ("generator", "fuse_weights", True),
        ("generator", "run_engines_locally", False),
    ],
)
def test_observation_probe_rejects_unsupported_resolved_runtime(section, key, value):
    config = yaml.safe_load(async_snowball.training_config(async_snowball.Scale.CADENCE_GATE))
    target = config["trainer"]["placement"] if section == "placement" else config[section]
    target[key] = value
    with pytest.raises(ValueError, match="Weight change probe requires"):
        async_rl.apply_observation_options(config, initial_eval_repeat_count=1, weight_change_probe=True)
    assert "weight_change_probe" not in config["trainer"]


def test_pool_flag_preserves_default_preview_and_selects_verified_fixed_views():
    baseline = qwen_metrics_request()
    explicit_default = qwen_metrics_request(pool_artifact=None)
    selected = qwen_metrics_request(pool_artifact=POOL_ARGUMENT)
    assert baseline.run_id == explicit_default.run_id
    assert baseline.config_yaml == explicit_default.config_yaml
    assert baseline.train_data == explicit_default.train_data
    assert baseline.validation_data == explicit_default.validation_data
    assert selected.run_id != baseline.run_id
    assert selected.config_yaml == baseline.config_yaml
    assert selected.train_data[0].relative_path == "qwen/train.parquet"
    assert selected.validation_data[0].relative_path == BATTERY_PATH + "/qwen/dev.parquet"
    result = CliRunner().invoke(
        async_rl.main,
        ["--version", "2026.09.06.10", "--stage", "rl", "--completion", "metrics", "--pool-artifact", POOL_ARGUMENT],
    )
    assert result.exit_code == 0, result.output
    cli = json.loads(result.output)["request"]
    assert cli["train_data"][0]["relative_path"] == "qwen/train.parquet"
    assert cli["validation_data"][0]["relative_path"] == BATTERY_PATH + "/qwen/dev.parquet"


@pytest.mark.parametrize("changes", [{"validation_offset": 128}, {"context_tokens": 1536}])
def test_pool_flag_rejects_legacy_window_or_insufficient_prompt_budget(changes):
    with pytest.raises(ValueError, match=r"frozen pool|Context budget"):
        qwen_metrics_request(pool_artifact=POOL_ARGUMENT, **changes)


def test_qwen_rno_exception_is_explicit_and_preserves_recipe(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    args = [
        "--version",
        "2026.09.08.1",
        "--cluster",
        "cw-rno2a",
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "metrics",
    ]
    before = CliRunner().invoke(async_rl.main, args)
    allowed = CliRunner().invoke(async_rl.main, [*args, "--allow-cross-region-io"])
    assert before.exit_code == allowed.exit_code == 0, allowed.output
    before_request = json.loads(before.output)["request"]
    allowed_request = json.loads(allowed.output)["request"]
    before_request.pop("attempt_id")
    allowed_request.pop("attempt_id")
    assert before_request == allowed_request
    blocked = CliRunner().invoke(async_rl.main, [*args, "--run"])
    assert blocked.exit_code != 0
    assert "not local to cw-rno2a" in blocked.output
    submitted = []
    monkeypatch.setattr(async_rl, "run", lambda *args, **kwargs: submitted.append((args, kwargs)))
    executed = CliRunner().invoke(async_rl.main, [*args, "--allow-cross-region-io", "--run"])
    assert executed.exit_code == 0, executed.output
    assert len(submitted) == 1


@pytest.mark.parametrize(
    "change", [["--scale", "qualification"], ["--cluster", "cw-us-east-02a"], ["--stage", "evaluation"]]
)
def test_qwen_rno_exception_rejects_other_scopes(change):
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            "2026.09.08.1",
            "--cluster",
            "cw-rno2a",
            "--scale",
            "screening",
            "--stage",
            "rl",
            "--allow-cross-region-io",
            "--completion",
            "metrics",
            *change,
        ],
    )
    assert result.exit_code != 0
    assert "restricted to Qwen screening" in result.output


def test_rno_exception_rejects_other_buckets_and_snowball(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-west-04a/marin")
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            "2026.09.08.1",
            "--cluster",
            "cw-rno2a",
            "--scale",
            "screening",
            "--stage",
            "rl",
            "--allow-cross-region-io",
            "--completion",
            "metrics",
        ],
    )
    assert result.exit_code != 0
    assert "not local to cw-rno2a" in result.output
    snowball = CliRunner().invoke(async_snowball.main, ["--version", "2026.09.08.1", "--allow-cross-region-io"])
    assert snowball.exit_code != 0
    assert "No such option" in snowball.output


@pytest.mark.parametrize(
    "changed",
    [
        "model",
        "same_east_model",
        "train",
        "dev",
        "output",
        "ood",
        "dump",
        "hub",
        "save",
        "tokenizer",
        "revision",
        "identity",
    ],
)
def test_rno_resolved_request_guard_checks_model_and_every_io_surface(changed):
    request = qwen_metrics_request()
    prefix = "s3://marin-us-east-02a/marin"
    request = replace(
        request,
        model=replace(request.model, uri=prefix + "/users/ahmad/models/async-rl-qwen3-0.6b/2026.09.06.10/hf"),
        train_data=tuple(replace(item, uri=prefix + "/train") for item in request.train_data),
        validation_data=tuple(replace(item, uri=prefix + "/dev") for item in request.validation_data),
        output=replace(request.output, **{key: prefix + "/" + key for key in asdict(request.output)}),
    )
    async_rl.validate_qwen_cross_region_request(request)
    foreign = "s3://marin-us-west-04a/changed"
    if changed == "model":
        request = replace(request, model=replace(request.model, uri=foreign))
    elif changed == "same_east_model":
        request = replace(request, model=replace(request.model, uri=prefix + "/models/snowball/hf"))
    elif changed in {"train", "dev"}:
        field = "train_data" if changed == "train" else "validation_data"
        request = replace(request, **{field: (replace(getattr(request, field)[0], uri=foreign),)})
    elif changed == "output":
        request = replace(request, output=replace(request.output, terminal_manifest_uri=foreign))
    elif changed == "ood":
        request = replace(request, config_yaml=request.config_yaml + "\nood_input: " + foreign)
    elif changed == "dump":
        request = replace(request, overrides=(*request.overrides, "++dump_path=" + foreign))
    elif changed == "hub":
        request = replace(request, overrides=(*request.overrides, "++trainer.hf_hub_repo_id=some/repository"))
    elif changed == "save":
        request = replace(request, overrides=(*request.overrides, "++trainer.hf_save_interval=1"))
    elif changed == "tokenizer":
        request = replace(request, model=replace(request.model, tokenizer_uri="Qwen/Qwen3-30B-A3B"))
    elif changed == "revision":
        request = replace(request, model=replace(request.model, tokenizer_revision="main"))
    else:
        request = replace(request, model=replace(request.model, identity="another/model@1.0:8a30d2b5"))
    with pytest.raises(ValueError, match="Cross-region I/O requires"):
        async_rl.validate_qwen_cross_region_request(request)


def test_rno_guard_executes_when_actual_training_configuration_resolves():
    step, _ = async_rl.build_experiment(
        version="2026.09.06.16",
        cluster="cw-rno2a",
        runner=async_rl.Runner.ASYNC,
        scale=async_rl.Scale.SCREENING,
        completion="metrics",
        allow_cross_region_io=True,
    )
    prefix = "s3://marin-us-east-02a/marin"
    ctx = StepContext(
        output_path=prefix + "/output",
        prefix=prefix,
        region="us-east-02a",
        is_fingerprint=False,
        _dep_ref=lambda dependency: prefix + "/" + dependency.name + "/" + dependency.version,
        _runtime_args=step.runtime_args,
        _deps=step.deps,
    )
    request = step.build_config(ctx).request
    assert request.model.tokenizer_uri == "Qwen/Qwen3-0.6B"
    foreign = replace(ctx, _dep_ref=lambda _dependency: "s3://marin-us-west-04a/changed")
    with pytest.raises(ValueError, match=r"pinned Qwen3-0.6B model path"):
        step.build_config(foreign)


@pytest.mark.parametrize("runner", ["sync", "async"])
def test_dataloader_worker_override_reaches_fingerprinted_recipe(runner):
    args = ["--version", "2026.09.08.1", "--runner", runner, "--stage", "rl", "--completion", "metrics"]
    legacy = CliRunner().invoke(async_rl.main, args)
    explicit = CliRunner().invoke(async_rl.main, [*args, "--dataloader-workers", "0"])
    assert legacy.exit_code == explicit.exit_code == 0
    before = json.loads(legacy.output)["request"]
    after = json.loads(explicit.output)["request"]
    assert before["run_id"] != after["run_id"]
    config = yaml.safe_load(after["config_yaml"])
    assert config["data"].pop("num_workers") == 0
    if not config["data"]:
        config.pop("data")
    assert config == yaml.safe_load(before["config_yaml"])


@pytest.mark.parametrize("mode", ["blocking", "background"])
def test_eval_scheduling_flags_are_opt_in_and_fingerprinted(mode):
    baseline = qwen_metrics_request(runner=async_rl.Runner.ASYNC)
    explicit = qwen_metrics_request(runner=async_rl.Runner.ASYNC, eval_on_installed_weights=False, eval_mode="blocking")
    selected = qwen_metrics_request(runner=async_rl.Runner.ASYNC, eval_on_installed_weights=True, eval_mode=mode)
    assert baseline.run_id == explicit.run_id
    assert baseline.config_yaml == explicit.config_yaml
    assert selected.run_id != baseline.run_id
    config = yaml.safe_load(selected.config_yaml)
    assert config["trainer"]["fully_async"].pop("eval_on_installed_weights") is True
    assert config["trainer"]["fully_async"].pop("eval_mode") == mode
    assert config == yaml.safe_load(baseline.config_yaml)


@pytest.mark.parametrize(
    "changes",
    [
        {"runner": async_rl.Runner.SYNC, "eval_on_installed_weights": True},
        {"eval_mode": "background"},
        {"eval_mode": "invalid"},
        {"eval_on_installed_weights": 1},
    ],
)
def test_eval_scheduling_rejects_unsupported_modes(changes):
    with pytest.raises(ValueError, match=r"Evaluation|evaluation"):
        qwen_metrics_request(**changes)


def test_background_eval_cli_reaches_native_recipe():
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            "2026.09.08.73",
            "--runner",
            "async",
            "--stage",
            "rl",
            "--completion",
            "metrics",
            "--eval-on-installed-weights",
            "--eval-mode",
            "background",
        ],
    )
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.output)["request"]["config_yaml"])
    assert config["trainer"]["fully_async"]["eval_mode"] == "background"
    assert config["trainer"]["fully_async"]["eval_on_installed_weights"] is True


@pytest.mark.parametrize("runner", list(async_rl.Runner))
def test_seeded_control_cli_and_builder_bind_identical_requests(runner):
    options = dict(
        version="2026.09.08.92",
        cluster="cw-us-east-02a",
        runner=runner,
        completion="metrics",
        scale=async_rl.Scale.SCREENING,
        epoch_seeded_shuffle=True,
        staleness=0,
        seeded_sampling_control=True,
    )
    built, _ = async_rl.build_experiment(**options)
    expected = built.build_config(StepContext.for_fingerprint(built.runtime_args, built.deps)).request
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            options["version"],
            "--runner",
            runner.value,
            "--completion",
            "metrics",
            "--scale",
            "screening",
            "--stage",
            "rl",
            "--epoch-seeded-shuffle",
            "--staleness",
            "0",
            "--seeded-sampling-control",
        ],
    )
    assert result.exit_code == 0, result.output
    actual = json.loads(result.output)["request"]
    assert actual["config_yaml"] == expected.config_yaml and actual["run_id"] == expected.run_id
    config = yaml.safe_load(actual["config_yaml"])
    assert config["generator"]["seed_by_trajectory"] is True
    assert config["generator"]["enable_prefix_caching"] is False
    assert config["extra_env"]["VLLM_BATCH_INVARIANT"] == "1"
    baseline, _ = async_rl.build_experiment(**(options | {"seeded_sampling_control": False}))
    baseline_config = yaml.safe_load(
        baseline.build_config(StepContext.for_fingerprint(baseline.runtime_args, baseline.deps)).request.config_yaml
    )
    del config["generator"]["seed_by_trajectory"]
    del config["generator"]["enable_prefix_caching"]
    del config["extra_env"]
    assert config == baseline_config


@pytest.mark.parametrize("runner", list(async_rl.Runner))
def test_symmetric_environment_control_changes_only_paired_worker_settings(runner):
    options = dict(
        version="2026.09.08.96",
        cluster="cw-us-east-02a",
        runner=runner,
        completion="metrics",
        scale=async_rl.Scale.SCREENING,
        epoch_seeded_shuffle=True,
        staleness=0,
        seeded_sampling_control=True,
    )
    baseline, _ = async_rl.build_experiment(**options)
    candidate, _ = async_rl.build_experiment(**options, symmetric_weight_sync_environment=True)
    old = baseline.build_config(StepContext.for_fingerprint(baseline.runtime_args, baseline.deps)).request
    new = candidate.build_config(StepContext.for_fingerprint(candidate.runtime_args, candidate.deps)).request
    config = yaml.safe_load(new.config_yaml)
    assert config["extra_env"].pop("RAY_DEDUP_LOGS_ALLOW_REGEX") == "WEIGHT_SYNC_ENVIRONMENT_PRE_PG"
    assert config["trainer"]["algorithm"].pop("weight_sync_invariant_env") is True
    assert config["generator"]["engine_init_kwargs"].pop("worker_cls") == (
        "skyrl_train.inference_engines.vllm.invariant_worker.InvariantWeightSyncWorker"
    )
    if not config["generator"]["engine_init_kwargs"]:
        del config["generator"]["engine_init_kwargs"]
    assert config == yaml.safe_load(old.config_yaml)
    assert old.run_id != new.run_id
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            options["version"],
            "--runner",
            runner.value,
            "--completion",
            "metrics",
            "--scale",
            "screening",
            "--stage",
            "rl",
            "--epoch-seeded-shuffle",
            "--staleness",
            "0",
            "--seeded-sampling-control",
            "--symmetric-weight-sync-environment",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["request"]["config_yaml"] == new.config_yaml


@pytest.mark.parametrize("value", [True, "true", 1])
def test_symmetric_environment_rejects_unpaired_or_nonboolean_control(value):
    with pytest.raises(ValueError):
        async_rl.build_experiment(
            version="2026.09.08.96",
            cluster="cw-us-east-02a",
            runner=async_rl.Runner.SYNC,
            scale=async_rl.Scale.SCREENING,
            symmetric_weight_sync_environment=value,
        )


@pytest.mark.parametrize("value", [1, None, "true"])
def test_qwen_weight_sync_trace_requires_boolean(value):
    with pytest.raises(ValueError, match="publication_stage_timing must be a boolean"):
        qwen_metrics_request(publication_stage_timing=value)


def test_qwen_serial_engine_startup_is_opt_in_and_changes_request_identity():
    args = ["--version", "2026.09.08.1", "--stage", "rl", "--completion", "metrics"]
    requests = []
    for extra in [[], ["--no-serial-engine-startup"], ["--serial-engine-startup"]]:
        result = CliRunner().invoke(async_rl.main, [*args, *extra])
        assert result.exit_code == 0, result.output
        requests.append(json.loads(result.output)["request"])
    before, off, after = requests
    assert before["run_id"] == off["run_id"] != after["run_id"]
    assert before["config_yaml"] == off["config_yaml"]
    config = yaml.safe_load(after["config_yaml"])
    assert config["generator"].pop("inference_engine_serial_startup") is True
    assert config == yaml.safe_load(before["config_yaml"])


@pytest.mark.parametrize("value", [1, None, "true"])
def test_qwen_serial_startup_requires_boolean(value):
    with pytest.raises(ValueError, match="serial_engine_startup must be a boolean"):
        qwen_metrics_request(serial_engine_startup=value)


def test_qwen_weight_sync_trace_is_opt_in_and_changes_request_identity():
    args = ["--version", "2026.09.08.1", "--stage", "rl", "--completion", "metrics"]
    default = CliRunner().invoke(async_rl.main, args)
    disabled = CliRunner().invoke(async_rl.main, [*args, "--no-publication-stage-timing"])
    enabled = CliRunner().invoke(async_rl.main, [*args, "--publication-stage-timing"])
    assert default.exit_code == disabled.exit_code == enabled.exit_code == 0
    before = json.loads(default.output)["request"]
    off = json.loads(disabled.output)["request"]
    after = json.loads(enabled.output)["request"]
    assert before["run_id"] == off["run_id"] != after["run_id"]
    assert before["config_yaml"] == off["config_yaml"]
    config = yaml.safe_load(after["config_yaml"])
    assert config["generator"].pop("publication_stage_timing") is True
    assert config["generator"].pop("inference_stats_poll_seconds") == 1.0
    assert config == yaml.safe_load(before["config_yaml"])


@pytest.mark.parametrize("foreign_field", [None, "model", "train", "dev", "output", "dump"])
def test_snowball_cross_region_guard_preserves_east_artifacts(foreign_field):
    step = async_snowball.build_experiment(
        version="2026.09.06.10",
        scale=async_snowball.Scale.QUALIFICATION,
        timeout_seconds=4800,
        completion="metrics",
    )
    prefix = "s3://marin-us-east-02a/marin"
    context = StepContext(
        output_path=prefix + "/output",
        prefix=prefix,
        region="us-east-02a",
        is_fingerprint=False,
        _dep_ref=lambda dependency: prefix + "/" + dependency.name + "/" + dependency.version,
        _runtime_args=step.runtime_args,
        _deps=step.deps,
    )
    request = step.build_config(context).request
    before = asdict(request)
    async_rl.validate_cross_region_request(request)
    assert asdict(request) == before
    if foreign_field is None:
        return
    foreign = "s3://marin-us-west-04a/changed"
    if foreign_field == "model":
        request = replace(request, model=replace(request.model, uri=foreign))
    elif foreign_field in {"train", "dev"}:
        field = "train_data" if foreign_field == "train" else "validation_data"
        request = replace(request, **{field: (replace(getattr(request, field)[0], uri=foreign),)})
    elif foreign_field == "output":
        request = replace(request, output=replace(request.output, terminal_manifest_uri=foreign))
    else:
        request = replace(request, overrides=(*request.overrides, "++dump_path=" + foreign))
    with pytest.raises(ValueError, match="Cross-region I/O requires east S3"):
        async_rl.validate_cross_region_request(request)
