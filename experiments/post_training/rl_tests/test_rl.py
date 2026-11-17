# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import click
import pytest
import yaml
from levanter import model_cache
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint
from rigging.provenance import Provenance

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import grafana_url
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, SELECTION_PREFIX

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    changed = snowball.build_run(
        learning,
        settings=(
            'trainer={"policy_mini_batch_size":64,"train_batch_size":64,"bf16":false,"algorithm":{"batch_invariant":true}}',
            "generator.model_dtype=float16",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
            "trainer.policy.optimizer_config.lr=2e-6",
            "generator.engine_init_kwargs.probe={}",
        ),
        table=snowball_model.SNOWBALL,
    ).runs[0]
    promoted = replace(
        learning,
        recipe=replace(
            learning.recipe,
            config=learning.recipe.config.with_settings(
                (
                    "trainer.train_batch_size=64",
                    "trainer.policy_mini_batch_size=64",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                    "trainer.policy.optimizer_config.lr=2e-6",
                )
            ),
        ),
    )
    checkpoint = tmp_path / "checkpoint"
    export = checkpoint / "hf"
    export.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (export / name).write_text("{}")
    metadata = {"train_config": {"data": {"tokenizer": "tests/own-tokenizer"}}}
    monkeypatch.setattr(model_cache, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    initializer = ArtifactStep.adopt(
        "tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint, config=metadata
    )
    alias = tmp_path / initializer.name / initializer.version
    alias.mkdir(parents=True)
    (alias / RECORD_FILENAME).write_text(
        ArtifactRecord(
            name=initializer.name,
            version=initializer.version,
            result_type=result_type_name(LevanterCheckpoint),
            config=metadata,
            source=str(checkpoint),
        ).model_dump_json()
    )
    cli_initializer = recipe.published_model(f"{initializer.name}@{initializer.version}", preset=learning)
    published = []
    for suffix, config in (
        ("own", {}),
        (
            "explicit",
            {
                "tokenizer_uri": "tests/top-tokenizer",
                "tokenizer_revision": "top-revision",
                "train_config": {"data": {"tokenizer": "tests/nested-ignored"}},
            },
        ),
    ):
        name = "tests/checkpoint-" + suffix
        record = tmp_path / name / initializer.version / RECORD_FILENAME
        record.parent.mkdir(parents=True)
        record.write_text(
            ArtifactRecord(
                name=name,
                version=initializer.version,
                result_type=result_type_name(Artifact),
                config=config,
                source=str(checkpoint),
            ).model_dump_json()
        )
        published.append(recipe.published_model(f"{name}@{initializer.version}", preset=learning))

    revisioned = replace(snowball_model.SNOWBALL.by_name["speed-short"], revision="2026.11.12.2")
    steps = (
        changed,
        snowball.build_run(promoted, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(
            learning,
            init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision"),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(
            snowball_model.SNOWBALL.by_name["train-long"],
            settings=("trainer.max_steps=2",),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(revisioned, control=True, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(learning, init=cli_initializer, table=snowball_model.SNOWBALL).runs[0],
        *(snowball.build_run(learning, init=model, table=snowball_model.SNOWBALL).runs[0] for model in published),
    )
    paths = []
    for index, step in enumerate(steps):
        path = tmp_path / f"variant-{index}.yaml"
        path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    for config in configs[:2]:
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert (
            trainer["train_batch_size"],
            trainer["policy_mini_batch_size"],
            generator["n_samples_per_prompt"],
            trainer["micro_train_batch_size_per_gpu"],
            trainer["micro_forward_batch_size_per_gpu"],
            trainer["rollout_buffer"]["max_in_flight"],
            trainer["policy"]["optimizer_config"]["lr"],
        ) == (64, 64, 8, 2, 2, 64, 2e-6)
        assert (
            trainer["policy"]["optimizer_config"]["weight_decay"],
            trainer["policy"]["optimizer_config"]["adam_betas"],
        ) == (0, [0.9, 0.999])
    trainer, generator = configs[0]["skyrl"]["trainer"], configs[0]["skyrl"]["generator"]
    assert (
        trainer["bf16"],
        generator["model_dtype"],
        trainer["algorithm"]["batch_invariant"],
        generator["engine_init_kwargs"]["probe"],
    ) == (False, "float16", True, {})
    assert configs[2]["inputs"]["model"]["uri"] == str(export)
    assert configs[2]["iris"]["wandb_entity"] == "test-team"
    for config, revision in ((configs[2], "tokenizer-revision"), (configs[5], "a" * 40)):
        assert (config["inputs"]["model"]["tokenizer_uri"], config["inputs"]["model"]["tokenizer_revision"]) == (
            "tests/own-tokenizer",
            revision,
        )
        assert config["inputs"]["model"]["uri"] == str(export)
    assert (configs[6]["inputs"]["model"]["tokenizer_uri"], configs[6]["inputs"]["model"]["tokenizer_revision"]) == (
        None,
        None,
    )
    assert (configs[7]["inputs"]["model"]["tokenizer_uri"], configs[7]["inputs"]["model"]["tokenizer_revision"]) == (
        "tests/top-tokenizer",
        "top-revision",
    )
    callbacks = configs[3]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[3]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[4]["inputs"]["validation_data"][0]["identity"].partition(":")[0].endswith("@" + revisioned.revision)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def export_model(output):
        destination = Path(output) / "hf"
        destination.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json"):
            (destination / name).write_text("{}")
        return Artifact(path=output)

    producer = apply("tests/computed-init", export_model, version="2026.11.17", output=OUT)
    own_model = ArtifactHfModel(producer)
    train = snowball_model.SNOWBALL.by_name["train-long"]

    def reject_export_read(*args, **kwargs):
        raise AssertionError("planning read an unbuilt export")

    with monkeypatch.context() as planning:
        planning.setattr(skyrl_api, "discover_hf_checkpoints", reject_export_read)
        planned = snowball.build_run(train, init=own_model, table=snowball_model.SNOWBALL)
        assert producer in planned.runs[0].deps
        planned.runs[0].lower()
        assert (
            snowball.main(
                snowball_model.SNOWBALL, argv=["--preset", train.name, "--version", train.revision], init=own_model
            )
            == 0
        )
    assert not Path(producer.path()).exists()
    run(producer)
    document = yaml.safe_load(materialized_config(planned.runs[0], str(tmp_path)).launch_config_yaml)
    assert document["inputs"]["model"]["uri"] == str(Path(producer.path()) / "hf")
    assert document["inputs"]["model"]["tokenizer_uri"] is None
    assert document["inputs"]["model"]["tokenizer_revision"] is None
    best = next(row for row in document["skyrl"]["trainer"]["callbacks"] if row["type"] == "best_checkpoint")
    assert best["initial_model_uri"] == document["inputs"]["model"]["uri"]
    with pytest.raises(ValueError, match="require guards"):
        replace(learning, guards=())
    with pytest.raises(ValueError, match="evaluation"):
        replace(snowball_model.SNOWBALL.by_name["train-long"], evaluations=())
    summary = Path(recipe.selection_artifact(learning.tasks).path()) / recipe.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(learning, table=snowball_model.SNOWBALL)


def test_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch):
    revisions = {
        "speed-short": {"2026.11.13.1": "96e0d088"},
        "data-short": {"2026.11.13.1": "7d9a590f"},
        "speed-medium": {"2026.11.12.1": "e75349ff"},
        "learning-medium": {"2026.11.12.1": "ba638a12"},
        "data-medium": {"2026.11.12.1": "ba638a12"},
        "speed-long": {"2026.11.11.1": "cd0a5f4e"},
        "train-long": {"2026.11.11.1": "4945b701"},
    }
    configs = assert_revisioned_native_configs(
        snowball_model.SNOWBALL, revisions, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    for name, config in zip(revisions, configs, strict=True):
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert config["iris"]["wandb_entity"] == "marin-community"
        assert config["iris"]["allocation"]["num_nodes"] * config["iris"]["allocation"]["gpus_per_node"] <= 40
        generating = name in ("speed-short", "data-short")
        assert (
            config["runtime"]["training_type"],
            trainer["rollout_buffer"]["max_staleness_steps"],
            trainer["algorithm"]["off_policy_correction"],
            trainer["rollout_buffer"]["batch_policy"],
        ) == ((None, 0, "none", "full_batch") if generating else ("async", 2, "tis", "full_batch"))
        if generating:
            assert config["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
            assert config["iris"]["allocation"]["num_nodes"] == 1
            assert config["skyrl"]["trajectory_runner"]["max_concurrent_tasks"] == (
                2048 if name == "speed-short" else 1024
            )
            assert trainer["eval_interval"] > 0 and config["inputs"]["validation_data"]
            sampling = generator["eval_sampling_params"]
            assert (sampling["temperature"], sampling["max_generate_length"], sampling["ignore_eos"]) == (
                1,
                4096 if name == "speed-short" else 8192,
                name == "speed-short",
            )
        else:
            assert generator["apply_overlong_filtering"] is False
        assert (trainer["bf16"], generator["model_dtype"], trainer["algorithm"]["batch_invariant"]) == (
            True,
            "bfloat16",
            False,
        )
        if name in ("speed-short", "learning-medium", "data-medium", "train-long"):
            profile = next(
                (
                    row["additional_evaluations"]["eval_lite"]
                    for row in trainer["callbacks"]
                    if "eval_lite" in row.get("additional_evaluations", {})
                ),
                None,
            )
            for source in config["inputs"].get("evaluation_data", {}).get("eval_lite", []):
                assert (
                    source["identity"].partition(":")[0]
                    == f"{SELECTION_PREFIX}/{snowball_model.SNOWBALL.by_name[name].eval_lite}"
                )
                assert source["relative_path"] == "full_validation.parquet"
            sampling = profile["sampling_params"] if profile else generator["eval_sampling_params"]
            assert (
                sampling["temperature"],
                sampling["ignore_eos"],
                sampling["max_generate_length"],
                profile["n_samples_per_prompt"] if profile else generator["eval_n_samples_per_prompt"],
            ) == (0, False, 8192, 1)
        if name in ("learning-medium", "data-medium"):
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_before_train"] is True
            assert config["run"]["export_hf"] and "hf_model_save" in [row["type"] for row in trainer["callbacks"]]
        if name == "speed-long":
            assert not config["inputs"]["validation_data"] and not config["run"]["export_hf"]
        if name == "train-long":
            kinds = [row["type"] for row in trainer["callbacks"]]
            assert kinds.index("evaluation") < kinds.index("best_checkpoint")
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_on_train_end"] is False


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    data = snowball_model.SNOWBALL.by_name["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_run(
            replace(learning, accepts_settings=False),
            settings=("trainer.policy.optimizer_config.lr=2e-6",),
            table=snowball_model.SNOWBALL,
        )

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=settings, **kwargs, table=snowball_model.SNOWBALL).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    assert name(learning, "generator.engine_init_kwargs.probe={}") != name()
    repeated = snowball.build_run(learning, replicates=3, table=snowball_model.SNOWBALL)
    assert [step.name for step in snowball.build_run(learning, replicates=2, table=snowball_model.SNOWBALL).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    task_set_name = snowball_model.DEFAULT_EVAL_LITE_NAME
    task_set_version = snowball_model.DEFAULT_EVAL_LITE_VERSION
    new_task_set = tmp_path / task_set_name / "2026.11.05"
    new_task_set.mkdir(parents=True)
    (new_task_set / "summary.json").write_bytes(
        (tmp_path / task_set_name / task_set_version / "summary.json").read_bytes()
    )
    changed_task_set = replace(
        learning,
        eval_lite=f"{task_set_name}@2026.11.05",
    )
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision", relative_path=""))),
        (name(), name(changed_task_set)),
    ]
    assert all(left != right for left, right in pairs), pairs
    assert name(
        learning,
        'trainer.policy.optimizer_config={"lr": 2e-06}',
        'trainer.policy.optimizer_config={"weight_decay": 0.1}',
    ) == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002", "trainer.policy.optimizer_config.weight_decay=0.1"
    )
    assert name(learning, "trainer.policy.optimizer_config.lr=2e-6") == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002"
    )
    pin = gate.control_pin(snowball_model.SNOWBALL.by_name["learning-medium"], table=snowball_model.SNOWBALL)
    for index in range(pin.runs):
        recorded_runs("learning-medium", gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL))
    first = snowball.build_run(learning, requirements=("eval_lite/mean>=0",), table=snowball_model.SNOWBALL)
    second = snowball.build_run(learning, requirements=("eval_lite/mean>=0.1",), table=snowball_model.SNOWBALL)
    assert first.runs[0].name == second.runs[0].name == name()
    assert first.gate.name != second.gate.name
    original_name = name()
    monkeypatch.setenv("WANDB_ENTITY", "another-team")
    assert name() == original_name
    monkeypatch.setattr(
        Provenance, "capture", lambda *args, **kwargs: Provenance("tree-two", "commit-one", False, "branch", "test")
    )
    assert name() != original_name

    summary_path = Path(selection.path()) / "summary.json"
    summary = json.loads(summary_path.read_text())
    original_exclusions = summary["recipe"]["exclusions"]
    heldout = summary["recipe"]["heldout"]
    summary["recipe"]["heldout"] = [{"version": "different-development-input"}]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(click.BadParameter, match="different heldout"):
        snowball.build_run(data, tasks=selection, table=snowball_model.SNOWBALL)
    summary["recipe"]["heldout"] = heldout
    summary_path.write_text(json.dumps(summary))
    for current in (data, snowball_model.SNOWBALL.by_name["data-medium"]):
        selected = replace(current, eval_lite=f"{task_set_name}@2026.11.05")
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL)
        summary["recipe"]["exclusions"] = [
            {"uri": str(new_task_set / FULL_VALIDATION_FILENAME), "version": "2026.11.05"}
        ]
        summary_path.write_text(json.dumps(summary))
        assert snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL).gate is not None
        summary["recipe"]["exclusions"] = original_exclusions
        summary_path.write_text(json.dumps(summary))
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (recipe.selection_artifact(snowball_model.SNOWBALL.by_name["learning-medium"].tasks), selection):
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(data, tasks=tasks, table=snowball_model.SNOWBALL)


def test_grafana_url_selects_native_run_and_utc_window():
    result = urlparse(grafana_url("users/test/run", "marin-east", "rl", "2026-11-06T18:40:00Z", "2026-11-06T19:40:00Z"))
    assert result.path == "/d/marin-rl-runs/rl-post-training"
    assert parse_qs(result.query) == {
        "orgId": ["1"],
        "timezone": ["utc"],
        "from": ["1793990400000"],
        "to": ["1793994000000"],
        "var-training_type": ["rl"],
        "var-cluster": ["marin-east"],
        "var-run": ["users/test/run"],
        "var-job": ["$__all"],
        "var-execution": ["$__all"],
    }
    open_window = parse_qs(urlparse(grafana_url("generation", "marin-east", None, "2026-11-06T18:40:00Z")).query)
    assert open_window["to"] == ["now"] and open_window["var-training_type"] == ["$__all"]
