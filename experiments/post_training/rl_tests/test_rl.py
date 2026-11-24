# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse, urlsplit

import click
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import AttemptRecord, read_attempt, read_evaluations, write_attempt
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.rl.skyrl_records import read_skyrl_metrics
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from skyrlrecipe.documents import set_path

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    append_metrics,
    cache_native_compositions,
    controls,
    copy_evaluations,
    edit_receipt,
    fake_iris,
    fake_publication,
    fake_s3,
    fake_s3_checkpoint,
    replace_evaluations,
    write_published_artifact,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs, native_json
from experiments.post_training.snowball_sft import OPENTHOUGHTS_32K_4NODE
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, SELECTION_PREFIX

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def printed_status(output):
    # CliRunner merges stderr into result.output; status is on result.stdout.
    marker = output.rfind("\npreset:")
    return yaml.safe_load(output[marker + 1 if marker >= 0 else 0 :].split("\n# ", 1)[0])


def test_preset_table_rejects_duplicate_names():
    table = snowball_model.SNOWBALL
    with pytest.raises(ValueError, match="Duplicate preset name 'eval-short'"):
        replace(table, presets=(*table.presets, table.by_name["eval-short"]))


def test_all_presets_plan_with_their_declared_controls(selection, tmp_path, monkeypatch):
    launch_paths = []

    def collect_plan(paths, **_):
        # A CLI plan discards the composed config. Validate the emitted documents in one native process.
        (source,) = paths
        shutil.copyfile(source, launch_path)
        return []

    table = snowball_model.SNOWBALL
    name, version = "tests/evaluation-init", "2026.11.17"
    initializer = f"{name}@{version}"
    write_published_artifact(tmp_path, name, version, Artifact, source=str(tmp_path / name / version))
    for preset in table.presets:
        arguments = ["--preset", preset.name, "--version", preset.revision, "--plan"]
        if preset.tasks_input == "required":
            arguments += ["--tasks", SELECTION_IDENTITY]
        if preset.name == "eval-short":
            arguments += ["--init", initializer]
        launch_path = tmp_path / f"{preset.name}.yaml"
        launch_paths.append(str(launch_path))
        with monkeypatch.context() as capture:
            capture.setattr(gate, "compose_configs", collect_plan)
            result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments)), arguments)
        assert result.exit_code == 0, (preset.name, result.output, result.exception)
        if preset.name == "eval-short":
            status = printed_status(result.stdout.partition("\nStepSpec(")[0])
            assert status["inputs"]["model"].partition(":")[0] == initializer
            assert status["updates"] == 0 and status["guards"] == []
        elif preset.name in ("speed-short", "data-short"):
            invalid = [*arguments, "--init", initializer]
            result = CliRunner().invoke(snowball.command(table, argv=tuple(invalid)), invalid)
            assert result.exit_code == 4 and "does not accept --init" in result.output, result.output

    gate.compose_configs(launch_paths, validate_driver=True)


def test_plan_rejects_gated_preset_without_control_pin(selection):
    table = snowball_model.SNOWBALL
    preset = next(preset for preset in table.presets if preset.gated)
    arguments = ["--preset", preset.name, "--version", preset.revision]
    if preset.tasks_input == "required":
        arguments += ["--tasks", SELECTION_IDENTITY]
    unpinned = replace(table, control_pins={})
    result = CliRunner().invoke(snowball.command(unpinned, argv=tuple(arguments)), arguments)
    assert result.exit_code == 4 and "no control pin" in result.output, (result.output, result.exception)
    assert "--control --replicates" in result.output


@pytest.mark.parametrize(
    ("mode", "recipe_settings", "override", "reason"),
    [
        (
            "--plan",
            ("trainer.policy_mini_batch_size=2", "generator.n_samples_per_prompt=8"),
            "trainer.train_batch_size=2",
            "lcm_dp_size=16",
        ),
        (
            "--run",
            ("trainer.policy_mini_batch_size=2", "generator.n_samples_per_prompt=8"),
            "trainer.train_batch_size=2",
            "lcm_dp_size=16",
        ),
    ],
    ids=["plan-batch-size", "run-batch-size"],
)
def test_launcher_refuses_invalid_driver_settings(
    selection, preset_table, monkeypatch, capsys, mode, recipe_settings, override, reason
):
    client, _ = fake_iris(monkeypatch, "invalid-driver-config")
    if mode == "--run":
        monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    config = preset.recipe.config.with_settings(recipe_settings)
    table = preset_table(replace(preset, recipe=replace(preset.recipe, config=config)))

    exit_code = snowball.main(
        table,
        argv=["--preset", preset.name, "--version", preset.revision, mode, "--set", override],
        entry_module="experiments.post_training.snowball_rl",
    )

    output = capsys.readouterr().out
    client.submit.assert_not_called()
    assert exit_code == 4, output
    result = json.loads(output)
    assert result["verdict"] == "invalid"
    assert reason in result["reason"]


@pytest.mark.parametrize(
    ("num_workers", "static_preferences", "exit_code"),
    [(5, False, 4), (4, False, 0), (5, True, 0)],
    ids=["overbooked", "fits", "static-preferences"],
)
def test_plan_checks_pinned_workers_against_driver_cpu_request(
    selection, preset_table, monkeypatch, num_workers, static_preferences, exit_code
):
    client, _ = fake_iris(monkeypatch, "rollout-worker-capacity")
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    settings = (
        f"trajectory_runner.rollout_workers.num_workers={num_workers}",
        "trajectory_runner.rollout_workers.cpus_per_worker=8",
    )
    if static_preferences:
        settings += (
            "environment.env_class=preference_pair",
            "trainer.evaluation_runner=training",
            "generator.weight_sync_transport=broadcast",
            "trainer.rollout_buffer.max_staleness_steps=0",
        )
    layout = replace(preset.recipe.layout, host_cpu=32)
    if static_preferences:
        layout = replace(
            layout,
            workload=replace(
                layout.workload, plan=replace(layout.workload.plan, colocate_all=True, num_inference_engines=4)
            ),
        )
    config = preset.recipe.config.with_settings(settings)
    table = preset_table(replace(preset, recipe=replace(preset.recipe, layout=layout, config=config)))

    arguments = ["--preset", preset.name, "--version", preset.revision, "--plan"]
    result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments)), arguments)

    assert result.exit_code == exit_code, (result.output, result.exception)
    client.submit.assert_not_called()


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
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
    init = OPENTHOUGHTS_32K_4NODE.model()
    [sft_run] = snowball.build_experiment(train, init=init, table=snowball_model.SNOWBALL).runs
    sft_run.lower()
    assert init.step in sft_run.deps
    config = yaml.safe_load(materialized_config(sft_run, str(tmp_path)).launch_config_yaml)
    assert config["inputs"]["model"]["uri"] == f"{init.step.path(str(tmp_path))}/hf/step-9"

    dev_sft = replace(OPENTHOUGHTS_32K_4NODE, recipe=replace(OPENTHOUGHTS_32K_4NODE.recipe, version="dev"))
    with pytest.raises(click.BadParameter, match="the initializer requires an immutable artifact version"):
        snowball.build_experiment(train, init=dev_sft.model(), table=snowball_model.SNOWBALL)

    def reject_export_read(*args, **kwargs):
        raise AssertionError("planning read an unbuilt export")

    with monkeypatch.context() as planning:
        planning.setattr(skyrl_api, "discover_hf_checkpoints", reject_export_read)
        planned = snowball.build_experiment(train, init=own_model, table=snowball_model.SNOWBALL)
        assert producer in planned.runs[0].deps
        planned.runs[0].lower()
        cloud_model = replace(own_model, step=replace(producer, override_path="s3://fixture-models/computed-init"))
        assert (
            snowball.main(
                snowball_model.SNOWBALL, argv=["--preset", train.name, "--version", train.revision], init=cloud_model
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
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    changed = snowball.build_experiment(
        learning,
        settings=(
            "trainer.train_batch_size=64",
            "trainer.policy_mini_batch_size=64",
            "trainer.bf16=false",
            "trainer.algorithm.batch_invariant=true",
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
    checkpoint = StoragePath("s3://marin-test/models/checkpoint")
    export = checkpoint / "hf"
    checkpoint_dir = tmp_path / "model"
    (checkpoint_dir / "hf").mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (checkpoint_dir / "hf" / name).write_text("{}")
    fake_s3_checkpoint(monkeypatch, str(checkpoint), checkpoint_dir)
    initializer = ArtifactStep.adopt("tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint)

    revisioned = replace(snowball_model.SNOWBALL.by_name["speed-short"], revision="2026.11.12.2")
    steps = (
        changed,
        snowball.build_experiment(promoted, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_experiment(
            learning,
            init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision"),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_experiment(
            snowball_model.SNOWBALL.by_name["train-long"],
            settings=("trainer.max_steps=2", "context_budget.request_window_tokens=32768"),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_experiment(revisioned, control=True, table=snowball_model.SNOWBALL).runs[0],
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
    trainer, generator = configs[0]["skyrl"]["trainer"], configs[0]["skyrl"]["generator"]
    assert (
        trainer["bf16"],
        generator["model_dtype"],
        trainer["algorithm"]["batch_invariant"],
        generator["engine_init_kwargs"]["probe"],
    ) == (False, "float16", True, {})
    assert configs[2]["inputs"]["model"]["uri"] == str(export)
    assert configs[2]["iris"]["wandb_entity"] == "test-team"
    assert (configs[2]["inputs"]["model"]["tokenizer_uri"], configs[2]["inputs"]["model"]["tokenizer_revision"]) == (
        "tests/own-tokenizer",
        "tokenizer-revision",
    )
    callbacks = configs[3]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[3]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[3]["skyrl"]["trainer"]["max_prompt_length"] == 24576
    assert configs[4]["inputs"]["validation_data"][0]["identity"].partition(":")[0].endswith("@" + revisioned.revision)
    summary = Path(recipe.selection_artifact(learning.tasks).path()) / recipe.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_experiment(learning, table=snowball_model.SNOWBALL)


@pytest.mark.parametrize("retry_budget", [None, 0, 3], ids=["default-retry", "disabled-retry", "custom-retry"])
def test_checkpoint_cohort_coverage_controls_best_checkpoint(selection, tmp_path, retry_budget):
    table = snowball_model.SNOWBALL
    preset = table.by_name["train-long-checkpoint-eval"]
    spec = recipe.run_spec(
        preset,
        name="checkpoint-evaluation-coverage",
        model_name=table.model,
        settings=() if retry_budget is None else (f"environment.task_sessions.retry.max_retries={retry_budget}",),
    )
    step = recipe.run_step(preset, spec)
    path = tmp_path / "checkpoint-evaluation.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
    results = native_json()(
        """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from cloud.iris.launch_config import load_launch_config
from skyrl_train.callbacks.base import CallbackHandler, TrainerState, TrainerControl
from skyrl_train.callbacks.builtin import create_callbacks_from_config
from rigging.task_errors import ErrorIdentity, RetryDisposition, RetrySettings, retry_delay

path = Path(sys.argv[1])
config = load_launch_config(path)
assert not config.skyrl.generator.enable_http_endpoint
retry = RetrySettings.model_validate(dict(config.skyrl.environment.task_sessions.retry))
failure = ErrorIdentity(exception_type="TaskMachineError", retry_disposition=RetryDisposition.TRANSIENT)
budget = int(sys.argv[2])
for used in range(budget):
    assert retry_delay(failure, retry, used) is not None
assert retry_delay(failure, retry, budget) is None
state = TrainerState(global_step=0, epoch=0, total_steps=100, num_steps_per_epoch=100)
scenarios = ((100, 99, 30, 27), (100, 98, 30, 30), (100, 100, 30, 26),
             (100, 100, 29, 29), (99, 99, 31, 30))
results = []
for index, (primary_attempted, primary_scored, bfcl_attempted, bfcl_scored) in enumerate(scenarios):
    directory = path.parent / f"coverage-{index}"
    config.skyrl.trainer.ckpt_path = str(directory)
    config.skyrl.trainer.export_path = str(directory / "exports")
    metrics = {
        "eval/all/num_attempted": primary_attempted,
        "eval/all/num_scored": primary_scored,
        "eval/all/avg_verifier_score": 0.25,
    }
    for name, attempted, scored, average in (
        ("bfcl_parity", bfcl_attempted, bfcl_scored, 1.0),
        ("swebench_verified", 30, 28, 0.5),
        ("openthoughts_tblite", 30, 30, 0.0),
    ):
        prefix = f"eval/{name}/all/"
        metrics.update({prefix + "num_attempted": attempted, prefix + "num_scored": scored,
                        prefix + "avg_verifier_score": average})
    handler = CallbackHandler(create_callbacks_from_config(config.skyrl))
    asyncio.run(handler.call_event_async("on_evaluate", state, TrainerControl(), metrics=metrics,
                                         trainer=SimpleNamespace(cfg=config.skyrl)))
    record = directory / "best_checkpoint.json"
    results.append(json.loads(record.read_text()) if record.exists() else None)
print(json.dumps(results))
""",
        [str(path), str(1 if retry_budget is None else retry_budget)],
    )
    chosen, *rejected = results
    assert chosen["attempted"] == 190
    assert chosen["scored"] == 184
    assert chosen["accuracy"] == pytest.approx(65.75 / 184)
    assert rejected == [None] * 4


@pytest.mark.parametrize(
    ("kind", "record_config", "tokenizer", "revision"),
    [
        (LevanterCheckpoint, {"train_config": {"data": {"tokenizer": "legacy/tokenizer"}}}, None, None),
        (Artifact, {}, None, None),
        (
            Artifact,
            {
                "tokenizer_uri": "explicit/tokenizer",
                "tokenizer_revision": "explicit-revision",
                "train_config": {"data": {"tokenizer": "legacy/tokenizer"}},
            },
            "explicit/tokenizer",
            "explicit-revision",
        ),
    ],
    ids=("sft-export", "export", "record-override"),
)
def test_published_init_binds_export_tokenizer(
    selection, tmp_path, monkeypatch, kind, record_config, tokenizer, revision
):
    root = tmp_path / "published-checkpoint"
    export = root / "hf"
    export.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (export / name).write_text("{}")
    uri = "s3://marin-test/models/published-checkpoint"
    fake_s3_checkpoint(monkeypatch, uri, root)
    write_published_artifact(tmp_path, "tests/checkpoint", "2026.11.10", kind, source=uri, config=record_config)
    table = snowball_model.SNOWBALL
    preset = table.by_name["learning-medium"]
    model = recipe.published_model("tests/checkpoint@2026.11.10", preset=preset)
    step = snowball.build_experiment(preset, init=model, table=table).runs[0]
    launch_model = yaml.safe_load(materialized_config(step, str(tmp_path)).launch_config_yaml)["inputs"]["model"]
    assert (launch_model["uri"], launch_model["tokenizer_uri"], launch_model["tokenizer_revision"]) == (
        uri + "/hf",
        tokenizer,
        revision,
    )


@pytest.mark.parametrize("model", ("snowball", "mochi"))
def test_presets_match_revisioned_native_configs(model, tmp_path, monkeypatch, request):
    if model == "mochi":
        revisions = {
            "train-medium": {
                "2026.11.19.1": "8b5a970e",
                "2026.11.20.2": "15b0f019",
                "2026.11.20.4": "3da1da70",
                "2026.11.20.8": "5acba3d6",
                "2026.11.24.1": "98b8c2d6",
            },
            "train-long": {"2026.11.20.4": "7fa51de5", "2026.11.20.8": "58a8c866", "2026.11.24.1": "974efebe"},
            "train-long-3rack": {"2026.11.20.4": "e2c597b1", "2026.11.20.8": "33f8b528", "2026.11.24.1": "4fcf777a"},
            "train-medium-agentic": {
                "2026.11.19.1": "3f203da5",
                "2026.11.20.4": "1617b1d3",
                "2026.11.20.8": "86aebda3",
                "2026.11.23.1": "8fcc3497",
                "2026.11.24.1": "724ec7da",
            },
            "train-long-agentic": {
                "2026.11.20.6": "a3e2befe",
                "2026.11.20.7": "e1245985",
                "2026.11.20.8": "d605dbf9",
                "2026.11.23.1": "2135b5c3",
                "2026.11.24.1": "fe813cb7",
            },
            "train-long-agentic-3rack": {
                "2026.11.20.6": "b8f3bba7",
                "2026.11.20.7": "d540f763",
                "2026.11.20.8": "6c800725",
                "2026.11.23.1": "c9f51f67",
                "2026.11.24.1": "855d3d08",
            },
        }
        with monkeypatch.context() as local:
            local.delenv("MARIN_PREFIX", raising=False)
            local.delenv("MARIN_TEMP_PREFIX", raising=False)
            local.setattr(recipe, "artifact_prefix", lambda _cluster: "s3://marin-us-east-08a/marin")
            for preset in MOCHI.presets:
                assert recipe.run_prefix(preset).rstrip("/") == "s3://marin-us-east-08a/tmp/ttl=3d"
        produced = request.getfixturevalue("mochi_sft_presets")["periodic"].model()
        for preset in MOCHI.presets:
            model, prefix = preset.model, recipe.artifact_prefix(preset.recipe.layout.cluster)
            assert (model.step.name, model.step.version, model.step.path(prefix)) == (
                produced.step.name,
                produced.step.version,
                produced.step.path(prefix),
            )
            assert model.relative_path == produced.relative_path and model.tokenizer_uri is None
        configs = assert_revisioned_native_configs(
            MOCHI,
            revisions,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            mixed_tasks=request.getfixturevalue("selection"),
        )
        _assert_mochi_preset_documents(revisions, configs)
        return
    revisions = {
        "speed-short": {
            "2026.11.13.1": "96e0d088",
            "2026.11.17.1": "363225a9",
            "2026.11.19.1": "371233d9",
            "2026.11.20.1": "8f532da2",
            "2026.11.20.3": "60ba5ff4",
            "2026.11.20.8": "00119d64",
            "2026.11.24.1": "a5ba23fa",
        },
        "data-short": {
            "2026.11.13.1": "7d9a590f",
            "2026.11.17.1": "772467a8",
            "2026.11.19.1": "cb07d91c",
            "2026.11.20.1": "a379319d",
            "2026.11.20.3": "c1616f4c",
            "2026.11.20.8": "76dca2fb",
            "2026.11.24.1": "0e4173c2",
        },
        "speed-medium": {
            "2026.11.12.1": "e75349ff",
            "2026.11.17.1": "5cd852f4",
            "2026.11.19.1": "f8281b9d",
            "2026.11.20.1": "8b7e47c8",
            "2026.11.20.3": "2889e790",
            "2026.11.20.8": "ed50475d",
            "2026.11.24.1": "531fdb00",
        },
        "learning-medium": {
            "2026.11.12.1": "ba638a12",
            "2026.11.17.1": "2be6e5e2",
            "2026.11.19.1": "4aa3aed9",
            "2026.11.20.1": "67306e75",
            "2026.11.20.2": "b3044b9b",
            "2026.11.20.3": "ce8666ce",
            "2026.11.20.4": "c175bc06",
            "2026.11.20.8": "39a87700",
            "2026.11.24.1": "ab6f2e56",
        },
        "data-medium": {
            "2026.11.12.1": "ba638a12",
            "2026.11.17.1": "2be6e5e2",
            "2026.11.19.1": "4aa3aed9",
            "2026.11.20.1": "67306e75",
            "2026.11.20.2": "b3044b9b",
            "2026.11.20.3": "ce8666ce",
            "2026.11.20.4": "c175bc06",
            "2026.11.20.8": "39a87700",
            "2026.11.24.1": "ab6f2e56",
        },
        "speed-long": {
            "2026.11.11.1": "cd0a5f4e",
            "2026.11.17.1": "f29155cc",
            "2026.11.19.1": "9edc7b70",
            "2026.11.20.1": "3d1fe22d",
            "2026.11.20.3": "459f1e2c",
            "2026.11.20.8": "fafdb7f1",
            "2026.11.24.1": "db79277d",
        },
        "train-long": {
            "2026.11.11.1": "4945b701",
            "2026.11.17.1": "b411c765",
            "2026.11.19.1": "e56a62fa",
            "2026.11.20.1": "22473465",
            "2026.11.20.2": "412d0576",
            "2026.11.20.3": "40c0256e",
            "2026.11.20.4": "8917eb9e",
            "2026.11.20.8": "984f3cfb",
            "2026.11.24.1": "c8577baa",
        },
        "train-long-checkpoint-eval": {
            "2026.11.23.1": "6f49b36f",
            "2026.11.24.1": "9885d84b",
            "2026.11.24.2": "af1e1b7b",
        },
        "train-medium-agentic": {
            "2026.11.19.1": "756f58ef",
            "2026.11.20.1": "b557a4f0",
            "2026.11.20.4": "f76d6b51",
            "2026.11.20.5": "aa36f9de",
            "2026.11.20.8": "4dba0215",
            "2026.11.23.1": "f64150c5",
            "2026.11.24.1": "43c5c764",
        },
        "train-long-agentic": {
            "2026.11.20.4": "7123c960",
            "2026.11.20.5": "0043357f",
            "2026.11.20.8": "e4d758ba",
            "2026.11.23.1": "a7e2e075",
            "2026.11.24.1": "9b86afd3",
        },
        "train-medium-tools": {"2026.11.20.1": "6a97fd33", "2026.11.20.9": "1bc97869", "2026.11.24.1": "0bef6216"},
        "eval-short": {
            "2026.11.19.1": "8144570a",
            "2026.11.20.1": "fad362ca",
            "2026.11.20.3": "8868887c",
            "2026.11.20.8": "4b439968",
            "2026.11.24.1": "f4b7c40b",
        },
    }
    configs = assert_revisioned_native_configs(
        snowball_model.SNOWBALL,
        revisions,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        mixed_tasks=request.getfixturevalue("selection"),
    )
    for name, config in zip(revisions, configs, strict=True):
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        if snowball_model.SNOWBALL.by_name[name].batch_mix is not None:
            continue
        assert config["iris"]["allocation"]["num_nodes"] * config["iris"]["allocation"]["gpus_per_node"] <= 40
        generating = name in ("speed-short", "data-short", "eval-short")
        assert (
            config["runtime"]["training_type"],
            trainer["rollout_buffer"]["max_staleness_steps"],
            trainer["algorithm"]["off_policy_correction"],
            trainer["rollout_buffer"]["batch_policy"],
        ) == ((None, 0, "none", "full_batch") if generating else ("async", 2, "tis", "full_batch"))
        if generating:
            assert config["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
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
                    == f"{SELECTION_PREFIX}/{snowball_model.SNOWBALL.by_name[name].held_out_tasks}"
                )
                assert source["relative_path"] == "full_validation.parquet"
            sampling = profile["sampling_params"] if profile else generator["eval_sampling_params"]
            assert (
                sampling["temperature"],
                sampling["ignore_eos"],
                sampling["max_generate_length"],
                profile["n_samples_per_prompt"] if profile else generator["eval_n_samples_per_prompt"],
            ) == (0, False, 8192, 1)
        if name in ("learning-medium", "data-medium", "train-long", "train-long-checkpoint-eval"):
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_before_train"] is True
            kinds = [row["type"] for row in trainer["callbacks"]]
            assert kinds.index("evaluation") < kinds.index("best_checkpoint") < kinds.index("hf_model_save")
        if name == "train-long":
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_on_train_end"] is False
            historical = deepcopy(config)
            historical["skyrl"]["data"].update(task_cache_dir="~/.cache/marinskyrl/tasks", terminal_bench_data=[])
            historical["skyrl"]["environment"]["task_sessions"].pop("agent_harnesses")
            historical["skyrl"]["trajectory_runner"].pop("max_concurrent_training_machine_tasks")
            assert gate.comparison_config(historical) == gate.comparison_config(config)


def _assert_mochi_preset_documents(revisions, configs):
    documents = dict(zip(revisions, configs, strict=True))
    for preset in MOCHI.presets:
        document = documents[preset.name]
        assert not document["run"]["export_hf"]
        assert not any(
            callback["type"] in {"checkpoint", "best_checkpoint", "hf_model_save"}
            for callback in document["skyrl"]["trainer"]["callbacks"]
        )
        if preset.evaluations:
            (validation,) = document["inputs"]["validation_data"]
            assert validation["relative_path"] == "validation.parquet"

    docs = (recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
    fragment = docs.split("## Mochi inference on GB200", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
    served = gate._config_values(documents["train-long-3rack"]["skyrl"])
    for path, value in gate._config_values(yaml.safe_load(fragment)).items():
        assert served[path] == value, path


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    data = snowball_model.SNOWBALL.by_name["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_experiment(
            replace(learning, accepts_settings=False),
            settings=("trainer.policy.optimizer_config.lr=2e-6",),
            table=snowball_model.SNOWBALL,
        )

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_experiment(preset, settings=settings, **kwargs, table=snowball_model.SNOWBALL).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    assert name(learning, "generator.engine_init_kwargs.probe={}") != name()
    repeated = snowball.build_experiment(learning, replicates=3, table=snowball_model.SNOWBALL)
    assert [
        step.name for step in snowball.build_experiment(learning, replicates=2, table=snowball_model.SNOWBALL).runs
    ] == [step.name for step in repeated.runs[:2]]
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
        held_out_tasks=f"{task_set_name}@2026.11.05",
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
    controls(recorded_runs, "learning-medium")
    first = snowball.build_experiment(learning, requirements=("eval_lite/mean>=0",), table=snowball_model.SNOWBALL)
    second = snowball.build_experiment(learning, requirements=("eval_lite/mean>=0.1",), table=snowball_model.SNOWBALL)
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
    with pytest.raises(click.BadParameter, match="different development panel"):
        snowball.build_experiment(data, tasks=selection, table=snowball_model.SNOWBALL)
    summary["recipe"]["heldout"] = heldout
    summary_path.write_text(json.dumps(summary))
    for current in (data, snowball_model.SNOWBALL.by_name["data-medium"]):
        selected = replace(current, held_out_tasks=f"{task_set_name}@2026.11.05")
        with pytest.raises(click.BadParameter, match="full held-out selection"):
            snowball.build_experiment(selected, tasks=selection, table=snowball_model.SNOWBALL)
        summary["recipe"]["exclusions"] = [
            {"uri": str(new_task_set / FULL_VALIDATION_FILENAME), "version": "2026.11.05"}
        ]
        summary_path.write_text(json.dumps(summary))
        assert snowball.build_experiment(selected, tasks=selection, table=snowball_model.SNOWBALL).gate is not None
        summary["recipe"]["exclusions"] = original_exclusions
        summary_path.write_text(json.dumps(summary))
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (recipe.selection_artifact(snowball_model.SNOWBALL.by_name["learning-medium"].tasks), selection):
        with pytest.raises(click.BadParameter, match="full held-out selection"):
            snowball.build_experiment(data, tasks=tasks, table=snowball_model.SNOWBALL)
    with pytest.raises(ValueError, match="held-out evaluations require a held-out task selection"):
        replace(learning, held_out_tasks=None)
    training_only = replace(snowball_model.SNOWBALL.by_name["speed-medium"], tasks_input="required")
    assert snowball.build_experiment(training_only, tasks=selection, table=snowball_model.SNOWBALL).runs


@pytest.mark.parametrize(
    "case,expected",
    [
        ("fail", 1),
        ("missing", 3),
        ("native-failed", 2),
        ("child", 0),
        ("changed-source", 0),
        ("stale-control", 4),
        ("candidate-compose", 4),
        ("malformed-control", 4),
        ("mixed-native-failed", 2),
        ("control-missing-receipt", 3),
        ("partial-retry", 0),
        ("data-short", 0),
        ("unexpected", 3),
    ],
)
def test_submission_exit_codes_and_publication_retries(
    recorded_runs, selection, tmp_path, monkeypatch, preset_table, case, expected
):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    publication_case = case in {
        "changed-source",
        "child",
        "mixed-native-failed",
        "control-missing-receipt",
        "partial-retry",
        "data-short",
    }
    # The data-short case rewrites its validation data before launch.
    if publication_case and case != "data-short":
        cache_native_compositions(monkeypatch)
    preset_name = "data-short" if case == "data-short" else "learning-medium" if publication_case else "speed-medium"
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    if case == "child":
        model = "s3://marin-test/models/child-model"
        preset = replace(
            preset,
            model=replace(
                preset.model,
                step=ArtifactStep.adopt(
                    preset.model.step.name, preset.model.step.version, str(model), kind=preset.model.step.artifact_type
                ),
            ),
        )
    preset_table(preset)
    if case == "child":
        fake_s3(monkeypatch, "marin-test/models/python-init", "marin-test/models/child-model")
    roots = controls(recorded_runs, preset.name) if preset.gated else []
    replicas = 2 if case in {"partial-retry", "mixed-native-failed", "control-missing-receipt"} else 1
    control = case == "control-missing-receipt"
    built = snowball.build_experiment(
        preset,
        table=snowball_model.SNOWBALL,
        control=control,
        replicates=replicas,
        tasks=selection if case == "data-short" else None,
    )
    sources = (
        []
        if case == "candidate-compose"
        else [
            recorded_runs(
                preset.name,
                training.name,
                speed=150 if case == "fail" else 101,
                score=0.5,
                fingerprint=training.fingerprint(),
                inputs=built.inputs,
                attempt_id=f"native-{index}",
            )
            for index, training in enumerate(built.runs)
        ]
    )
    source = sources[0] if sources else None
    for index, root in enumerate(sources):
        native = read_attempt(str(root / "exports"), f"native-{index}")
        write_attempt(str(root / "exports"), AttemptRecord(run_id=native.run_id, attempt_id=native.attempt_id, issue=73))
    if case == "data-short":
        receipt = json.loads((source / "terminal.json").read_text())
        descriptor = receipt["config"]["inputs"]["validation_data"][0]
        data_path = Path(descriptor["uri"]) / descriptor["relative_path"]
        rows = pq.read_table(data_path).to_pylist()
        for row in rows:
            task = json.loads(row["task_spec"])
            task["id"] = "candidate-" + task["id"]
            row["task_spec"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(rows), data_path)
        evaluations = read_evaluations(str(source / "exports"), attempt_id="native-0", evaluation="eval/")
        replace_evaluations(
            source, [row.model_copy(update={"task_id": "candidate-" + row.task_id}) for row in evaluations]
        )
    mutation = {
        "native-failed": (source, "result.state", "failed"),
        "malformed-control": (roots[0], "config.skyrl.trainer.seed", None),
        "stale-control": (roots[-1], "config.skyrl.trainer.policy.optimizer_config.lr", 5e-06),
    }.get(case)
    if mutation is not None:
        root, field, value = mutation
        with edit_receipt(root) as receipt:
            set_path(receipt, field, value)
        if case == "native-failed":
            (source / "terminal.json").unlink()
    pending = {}
    if publication_case:
        for root in sources:
            for filename in ("terminal.json", "resolved-launch.yaml"):
                path = root / filename
                pending[path] = path.read_bytes()
                path.unlink()
    phase = {"retry": False}
    if case == "unexpected":
        path = Path(recipe.selection_artifact(preset.tasks).path()) / recipe.SUMMARY_FILENAME
        summary = json.loads(path.read_text())
        summary.pop("grader_kind_counts")
        path.write_text(json.dumps(summary))

    def assert_submission_link():
        output = click.get_binary_stream("stdout").getvalue().decode()
        urls = [
            line.split("run: ", 1)[1].strip() for line in output.splitlines() if "run: https://grafana.oa.dev/" in line
        ]
        query = next(parse_qs(urlparse(url).query) for url in urls if parse_qs(urlparse(url).query)["to"] == ["now"])
        assert query["var-run"] == [built.runs[0].name + "-" + preset.revision]
        assert query["var-cluster"] == [preset.recipe.layout.cluster]
        assert query["var-training_type"] == ["sync" if preset_name == "data-short" else "async"]
        assert int(query["from"][0]) > 0

    def finish(*args, **kwargs):
        if case != "child":
            assert_submission_link()
        for path, data in pending.items():
            path.write_bytes(data)
        if case == "mixed-native-failed":
            receipt = json.loads((source / "terminal.json").read_text())
            if not phase["retry"]:
                receipt["result"]["state"] = "failed"
                write_receipt(source, receipt)
                (source / "terminal.json").unlink()
                return SimpleNamespace(state=JobState.FAILED)
            metrics = read_skyrl_metrics(str(source), attempt_id="native-0")
            receipt["config"]["run"]["attempt_id"] = "native-retry"
            write_receipt(source, receipt)
            write_attempt(
                str(source / "exports"),
                AttemptRecord(run_id=receipt["config"]["run"]["id"], attempt_id="native-retry", issue=74),
            )
            copy_evaluations(source, "native-0", "native-retry")
            for row in metrics.values():
                row["attempt_id"] = "native-retry"
            append_metrics(source, metrics)
        if control and (not phase["retry"]):
            (source / "terminal.json").unlink()
            (source / "resolved-launch.yaml").unlink()
            return SimpleNamespace(state=JobState.FAILED)
        if case not in {"missing", "native-failed"} and built.gate is not None:
            run(gate.completed_gate(built.gate))
        if case == "changed-source":
            monkeypatch.setattr(
                Provenance,
                "capture",
                lambda *args, **kwargs: Provenance(
                    "tree-edited-before-publication", "commit-one", True, "branch", "test"
                ),
            )
        return SimpleNamespace(state=JobState.FAILED if case in {"native-failed"} else JobState.SUCCEEDED)

    client, job = fake_iris(monkeypatch, "dashboard-coordinator")
    client.job.return_value.wait.side_effect = finish
    if case in {"malformed-control", "stale-control", "candidate-compose"}:
        client.submit.side_effect = AssertionError("invalid stored inputs reached the coordinator RPC")
    published = []
    unavailable = {"value": case in {"partial-retry"}}

    def capture_publication(record):
        if unavailable["value"] and record.attempt_id == "native-1":
            raise RuntimeError("Credential unavailable; run iris login")
        published.append(record.model_dump(mode="json"))

    fake_publication(monkeypatch, capture_publication, "https://progress.test")
    if case == "child":
        finish()
        monkeypatch.setattr(
            skyrl_api,
            "_run_launcher",
            lambda *args, **kwargs: pytest.fail("Cached coordinator run invoked native execution"),
        )
    arguments = ["--preset", preset.name, "--version", preset.revision, "--run"]
    if case != "data-short":
        arguments.extend(["--issue", "73"])
    if case != "unexpected":
        arguments.append("--wait")
    if "candidate-compose" in case:
        arguments.extend(["--set", "trainer.algorithm.policy_loss_type=gspo"])
    if control:
        arguments.append("--control")
    if case == "data-short":
        arguments.extend(["--tasks", SELECTION_IDENTITY])
    if replicas == 2:
        arguments.extend(["--replicates", "2"])
    selected = snowball.command(
        snowball_model.SNOWBALL, argv=tuple(arguments), entry_module="experiments.post_training.snowball_rl"
    )
    invoked = arguments
    if case == "child":

        def reenter(**kwargs):
            with monkeypatch.context() as child:
                for key, value in kwargs["environment"].env_vars.items():
                    child.setenv(key, value)
                child.setenv("IRIS_TASK_ID", "/test-user/coordinator/0")
                child.setattr(client, "submit", lambda **kwargs: pytest.fail("Coordinator submitted another job"))
                child.setattr(
                    "experiments.post_training.rl_publication.publish_result",
                    lambda *args: pytest.fail("Coordinator contacted Marina"),
                )
                info = JobInfo(task_id=JobName.from_wire("/test-user/coordinator/0"))
                with iris_ctx_scope(IrisContext.from_job_info(info)):
                    completed = CliRunner().invoke(selected, invoked)
                assert completed.exit_code == 0, (completed.output, completed.exception)
            set_job_info(None)
            return SimpleNamespace(job_id=job)

        client.submit.side_effect = reenter
    set_job_info(None)
    result = CliRunner().invoke(selected, invoked)
    assert result.exit_code == expected, (result.output, result.exception)
    if case.startswith("malformed-"):
        assert "malformed recorded configuration" in result.output and "seed" in result.output
    if case.startswith("stale-control"):
        assert "skyrl.trainer.policy.optimizer_config.lr" in result.output
    if "candidate-compose" in case:
        assert "native configuration cannot be composed:" in result.output and "gspo" in result.output.lower()
    if case == "unexpected":
        assert "KeyError" in result.output and "grader_kind_counts" in result.output
    if case == "child":
        assert len(published) == 1
        report = gate.completed_gate(built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"
        return
    if not publication_case:
        return
    snapshot = printed_status(result.stdout)
    if case == "data-short":
        assert "publication" not in snapshot and published == []
        return
    paths = [[f"{training.name}-{preset.revision}", f"native-{index}"] for index, training in enumerate(built.runs)]
    if unavailable["value"]:
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Credential unavailable; run iris login"],
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert [row["attempt_id"] for row in published] == ["native-0"]
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Gate report missing for this attempt; repeat the original command with --wait"],
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert published[0]["status"] == "failed" and published[0]["gate_result"] is None
    if expected == 0 and (not unavailable["value"]):
        for row in snapshot["runs"]:
            assert parse_qs(urlsplit(row["rl_progress_url"]).fragment)["attempt"] == [row["attempt_id"]]
    if case == "changed-source":
        return
    unavailable["value"] = False
    phase["retry"] = True
    arguments[arguments.index("--issue") + 1] = "74"
    previously_published = len(published)
    repeated = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL, argv=tuple(arguments), entry_module="experiments.post_training.snowball_rl"
        ),
        arguments,
    )
    assert repeated.exit_code == (0 if case in {"mixed-native-failed", "control-missing-receipt"} else expected), (
        repeated.output,
        repeated.exception,
    )
    recovered = printed_status(repeated.stdout)
    assert recovered["publication"]["unserved_records"] == [] and recovered["publication"]["errors"] == []
    assert {row["attempt_id"]: row["issue"] for row in recovered["runs"]} == (
        {"native-retry": 74, "native-1": 73} if case == "mixed-native-failed" else {attempt: 73 for _, attempt in paths}
    )
    if case == "mixed-native-failed":
        retry = next(row for row in published[previously_published:] if row["attempt_id"] == "native-retry")
        assert retry["status"] == "succeeded" and retry["gate_result"]["verdict"] == "pass" and (retry["issue"] == 74)
    else:
        cached = next(row for row in published[previously_published:] if row["attempt_id"] == "native-0")
        assert cached["issue"] == 73


@pytest.mark.parametrize("fault", ["untracked", "dirty"])
def test_control_refuses_actual_untracked_source_before_execution(selection, tmp_path, monkeypatch, fault):
    root = tmp_path / "git-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "experiments").mkdir()
    (root / "experiments/untracked.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(recipe, "REPO_ROOT", root)
    if fault == "dirty":
        subprocess.run(["git", "-C", str(root), "add", "experiments/untracked.py"], check=True)
        monkeypatch.setattr(
            Provenance, "capture", lambda *args, **kwargs: Provenance("tree", "commit", True, "branch", "test")
        )
    with pytest.raises(click.BadParameter, match="stage source" if fault == "untracked" else "clean checkout"):
        snowball.build_experiment(
            snowball_model.SNOWBALL.by_name["speed-medium"], control=True, table=snowball_model.SNOWBALL
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--wait",),
        ("python-init-conflict",),
        ("python-init-compare",),
        ("--set", "trainer.dump_eval_results=false"),
        ("--set", "trainer.hf_hub_repo_id=org/model"),
        ("--set", 'trainer={"algorithm":{"off_policy_correction":"none"}}'),
        ("--set", "trainer.policy.optimizer_config.lr=wrong"),
        ("--require", "typo>=0"),
    ],
)
def test_refusals_use_exit_four(selection, arguments, capsys, tmp_path, request):
    if arguments == ("python-init-compare",):
        assert (
            snowball.main(snowball_model.SNOWBALL, argv=["compare"], init=snowball_model.SNOWBALL.presets[0].model) == 4
        )
        assert "Python init" in capsys.readouterr().err
        return
    initializer = snowball_model.SNOWBALL.presets[0].model if arguments == ("python-init-conflict",) else None
    preset_name = "speed-medium"
    if initializer is not None:
        preset_name = "train-long"
        name, version = "tests/valid-published-init", "2026.11.17"
        root = tmp_path / name / version
        write_published_artifact(tmp_path, name, version, Artifact, source=str(root))
        arguments = ("--init", f"{name}@{version}")
    if arguments == ("--require", "typo>=0"):
        record = request.getfixturevalue("recorded_runs")
        table = snowball_model.SNOWBALL
        pin = gate.control_pin(table.by_name[preset_name], table=table)
        record(preset_name, gate.control_name(pin.preset, pin.commit, 0, table=table))
    result = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            init=initializer,
            entry_module="experiments.post_training.snowball_rl",
        ),
        ["--preset", preset_name, "--version", snowball_model.SNOWBALL.by_name[preset_name].revision, *arguments],
    )
    assert result.exit_code == 4, (result.output, result.exception)
    if initializer is not None:
        assert "Python init and --init" in result.output
    if arguments == ("--require", "typo>=0"):
        assert "does not compute" in result.output
    if "trainer.dump_eval_results=false" in arguments:
        assert "owned" in result.output
    if "trainer.hf_hub_repo_id=org/model" in arguments:
        assert "use the preset's declared owned keys" in result.output


def test_train_long_reenters_the_model_module_and_reports_checkpoint_evaluations(
    recorded_runs, preset_table, tmp_path, monkeypatch
):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(
        preset,
        checkpoints="best",
        export="best",
        recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_disk="3TB")),
    )
    table = preset_table(preset)
    initializer = ArtifactHfModel(
        ArtifactStep.adopt("tests/python-init", "2026.11.17", "s3://marin-test/models/python-init", kind=Artifact),
        "tests/python-tokenizer",
        "python-tokenizer-revision",
        relative_path="",
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.setenv("HF_TEST_KEY", "test-initializer-credential")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    fake_s3(monkeypatch, "marin-test/models/python-init")
    built = snowball.build_experiment(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=12",)
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    _json(source / "exports/best_checkpoint.json", {"step": 10})
    staged = source.with_name("before-native-execution")
    source.rename(staged)
    native_calls = []

    def native(command, *, env_vars=None):
        inspection = native_json()(
            "import json,sys\nfrom cloud.iris import launch\n"
            "args=launch.create_parser().parse_args(sys.argv[1:])\n"
            "config=launch.load_launch_config(args.config)\n"
            "print(json.dumps({'forward_env':args.forward_env,'disk':config.iris.allocation.disk,"
            "'tokenizer':config.inputs.model.tokenizer_uri,'model':config.run.model,"
            "'attempt':config.run.attempt_id,'export':config.artifacts.export_root}))\n",
            command[command.index("marinskyrl") + 1 :],
        )
        assert (inspection["forward_env"], inspection["disk"], inspection["tokenizer"]) == (
            ["HF_TEST_KEY"],
            "3TB",
            "tests/python-tokenizer",
        )
        assert read_attempt(inspection["export"], inspection["attempt"]) is None
        native_calls.append(command)
        shutil.copytree(staged, source, dirs_exist_ok=True)
        stored = json.loads((source / "terminal.json").read_text())
        stored["config"]["run"]["model"] = inspection["model"]
        write_receipt(source, stored)
        write_attempt(
            inspection["export"],
            AttemptRecord(
                run_id=stored["config"]["run"]["id"],
                attempt_id=stored["config"]["run"]["attempt_id"],
            ),
        )
        response = stored["result"]
        response["model"] = dict(
            policy_export_uri=str(source / "policy"),
            global_step=12,
            tokenizer_uri="tests/python-tokenizer",
            tokenizer_revision="python-tokenizer-revision",
            checkpoint_root=str(source / "checkpoints"),
            terminal_manifest_uri=str(source / "terminal.json"),
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(skyrl_api, "_run_launcher", native)
    published = []

    fake_publication(monkeypatch, published.append, "http://progress.test")
    arguments = [
        "--preset",
        preset.name,
        "--version",
        preset.revision,
        "--run",
        "--wait",
        "--marina-url",
        "http://127.0.0.1:55777",
        "--max-concurrent",
        "3",
        "--set",
        "trainer.max_steps=12",
    ]
    client, job = fake_iris(monkeypatch, "train-coordinator")
    mode, coordinator_module = "valid", "first_model_entry"

    def submit(**kwargs):
        assert kwargs["entrypoint"].command == ["python", "-m", coordinator_module, *arguments]
        environment = kwargs["environment"].env_vars
        if mode != "overlapping-environment-mismatch":
            assert environment["HF_TEST_KEY"] == "test-initializer-credential"
        assert "UNRELATED_SECRET" not in environment
        if mode == "failed-native":
            return SimpleNamespace(job_id=job)
        with monkeypatch.context() as child:
            for key, value in environment.items():
                child.setenv(key, value)
            child.setenv("IRIS_TASK_ID", "/test-user/train-coordinator/0")
            child.setattr(sys, "argv", [coordinator_module, *arguments])
            child.setattr(
                "experiments.post_training.rl_publication.publish_result",
                lambda *args: pytest.fail("Coordinator contacted Marina"),
            )
            info = JobInfo(task_id=JobName.from_wire("/test-user/train-coordinator/0"))
            with iris_ctx_scope(IrisContext.from_job_info(info)), pytest.raises(SystemExit) as completed:
                runpy.run_module(coordinator_module, run_name="__main__", alter_sys=True)
            assert completed.value.code == {"valid": 0, "failed-following": 3}.get(mode, 4)
        set_job_info(None)
        return SimpleNamespace(job_id=job)

    client.submit.side_effect = submit
    client.job.return_value.wait.side_effect = lambda *args, **kwargs: SimpleNamespace(
        state=JobState.SUCCEEDED if mode == "valid" else JobState.FAILED
    )

    policy_uri = "s3://marin-test/models/python-policy"
    policy_path = tmp_path / "python-policy"
    raw_load = SkyRLRun.raw_load
    monkeypatch.setattr(SkyRLRun, "raw_load", lambda path: raw_load(str(policy_path) if path == policy_uri else path))

    def module(
        name, *, entry_module=None, use_init=True, forward_env=("HF_TEST_KEY",), policy_forward=None, following="None"
    ):
        path = tmp_path / (name + ".py")
        dependency = ""
        if policy_forward is not None:
            dependency = (
                f"producer=rl.build_experiment(snowball_rl.SNOWBALL.by_name['train-long'],table=snowball_rl.SNOWBALL,"
                f"settings=('trainer.max_steps=2',),forward_env={policy_forward!r}).runs[0]\n"
                f"result=SkyRLRun.raw_load({str(source)!r}).model_copy(update={{'hf_model_uri':'s3://marin-test/models/python-init'}})\n"
                f"directory=Path({str(policy_path)!r});directory.mkdir(parents=True,exist_ok=True)\n"
                "record=ArtifactRecord(name=producer.name,version=producer.version,fingerprint=producer.fingerprint(),"
                "result_type=result_type_name(SkyRLRun),result=result.result_payload())\n"
                "(directory/RECORD_FILENAME).write_text(record.model_dump_json())\n"
                f"init=SkyRLPolicyModel(replace(producer,override_path={policy_uri!r}))\n"
            )
        path.write_text(
            "from dataclasses import replace\nfrom pathlib import Path\n"
            "from marin.execution.lazy import ArtifactStep, apply\n"
            "from marin.execution.artifact import Artifact,ArtifactRecord,RECORD_FILENAME,result_type_name\n"
            "from marin.rl.skyrl import ArtifactHfModel,SkyRLRun,SkyRLPolicyModel\n"
            "from experiments.post_training import rl, snowball_rl\n"
            "step=ArtifactStep.adopt('tests/python-init','2026.11.17','s3://marin-test/models/python-init',kind=Artifact)\n"
            "init=ArtifactHfModel(step,'tests/python-tokenizer','python-tokenizer-revision',relative_path='')\n"
            "def fail_following():\n    raise RuntimeError('following stage failed')\n"
            + dependency
            + f"following={following}\n"
            "table=snowball_rl.SNOWBALL\n"
            "if following is not None:\n"
            "    table=replace(table,presets=tuple(replace(preset,then=following) for preset in table.presets))\n"
            + f"raise SystemExit(rl.main(table,init={'init' if use_init else 'None'},"
            f"forward_env={forward_env!r},entry_module={entry_module!r}))\n"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        monkeypatch.setitem(sys.modules, name, importlib.util.module_from_spec(spec))

    @click.command()
    def entry():
        monkeypatch.setattr(sys, "argv", ["first_model_entry", *arguments])
        with pytest.raises(SystemExit) as completed:
            runpy.run_module("first_model_entry", run_name="__main__", alter_sys=True)
        raise click.exceptions.Exit(completed.value.code)

    def invoke_with_preflight_stub(command, args=()):
        # Native preflight has its own tests. Use the real composer to create
        # recorded fixtures and inspect the native launch in this test.
        with monkeypatch.context() as preflight:
            preflight.setattr(
                gate, "compose_configs", lambda paths, **_: [{"runtime": {"training_type": "async"}} for _ in paths]
            )
            return CliRunner().invoke(command, args)

    for override in (None, "second_model_entry"):
        coordinator_module = override or "first_model_entry"
        module("first_model_entry", entry_module=override)
        if override is not None:
            module(override)
        result = invoke_with_preflight_stub(entry)
        assert result.exit_code == 0, (result.output, result.exception)
        assert "best_checkpoint_step: 10" in result.output and f"{preset.evaluations[0].name}@12/scored" in result.output
        assert "http://127.0.0.1:55777/" in result.output and "test-initializer-credential" not in result.output
    assert len(native_calls) == 1
    assert published[-1].optimizer_steps == published[-1].planned_updates == 12
    assert (published[-1].model, published[-1].issue) == ("snowball", None)

    for mode in (
        "initializer-mismatch",
        "environment-mismatch",
        "overlapping-environment-mismatch",
        "dependency-environment-mismatch",
    ):
        coordinator_module = "second_model_entry"
        module(
            "first_model_entry",
            entry_module=coordinator_module,
            forward_env=("WANDB_API_KEY",) if mode == "overlapping-environment-mismatch" else ("HF_TEST_KEY",),
            policy_forward=("WANDB_API_KEY",) if mode == "dependency-environment-mismatch" else None,
        )
        module(
            coordinator_module,
            use_init=mode != "initializer-mismatch",
            forward_env=() if mode in {"environment-mismatch", "overlapping-environment-mismatch"} else ("HF_TEST_KEY",),
            policy_forward=() if mode == "dependency-environment-mismatch" else None,
        )
        result = invoke_with_preflight_stub(entry)
        assert result.exit_code == 3 and "coordinator rebuilt a different RL graph" in result.output, (
            result.output,
            result.exception,
        )
        assert "test-initializer-credential" not in result.output and len(native_calls) == 1

    mode, coordinator_module = "failed-following", "first_model_entry"
    module(coordinator_module, following="lambda _: apply('tests/following', fail_following, version='2026.11.20.1')")
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and "post_run_error:" in result.output, (result.output, result.exception)
    assert "RL has a committed result" in result.output

    evaluations = read_evaluations(str(source / "exports"), attempt_id="recorded-attempt", evaluation="eval/")
    replace_evaluations(
        source,
        [
            row.model_copy(
                update={
                    "answers": [answer.model_copy(update={"error_treatment": "passthrough"}) for answer in row.answers]
                }
            )
            for row in evaluations
            if row.step != 10
        ],
    )
    (source / "exports/best_checkpoint.json").unlink()
    with edit_receipt(source) as receipt:
        receipt["result"]["model"] = None
    step = built.runs[0]
    write_skyrl_run_record(source, step.name, step.version, fingerprint=step.fingerprint(), global_step=None)
    planning_args = ["--preset", preset.name, "--version", preset.revision, "--set", "trainer.max_steps=12"]
    result = invoke_with_preflight_stub(
        snowball.command(table, argv=tuple(planning_args), init=initializer, forward_env=("HF_TEST_KEY",)),
        planning_args,
    )
    assert result.exit_code == 1, (result.output, result.exception)
    [status] = printed_status(result.stdout)["runs"]
    assert status["best_checkpoint_step"] == "none eligible"
    facts = {fact["name"]: fact for fact in status["evaluation_invariants"]}
    assert not facts[f"{preset.evaluations[0].name}/step-10/answers_complete"]["holds"]
    assert not facts[f"{preset.evaluations[0].name}/task_coverage"]["holds"]

    mode, coordinator_module = "valid", "first_model_entry"
    module(coordinator_module)
    resumed = invoke_with_preflight_stub(entry)
    assert resumed.exit_code == 1 and len(native_calls) == 1, (resumed.output, resumed.exception)
    assert "best_checkpoint_step: none eligible" in resumed.output

    replace_evaluations(source, evaluations)

    live_preset = replace(preset, checkpoints="none", checkpoint_every=None, export="none")
    live = snowball.build_experiment(
        live_preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=12",)
    )
    assert live.runs[0].name != built.runs[0].name
    with monkeypatch.context() as live_record:
        live_record.setattr(snowball_model, "SNOWBALL", replace(table, presets=(live_preset,)))
        recorded_runs(preset.name, live.runs[0].name, fingerprint=live.runs[0].fingerprint(), inputs=live.inputs)
    live_args = [*planning_args, "--checkpoints", "none", "--checkpoint-every", "0", "--export", "none"]
    result = invoke_with_preflight_stub(
        snowball.command(table, argv=tuple(live_args), init=initializer, forward_env=("HF_TEST_KEY",)), live_args
    )
    assert result.exit_code == 1, (result.output, result.exception)
    facts = {
        fact["name"]: fact for row in printed_status(result.stdout)["runs"] for fact in row["evaluation_invariants"]
    }
    assert not facts[f"{preset.evaluations[0].name}/step-10/answers_complete"]["holds"]

    mode, coordinator_module = "failed-native", "first_model_entry"
    module("first_model_entry")
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "failed"
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)
