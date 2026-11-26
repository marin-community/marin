# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse, urlsplit

import click
import draccus
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import (
    AttemptRecord,
    LaunchResult,
    RunIndexIdentity,
    read_attempt,
    read_evaluations,
    write_attempt,
)
from finestore.rl.strata import TaskStrata
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.experiment.cli import graph_handles
from marin.rl import skyrl as skyrl_api
from marin.rl.cli import RUN_INDEX_ENV, _coordinator_request
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from skyrlrecipe import RecipePatch, declared, sections
from skyrlrecipe.documents import merge_mappings, set_path
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.mochi_assets import (
    MOCHI_STEP_400407_EXPORT,
    MOCHI_TOKENIZER_REVISION,
    MOCHI_TOKENIZER_URI,
)
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_tests import model_entry_fixture
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    append_metrics,
    cohort_from_fixtures,
    controls,
    copy_evaluations,
    edit_receipt,
    fake_iris,
    fake_publication,
    fake_s3,
    fake_s3_checkpoint,
    printed_status,
    replace_evaluations,
    write_published_artifact,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import (
    native_json,
    read_skyrl_metrics,
    run_native_test,
    selection_summary,
)
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE, OPENTHOUGHTS_32K_4NODE
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    SUMMARY_FILENAME,
    TASK_SCHEMA,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    TrainingHoldout,
    curriculum_holdout_step,
)
from experiments.sft.launcher import PreparedData
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]
SFT_ARM_TEST_PREFIX = "s3://marin-test/llom"


def test_preset_table_rejects_duplicate_names():
    table = snowball_model.SNOWBALL
    with pytest.raises(ValueError, match="Duplicate preset name 'eval-short'"):
        replace(table, presets=(*table.presets, table.by_name["eval-short"]))


def test_snowball_presets_plan_native_launches(selection, tmp_path, monkeypatch):
    launch_paths = {}

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
        arguments = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION]
        if preset.tasks_input == "required":
            arguments += ["--tasks", SELECTION_IDENTITY]
        if preset.name == "eval-short":
            arguments += ["--init", initializer]
        launch_path = tmp_path / f"{preset.name}.yaml"
        launch_paths[preset.name] = str(launch_path)
        with monkeypatch.context() as capture:
            capture.setattr(gate, "compose_configs", collect_plan)
            result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments)), arguments)
        assert result.exit_code == 0, (preset.name, result.output, result.exception)
        status = printed_status(result.stdout.partition("\nStepSpec(")[0])
        if preset.name == "eval-short":
            assert status["inputs"]["model"].partition(":")[0] == initializer
            assert status["updates"] == 0 and status["guards"] == []
        elif preset.name in ("speed-short", "data-short"):
            invalid = [*arguments, "--init", initializer]
            result = CliRunner().invoke(snowball.command(table, argv=tuple(invalid)), invalid)
            assert result.exit_code == 4 and "does not accept --init" in result.output, result.output

    composed = gate.compose_configs(list(launch_paths.values()), validate_driver=True)
    configs = dict(zip(launch_paths, composed, strict=True))
    for name in ("speed-short", "data-short", "eval-short"):
        assert configs[name]["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"

    # Published train-long controls still contain defaults removed by the TaskSpec migration.
    planned = configs["train-long"]
    trainer = planned["skyrl"]["trainer"]
    assert trainer["use_sample_packing"]
    for role in ("policy", "ref"):
        geometry = trainer[role]["megatron_config"]
        assert geometry["context_parallel_size"] == 2
        assert geometry["transformer_config_kwargs"]["cp_comm_type"] == "all_gather"
    for name in ("train-medium-tools", "train-long-checkpoint-eval"):
        assert configs[name]["skyrl"]["trainer"]["policy"]["megatron_config"]["context_parallel_size"] == 1
    recorded = deepcopy(planned)
    recorded["skyrl"]["data"].update(task_cache_dir="~/.cache/marinskyrl/tasks", terminal_bench_data=[])
    del recorded["skyrl"]["environment"]["task_sessions"]["agent_harnesses"]
    del recorded["skyrl"]["trajectory_runner"]["max_concurrent_training_machine_tasks"]
    assert gate.config_version(recorded) == gate.config_version(planned)


def test_snowball_held_out_evaluations_use_full_eval_lite(selection, tmp_path):
    table = snowball_model.SNOWBALL
    speed = table.by_name["speed-short"]
    learning = table.by_name["learning-medium"]
    paths = []
    for preset in (speed, learning):
        (step,) = snowball.build_experiment(preset, table=table).runs
        path = tmp_path / f"{preset.name}.yaml"
        path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
        paths.append(str(path))

    speed_config, learning_config = gate.compose_configs(paths)
    for preset, sources in (
        (speed, speed_config["inputs"]["evaluation_data"]["eval_lite"]),
        (learning, learning_config["inputs"]["validation_data"]),
    ):
        (source,) = sources
        held_out = recipe.selection_artifact(preset.held_out_tasks)
        assert (source["uri"], source["relative_path"]) == (held_out.path(), FULL_VALIDATION_FILENAME)

    # Best-checkpoint selection consumes evaluation metrics before the export callback runs.
    callbacks = [row["type"] for row in learning_config["skyrl"]["trainer"]["callbacks"]]
    evaluation = callbacks.index("evaluation")
    assert evaluation < callbacks.index("best_checkpoint") < callbacks.index("hf_model_save")


@pytest.mark.parametrize(("mode", "exit_code"), [("--plan", 0), ("--run", 4)])
def test_insufficient_controls_restates_the_exact_base_version(recorded_runs, mode, exit_code, monkeypatch):
    table = snowball_model.SNOWBALL
    expected_version = "missing-base-version"
    monkeypatch.setattr(snowball, "_base_preset", lambda *_: ("commit-one", expected_version))

    preset = table.by_name["learning-medium"]
    args = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, mode]
    result = CliRunner().invoke(
        snowball.command(table, argv=tuple(args), entry_module="experiments.post_training.snowball_rl"), args
    )
    assert result.exit_code == exit_code, (result.output, result.exception)
    if mode == "--plan":
        status = printed_status(result.stdout.partition("\nStepSpec(")[0])
        assert len(status["runs"]) == 1
        assert status["preset_version"]
    else:
        assert f"at base version {expected_version}: 0/5 completed main controls" in result.output


@pytest.mark.parametrize(
    ("mode", "recipe_settings", "override", "reason"),
    [
        (
            "--plan",
            ("trainer.policy_mini_batch_size=2", "generator.n_samples_per_prompt=8"),
            "trainer.train_batch_size=2",
            "lcm_dp_size=8",
        ),
        (
            "--run",
            ("trainer.policy_mini_batch_size=2", "generator.n_samples_per_prompt=8"),
            "trainer.train_batch_size=2",
            "lcm_dp_size=8",
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
        argv=["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, mode, "--set", override],
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

    arguments = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, "--plan"]
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
                snowball_model.SNOWBALL,
                argv=["--preset", train.name, "--version", recipe.RUN_ARTIFACT_VERSION],
                init=cloud_model,
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

    speed = snowball_model.SNOWBALL.by_name["speed-short"]
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
        snowball.build_experiment(speed, control=True, table=snowball_model.SNOWBALL).runs[0],
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
    assert (
        configs[4]["inputs"]["validation_data"][0]["identity"]
        .partition(":")[0]
        .endswith("@" + recipe.selection_artifact(speed.tasks).version)
    )
    summary = Path(recipe.selection_artifact(learning.tasks).path()) / recipe.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_experiment(learning, table=snowball_model.SNOWBALL)


def test_authored_learning_budget_survives_composition_and_stops_after_optimizer_start(selection, tmp_path):
    table = snowball_model.SNOWBALL
    preset = table.by_name["train-long"]
    callback = declared.WallTimeBudgetCallback(type="wall_time_budget", seconds=100, drain_margin=20, groups_per_step=64)
    preset = replace(
        preset,
        recipe=replace(
            preset.recipe,
            config=preset.recipe.config.merge(RecipePatch(trainer=sections.Trainer(callbacks=(callback,)))),
        ),
    )
    spec = recipe.run_spec(preset, name="learning-budget", model_name=table.model)
    path = tmp_path / "budget.yaml"
    path.write_text(spec.config_yaml)
    result = native_json()(
        """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import create_callback_from_config

path = Path(sys.argv[1])
config = OmegaConf.load(path)
config.trainer.export_path = str(path.parent)
handler = CallbackHandler([create_callback_from_config(row) for row in config.trainer.callbacks])
trainer = SimpleNamespace(cfg=config, first_optimizer_step_time=None)
control = TrainerControl()
async def event(name, step, **kwargs):
    await handler.call_event_async(name, TrainerState(step, 0, 100, 100), control, trainer=trainer, **kwargs)
asyncio.run(event("on_train_begin", 7))
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=100000):
    asyncio.run(event("on_log", 7, logs={}))
startup = control.should_training_stop
assert not (path.parent / "training-budget.json").exists()
trainer.first_optimizer_step_time = 1000.0
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=1099):
    asyncio.run(event("on_log", 8, logs={}))
before = control.should_training_stop
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=1105):
    asyncio.run(event("on_log", 9, logs={}))
print(json.dumps({"startup_stop": startup, "before_stop": before,
                  "expired_stop": control.should_training_stop,
                  "receipt": json.loads((path.parent / "training-budget.json").read_text())}))
""",
        [str(path)],
    )
    assert not result["startup_stop"]
    assert not result["before_stop"]
    assert result["expired_stop"]
    assert result["receipt"]["elapsed_seconds"] == 105
    assert result["receipt"]["completed_steps"] == 2
    assert result["receipt"]["additional_updates_at_observed_rate"] == 0
    assert result["receipt"]["consumed_groups_per_hour"] == pytest.approx(128 * 3600 / 105)


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
    results = run_native_test(
        "cloud.iris.tests.native_checkpoint_coverage",
        [str(path), str(recipe.CHECKPOINT_RETRY_BUDGET if retry_budget is None else retry_budget)],
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


@pytest.mark.parametrize(
    "table,preset_name",
    [
        pytest.param(MOCHI, "train-medium", id="mochi"),
        pytest.param(snowball_model.SNOWBALL, "speed-short", id="snowball"),
    ],
)
def test_draft_opt_out_preserves_other_generator_settings_and_inputs(selection, tmp_path, table, preset_name):
    preset = table.by_name[preset_name]
    settings = ("generator.speculative_decoding=null",)
    (draft_run,) = snowball.build_experiment(preset, table=table).runs
    (plain_run,) = snowball.build_experiment(preset, table=table, settings=settings).runs
    draft = yaml.safe_load(materialized_config(draft_run, str(tmp_path)).launch_config_yaml)
    plain = yaml.safe_load(materialized_config(plain_run, str(tmp_path)).launch_config_yaml)

    config = draft["skyrl"]["generator"]
    speculator = config.pop("speculative_decoding")
    assert draft_run.name != plain_run.name
    assert speculator["method"] == "dspark"
    assert speculator["num_speculative_tokens"] == 4
    assert plain["skyrl"]["generator"].pop("speculative_decoding", None) is None
    assert config == plain["skyrl"]["generator"]
    assert draft["inputs"] == plain["inputs"]


def test_mochi_three_rack_launch_configures_evaluation_without_model_exports(selection, tmp_path):
    preset = MOCHI.by_name["train-long-3rack"]
    (step,) = snowball.build_experiment(preset, table=MOCHI).runs
    path = tmp_path / "mochi.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)

    (config,) = gate.compose_configs([str(path)], validate_driver=True)
    assert config["skyrl"]["generator"].get("speculative_decoding") is None
    model = config["inputs"]["model"]
    assert model["uri"] == MOCHI_STEP_400407_EXPORT
    assert model["tokenizer_uri"] == MOCHI_TOKENIZER_URI
    assert model["tokenizer_revision"] == MOCHI_TOKENIZER_REVISION

    (source,) = config["inputs"]["validation_data"]
    held_out = recipe.selection_artifact(preset.held_out_tasks)
    assert (source["uri"], source["relative_path"]) == (held_out.path(), VALIDATION_FILENAME)
    assert not config["run"]["export_hf"]
    callbacks = {row["type"] for row in config["skyrl"]["trainer"]["callbacks"]}
    assert "evaluation" in callbacks
    assert callbacks.isdisjoint({"checkpoint", "best_checkpoint", "hf_model_save"})

    docs = (recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
    section = docs.split("`mochi-inference.yaml`", 1)[1]
    fragment = yaml.safe_load(section.split("```yaml\n", 1)[1].split("```", 1)[0])
    assert merge_mappings(config["skyrl"], fragment) == config["skyrl"]


def test_snowball_omits_default_draft_for_an_alternate_initializer(selection, tmp_path):
    table = snowball_model.SNOWBALL
    preset = table.by_name["eval-short"]
    initializer = ArtifactHfModel(
        ArtifactStep.adopt("tests/alternate-initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint),
        relative_path="",
    )
    [alternate] = snowball.build_experiment(preset, table=table, init=initializer).runs
    config = yaml.safe_load(materialized_config(alternate, str(tmp_path)).launch_config_yaml)

    assert config["skyrl"]["generator"].get("speculative_decoding") is None


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
    original_task_set = Path(recipe.selection_artifact(f"{task_set_name}@{task_set_version}").path())
    new_task_set = Path(recipe.selection_artifact(f"{task_set_name}@2026.11.05").path())
    new_task_set.mkdir(parents=True)
    (new_task_set / "summary.json").write_bytes((original_task_set / "summary.json").read_bytes())
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
    index_sources = []
    table = snowball_model.SNOWBALL
    paths = (
        repeated.runs[0],
        repeated.runs[1],
        run(learning, "trainer.policy.optimizer_config.lr=2e-6"),
    )
    for index, step in enumerate(paths):
        source = tmp_path / f"index-{index}.yaml"
        prefix = recipe.artifact_prefix(learning.recipe.layout.cluster)
        source.write_text(materialized_config(step, prefix).launch_config_yaml)
        index_sources.append(str(source))
    provenance = replace(Provenance.capture(), branch=None)
    configs = gate.compose_configs(index_sources)
    first_index, repeated_index, changed_index = (
        snowball.run_index_identity(config, learning, table, provenance, user="test-user") for config in configs
    )
    assert first_index.run_id == repeated_index.run_id != changed_index.run_id
    assert first_index.preset_version == repeated_index.preset_version != changed_index.preset_version
    config = configs[0]
    config["iris"]["priority"] = "production"
    control_index = snowball.run_index_identity(config, learning, table, provenance, user="test-user", control=True)
    assert control_index.is_control and control_index.preset_version == first_index.preset_version
    controls(recorded_runs, "learning-medium")
    first = snowball.build_experiment(learning, requirements=("eval_lite/mean>=0",), table=snowball_model.SNOWBALL)
    second = snowball.build_experiment(learning, requirements=("eval_lite/mean>=0.1",), table=snowball_model.SNOWBALL)
    assert first.runs[0].name == second.runs[0].name == name()
    candidate = recorded_runs(learning.name, first.runs[0].name)
    cohort = cohort_from_fixtures(candidate)
    first_gate = gate.gate_step(learning, first.runs, inputs=first.inputs, cohort=cohort, group_name=first.group_name)
    second_gate = gate.gate_step(
        learning, second.runs, inputs=second.inputs, cohort=cohort, group_name=second.group_name
    )
    assert first_gate.name != second_gate.name
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
        assert snowball.build_experiment(selected, tasks=selection, table=snowball_model.SNOWBALL).runs
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
        ("candidate-compose", 4),
        ("mixed-native-failed", 2),
        ("control-missing-receipt", 3),
        ("partial-retry", 0),
        ("data-short", 0),
        ("unexpected", 3),
    ],
)
def test_submission_exit_codes_and_publication_retries(
    recorded_runs, selection, tmp_path, monkeypatch, preset_table, cache_native_compositions, case, expected
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
    if preset.gated:
        controls(recorded_runs, preset.name)
    replicas = 2 if case in {"partial-retry", "mixed-native-failed", "control-missing-receipt"} else 1
    control = case == "control-missing-receipt"
    if control:
        clean_main = Provenance("tree-one", "commit-one", False, "main", "test")
        monkeypatch.setattr(Provenance, "capture", lambda *_: clean_main)
        monkeypatch.setattr(snowball, "_base_preset", lambda *_: ("commit-one", "current-version"))
    built = snowball.build_experiment(
        preset,
        table=snowball_model.SNOWBALL,
        control="test-batch" if control else None,
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
    if source is not None and preset.gated and not control:
        cohort = cohort_from_fixtures(source)
        monkeypatch.setattr(snowball, "_base_preset", lambda *_: ("commit-one", cohort.version))
        built = replace(
            built,
            gate=gate.gate_step(preset, built.runs, inputs=built.inputs, cohort=cohort, group_name=built.group_name),
        )
    if case != "child":
        for index, root in enumerate(sources):
            native = read_attempt(str(root / "exports"), f"native-{index}")
            write_attempt(
                str(root / "exports"), AttemptRecord(run_id=native.run_id, attempt_id=native.attempt_id, issue=73)
            )
    if case == "changed-source":
        step = built.runs[0]
        write_skyrl_run_record(source, step.name, step.version, fingerprint=step.fingerprint(), attempt_id="native-0")
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
    phase = {"retry": False, "canceled": False}
    original_provenance = Provenance.capture
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
        assert re.fullmatch(f"snowball-{preset.name}-[^/]+-[0-9a-f]{{8}}", query["var-run"][0])
        assert query["var-cluster"] == [preset.recipe.layout.cluster]
        assert query["var-training_type"] == ["sync" if preset_name == "data-short" else "async"]
        assert int(query["from"][0]) > 0

    def finish(*args, **kwargs):
        if phase["canceled"]:
            return SimpleNamespace(state=JobState.KILLED)
        if case != "child":
            assert_submission_link()
        for path, data in pending.items():
            path.write_bytes(data)
        mixed_failure = case == "mixed-native-failed" and not phase["retry"]
        if mixed_failure:
            receipt = json.loads((source / "terminal.json").read_text())
            receipt["result"]["state"] = "failed"
            write_receipt(source, receipt)
            (source / "terminal.json").unlink()
        elif case == "mixed-native-failed":
            receipt = json.loads((source / "terminal.json").read_text())
            metrics = read_skyrl_metrics(str(source), attempt_id="native-0")
            receipt["config"]["run"]["attempt_id"] = "native-retry"
            write_receipt(source, receipt)
            copy_evaluations(source, "native-0", "native-retry")
            for row in metrics.values():
                row["attempt_id"] = "native-retry"
            append_metrics(source, metrics)
        if control and (not phase["retry"]):
            (source / "terminal.json").unlink()
            (source / "resolved-launch.yaml").unlink()
            return SimpleNamespace(state=JobState.FAILED)
        for path in sources:
            native = snowball.read_stored_skyrl_attempt(str(path))
            if native is not None and case != "changed-source":
                launch = native.native.config
                launch["run"]["coordinator_iris_job_id"] = str(job)
                write_attempt(
                    str(path / "exports"),
                    AttemptRecord(run_id=native.run_id, attempt_id=native.attempt_id, config=launch),
                )
        if not mixed_failure and case not in {"missing", "native-failed", "child"} and built.gate is not None:
            run(gate.completed_gate(built.gate))
        if case == "changed-source":
            other = json.loads((source / "terminal.json").read_text())
            other["config"]["run"].update(attempt_id="foreign-attempt", coordinator_iris_job_id="/test-user/other")
            write_receipt(source, other)
            append_metrics(source, {1: {"step": 1, "attempt_id": "foreign-attempt", "policy/entropy": 0.3}})
            assert snowball.read_stored_skyrl_attempt(str(source)).attempt_id == "foreign-attempt"
            monkeypatch.setattr(
                Provenance,
                "capture",
                lambda *args, **kwargs: Provenance(
                    "tree-edited-before-publication", "commit-one", True, "branch", "test"
                ),
            )
        return SimpleNamespace(state=JobState.FAILED if mixed_failure or case == "native-failed" else JobState.SUCCEEDED)

    client, job = fake_iris(monkeypatch, "dashboard-coordinator")
    client.job.return_value.state_only.side_effect = lambda: finish().state
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
    arguments = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, "--run"]
    if case != "data-short":
        arguments.extend(["--issue", "73"])
    if case != "unexpected":
        arguments.append("--wait")
    if "candidate-compose" in case:
        arguments.extend(["--set", "trainer.algorithm.policy_loss_type=gspo"])
    if control:
        arguments.extend(("--control", "test-batch"))
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
                child.setenv("IRIS_TASK_ID", f"{job}/0")
                child.setattr(client, "submit", lambda **kwargs: pytest.fail("Coordinator submitted another job"))
                child.setattr(
                    "experiments.post_training.rl_publication.publish_result",
                    lambda *args: pytest.fail("Coordinator contacted Marina"),
                )
                child.setattr(
                    "experiments.post_training.rl_index.list_attempts",
                    lambda *_args: pytest.fail("Coordinator selected a different cohort"),
                )
                info = JobInfo(task_id=JobName.from_wire(f"{job}/0"))
                with iris_ctx_scope(IrisContext.from_job_info(info)):
                    completed = CliRunner().invoke(selected, invoked)
                assert completed.exit_code == 0, (completed.output, completed.exception)
                address = gate.gate_address(built.gate, recipe.run_prefix(preset))
                saved = gate.read_gate_report(address.completed_path())
                assert saved is not None
                submitted = gate.ControlCohort.model_validate_json(
                    kwargs["environment"].env_vars[snowball.CONTROL_COHORT_ENV]
                )
                assert saved.control.attempts == submitted.attempts
            set_job_info(None)
            return SimpleNamespace(job_id=job)

        client.submit.side_effect = reenter
    set_job_info(None)
    result = CliRunner().invoke(selected, invoked)
    assert result.exit_code == expected, (result.output, result.exception)
    if "candidate-compose" in case:
        assert "native configuration cannot be composed:" in result.output and "gspo" in result.output.lower()
        assert "Compose document 1/1" in result.output
    if case == "unexpected":
        assert "KeyError" in result.output and "grader_kind_counts" in result.output
    if case == "child":
        assert len(published) == 1 and published[0]["issue"] == 73
        assert snowball.rl_publication.lineage_issue(str(source / "exports")) == 73
        report = gate.completed_gate(built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"
        return
    if not publication_case:
        return
    snapshot = printed_status(result.stdout)
    if case == "data-short":
        assert snapshot["publication"]["unserved_records"] == [], snapshot["publication"]
        assert len(published) == 1 and published[0]["status"] == "succeeded"
        assert published[0]["optimizer_steps"] == 0 and published[0]["data_identity"] is None
        assert published[0]["evaluation_data_identity"]
        return
    paths = [
        [f"{training.name}-{recipe.RUN_ARTIFACT_VERSION}", f"native-{index}"]
        for index, training in enumerate(built.runs)
    ]
    if unavailable["value"]:
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Credential unavailable; run iris login"],
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert [row["attempt_id"] for row in published] == ["native-0", "native-1"]
        assert snapshot["publication"]["unserved_records"] == []
        assert published[1]["failed_facts"] == ["required_gate_report"]
    if case == "mixed-native-failed":
        assert published[0]["status"] == "failed" and published[0]["gate_result"] is None
    if expected == 0 and (not unavailable["value"]):
        for row in snapshot["runs"]:
            assert parse_qs(urlsplit(row["rl_progress_url"]).fragment)["attempt"] == [row["attempt_id"]]
    if case == "changed-source":
        [record] = published
        assert record["attempt_id"] == "native-0" and record["gate_result"]["verdict"] == "pass"
        phase["canceled"] = True
        monkeypatch.setattr(Provenance, "capture", original_provenance)
        repeated = CliRunner().invoke(selected, invoked)
        assert repeated.exit_code == 3, (repeated.output, repeated.exception)
        assert "verdict" not in printed_status(repeated.stdout) and published == [record]
        return
    unavailable["value"] = False
    phase["retry"] = True
    issue_offset = arguments.index("--issue")
    if case == "partial-retry":
        arguments[issue_offset + 1] = "74"
        refused = CliRunner().invoke(snowball.command(snowball_model.SNOWBALL, argv=tuple(arguments)), arguments)
        assert refused.exit_code == 4 and "conflicting lineage issues" in refused.output
    del arguments[issue_offset : issue_offset + 2]
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
        {"native-retry": 73, "native-1": 73} if case == "mixed-native-failed" else {attempt: 73 for _, attempt in paths}
    )
    if case == "mixed-native-failed":
        retry = next(row for row in published[previously_published:] if row["attempt_id"] == "native-retry")
        assert retry["status"] == "succeeded" and retry["gate_result"]["verdict"] == "pass" and retry["issue"] == 73
    else:
        cached = next(row for row in published[previously_published:] if row["attempt_id"] == "native-0")
        assert cached["issue"] == 73


@pytest.mark.parametrize(
    "model,outcome,pr_number",
    [
        ("snowball", "succeeded", 124),
        ("snowball", "native-failure", 124),
        ("snowball", "interrupted", None),
        ("mochi", "succeeded", 124),
    ],
)
def test_detached_ungated_launch_indexes_prepared_then_terminal_attempt(
    preset_table, recorded_runs, monkeypatch, model, outcome, pr_number, tmp_path
):
    early_failure = outcome != "succeeded"
    real_archive = model == "snowball" and not early_failure
    entry_module = f"experiments.post_training.{model}_rl"
    if real_archive:
        entry_module = "experiments.post_training.rl_tests.archive_preset_fixture"
        monkeypatch.setenv("MARIN_RL_TEST_SELECTION_ROOT", recipe.SELECTION_ROOT)
    if model == "mochi":
        table = MOCHI
        preset = nominal = table.by_name["train-medium"]
        binding = recipe.selection_artifact(preset.tasks)
        path = Path(binding.path()) / "summary.json"
        if not path.is_file():
            _json(path, selection_summary(100))
        marker = StoragePath(preset.model.step.path()) / MODEL_MANIFEST_FILENAME
        fake_s3(monkeypatch, str(marker).removeprefix("s3://"))
    else:
        table = snowball_model.SNOWBALL
        nominal = table.by_name["train-long"]
        preset = replace(
            nominal.with_updates(2),
            checkpoints="none",
            export="none",
            checkpoint_every=None,
            evaluations=(),
        )
        table = preset_table(preset)
    git_output = snowball.git_output
    source_ref = git_output(["rev-parse", "HEAD"], cwd=recipe.REPO_ROOT)
    archive_tree = git_output(["rev-parse", "HEAD^{tree}"], cwd=recipe.REPO_ROOT)
    monkeypatch.setattr(
        Provenance, "capture", lambda *a, **kw: Provenance("tree-one", source_ref, False, "branch", "test")
    )

    def base_ref(args, cwd, **kwargs):
        if args[0] == "merge-base":
            assert args == ["merge-base", source_ref, "origin/main"]
            if outcome == "interrupted":
                raise RuntimeError("main merge base unavailable")
            return archive_tree
        return git_output(args, cwd, **kwargs)

    monkeypatch.setattr(snowball, "git_output", base_ref)
    built = snowball.build_experiment(preset, table=table)
    source = recorded_runs(preset.name, built.runs[0].name, inputs=built.inputs, attempt_id="native-old", table=table)
    receipt = json.loads((source / "terminal.json").read_text())
    launch = receipt["config"]
    historical_launch = json.loads(json.dumps(launch))
    historical_launch["skyrl"]["trainer"]["max_steps"] += 1
    if real_archive:
        historical_launch = gate.compose_preset(nominal)
    historical_version = gate.config_version(historical_launch)
    client, job = fake_iris(monkeypatch, "index-coordinator")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.setattr(snowball, "_PUBLICATION_POLL_SECONDS", 0)
    publications = []
    unavailable = outcome == "interrupted"
    running_retries = outcome == "succeeded"

    def publish(record):
        nonlocal unavailable, running_retries
        if record.status == "running" and running_retries:
            running_retries = False
            raise RuntimeError("Marina unavailable during the running write")
        if record.attempt_id == "native-indexed" and record.status == "failed" and unavailable:
            unavailable = False
            raise RuntimeError("Marina unavailable during the terminal write")
        publications.append(record)

    fake_publication(monkeypatch, publish, "https://progress.test")
    followers = []

    def run_local(argv, **kwargs):
        if argv[0] == "gh":
            if pr_number is None:
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0, str(pr_number))
        if argv[:2] == ["git", "archive"]:
            assert argv[2] == archive_tree
        if argv[1:4] == ["-m", entry_module, "_preset-config"]:
            assert argv[4:] == [preset.name]
            assert (kwargs["cwd"] / "vendor/skyrl/cloud/iris/config_cli.py").is_file()
            assert str(kwargs["cwd"]) in kwargs["env"]["PYTHONPATH"]
            if not real_archive:
                return subprocess.CompletedProcess(argv, 0, json.dumps(historical_launch))
        return subprocess.run(argv, **kwargs)

    monkeypatch.setattr(
        snowball,
        "subprocess",
        SimpleNamespace(
            run=run_local,
            SubprocessError=subprocess.SubprocessError,
            DEVNULL=subprocess.DEVNULL,
            STDOUT=subprocess.STDOUT,
            Popen=lambda argv, **kwargs: followers.append(argv[3:]),
        ),
    )

    def native_prepared(**kwargs):
        run_id = launch["run"]["id"]
        index = RunIndexIdentity.model_validate(json.loads(kwargs["environment"].env_vars[RUN_INDEX_ENV])[run_id])
        launch["run"].update(attempt_id="native-indexed", coordinator_iris_job_id=str(job))
        other_config = {
            **launch,
            "run": {**launch["run"], "attempt_id": "other-coordinator", "coordinator_iris_job_id": "/test-user/other"},
        }
        export_root = launch["artifacts"]["export_root"]
        write_attempt(
            export_root,
            AttemptRecord(run_id=run_id, attempt_id="other-coordinator", config=other_config, index=index),
        )
        write_attempt(
            export_root,
            AttemptRecord(
                run_id=run_id,
                attempt_id="native-indexed",
                config=launch,
                index=index,
                metrics_started_at=None if early_failure else receipt["result"]["started_at"],
                elapsed_excluding_eval_seconds=None if early_failure else 123.0,
                result=(
                    LaunchResult(state="failed", iris_job_id="/test-user/unschedulable")
                    if outcome == "native-failure"
                    else None
                ),
            ),
        )
        return SimpleNamespace(job_id=job)

    def state():
        nonlocal running_observed
        if early_failure:
            return JobState.FAILED
        if running_observed < 2:
            running_observed += 1
            return JobState.RUNNING
        _json(source / "terminal.json", receipt)
        _json(source / "resolved-launch.yaml", {"config": launch})
        metrics = read_skyrl_metrics(str(source), attempt_id="native-old")
        for row in metrics.values():
            row["attempt_id"] = "native-indexed"
        append_metrics(source, metrics)
        return JobState.SUCCEEDED

    running_observed = 0
    client.submit.side_effect = native_prepared
    client.job.return_value.state_only.side_effect = state
    arguments = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, "--run"]
    result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments), entry_module=entry_module), arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    grafana_link = next(
        line.strip().removeprefix("run: ") for line in reversed(result.output.splitlines()) if "grafana.oa.dev" in line
    )
    [follower] = followers
    saved_attempt = ["--preset", preset.name, "--run-output", str(source), "--attempt", "native-indexed"]
    if outcome == "interrupted":
        still_prepared = CliRunner().invoke(snowball.rl_publication.command(table), [*saved_attempt, "--issue", "73"])
        assert still_prepared.exit_code == 1
        assert read_attempt(str(source / "exports"), "native-indexed").issue is None
        assert snowball.rl_publication.lineage_issue(str(source / "exports")) is None

    @click.command()
    def follow():
        raise click.exceptions.Exit(snowball.main(table, argv=follower))

    if outcome == "interrupted":
        with monkeypatch.context() as timeout:
            clock = iter((0.0, 1_000_000.0))
            timeout.setattr(snowball, "time", SimpleNamespace(monotonic=lambda: next(clock)))
            timeout.setattr(client.job.return_value, "state_only", lambda: JobState.RUNNING)
            timed_out = CliRunner().invoke(follow)
        assert timed_out.exit_code == 3 and "Check its Iris state before resubmitting" in timed_out.output
        assert read_attempt(str(source / "exports"), "native-indexed").interruption is None
        assert [row.status for row in publications] == ["running"]
        publications.clear()

    result = CliRunner().invoke(follow)
    assert result.exit_code == int(early_failure), (result.output, result.exception)
    if outcome == "interrupted":
        interrupted = read_attempt(str(source / "exports"), "native-indexed")
        assert interrupted.result is None and interrupted.interruption == "coordinator_terminated"
        replayed = CliRunner().invoke(snowball.rl_publication.command(table), saved_attempt)
        assert replayed.exit_code == 0, (replayed.output, replayed.exception)
        assert publications[-1].failed_facts == ["coordinator_interrupted"]
    assert {row.attempt_id for row in publications} == {"native-indexed"}
    running = [row for row in publications if row.status == "running"]
    assert [row.attempt_id for row in running] == ([] if outcome == "native-failure" else ["native-indexed"])
    terminal = next(row for row in publications if row.attempt_id == "native-indexed" and row.status != "running")
    assert terminal.status == ("failed" if early_failure else "succeeded")
    assert terminal.pr_number == pr_number
    assert all(row.run_id == terminal.run_id and row.pr_number == pr_number for row in running)
    assert parse_qs(urlsplit(grafana_link).query)["var-run"] == [terminal.run_id]
    assert re.fullmatch(rf"{model}-{preset.name}-[^/]+-[0-9a-f]{{8}}", terminal.run_id)
    assert terminal.source_revision == f"base={source_ref} tree=tree-one"
    expected_base = None if outcome == "interrupted" else historical_version
    assert terminal.base_preset_version == expected_base
    assert terminal.base_commit == (archive_tree if expected_base else None)
    assert terminal.preset_version == gate.config_version(launch) != historical_version
    assert all(row.base_preset_version == expected_base for row in running)
    if expected_base is None:
        assert "The base preset version is unavailable from the branch's main merge base." in terminal.evidence
    if not early_failure:
        assert parse_qs(urlsplit(terminal.grafana_url).query)["var-job"] == [receipt["result"]["iris_job_id"]]
    assert terminal.gate_result is None
    assert terminal.elapsed_excluding_eval_seconds == (None if early_failure else 123.0)
    assert terminal.artifact_uri == str(source)
    if outcome == "native-failure":
        result = CliRunner().invoke(snowball.rl_publication.command(table), saved_attempt)
        assert result.exit_code == 0, (result.output, result.exception)
        assert publications[-1].status == "failed"


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
            snowball_model.SNOWBALL.by_name["speed-medium"], control="test-batch", table=snowball_model.SNOWBALL
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
        controls(record, preset_name)
        source = record(preset_name, "tests/required-metric")
        cohort = cohort_from_fixtures(source)
        request.getfixturevalue("monkeypatch").setattr(
            snowball, "_base_preset", lambda *_: ("commit-one", cohort.version)
        )
        arguments = (*arguments, "--run")
    result = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            init=initializer,
            entry_module="experiments.post_training.snowball_rl",
        ),
        ["--preset", preset_name, "--version", recipe.RUN_ARTIFACT_VERSION, *arguments],
    )
    assert result.exit_code == 4, (result.output, result.exception)
    if initializer is not None:
        assert "Python init and --init" in result.output
    if "--require" in arguments:
        assert "does not compute" in result.output
    if "trainer.dump_eval_results=false" in arguments:
        assert "owned" in result.output
    if "trainer.hf_hub_repo_id=org/model" in arguments:
        assert "use the preset's declared owned keys" in result.output


@pytest.mark.parametrize("recorded_runs", [2], indirect=True, ids=["one-task-per-family"])
def test_train_long_reenters_the_model_module_and_reports_checkpoint_evaluations(
    recorded_runs, preset_table, tmp_path, monkeypatch
):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(
        preset,
        checkpoints="best",
        export="best",
        checkpoint_every=2,
        evaluations=(replace(preset.evaluations[0], every=2),),
        recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_disk="3TB")),
    )
    monkeypatch.setattr(snowball, "_branch_pr_number", lambda provenance: None)
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
    published_selections = Path(recipe.SELECTION_ROOT)

    def selection_files():
        return {
            path: (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in (published_selections, *published_selections.rglob("*"))
        }

    before_launch = selection_files()
    settings = ("trainer.max_steps=3", "trainer.project_name=fixture-project")
    set_args = [arg for setting in settings for arg in ("--set", setting)]
    built = snowball.build_experiment(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=settings
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    _json(source / "exports/best_checkpoint.json", {"step": 2})
    earlier = read_attempt(str(source / "exports"), "recorded-attempt")
    assert earlier is not None
    linked = earlier.model_copy(update={"issue": 73})
    write_attempt(str(source / "exports"), linked)
    staged = source.with_name("before-native-execution")
    source.rename(staged)
    write_attempt(str(source / "exports"), AttemptRecord(run_id=linked.run_id, attempt_id="earlier", issue=73))
    native_calls = []

    def native(command, *, env_vars=None):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        stored = json.loads((source / "terminal.json").read_text())
        response = stored["result"]
        response["model"] = dict(
            policy_export_uri=str(source / "policy"),
            global_step=3,
            tokenizer_uri="tests/python-tokenizer",
            tokenizer_revision="python-tokenizer-revision",
            checkpoint_root=str(source / "checkpoints"),
            terminal_manifest_uri=str(source / "terminal.json"),
        )
        response_path = tmp_path / "native-response.json"
        _json(response_path, response)
        recorded = run_native_test(
            "cloud.iris.tests.native_record_attempt",
            [str(response_path), *command[command.index("marinskyrl") + 1 :]],
        )
        assert recorded["forward_env"] == ["HF_TEST_KEY"]
        receipt = recorded["receipt"]
        export_root = receipt["config"]["artifacts"]["export_root"]
        completed = read_attempt(export_root, receipt["result"]["attempt_id"])
        assert completed is not None and completed.index is not None
        assert completed.result is not None and completed.result.state == "succeeded"
        assert completed.index.model == table.model and completed.index.preset == preset.name
        assert completed.index.wandb_project == "fixture-project"
        assert completed.native.config["iris"]["allocation"]["disk"] == "3TB"
        assert completed.native.config["inputs"]["model"]["tokenizer_uri"] == "tests/python-tokenizer"
        assert completed.issue is None
        assert snowball.rl_publication.lineage_issue(export_root) == 73
        native_calls.append(command)
        # The fixture supplies metrics and startup writes that normally come from the trainer.
        write_receipt(source, receipt)
        write_attempt(
            export_root,
            AttemptRecord(
                run_id=completed.run_id,
                attempt_id=completed.attempt_id,
                started_at=receipt["result"]["started_at"],
                metrics_started_at=receipt["result"]["started_at"],
                evaluation_storage_version=1,
            ),
        )
        metrics = read_skyrl_metrics(str(source), earlier.attempt_id)
        for row in metrics.values():
            row["attempt_id"] = completed.attempt_id
        append_metrics(source, metrics)
        copy_evaluations(source, earlier.attempt_id, completed.attempt_id)
        return subprocess.CompletedProcess(command, 0, json.dumps(receipt["result"]), "")

    monkeypatch.setattr(skyrl_api, "_run_launcher", native)
    published = []

    fake_publication(monkeypatch, published.append, "http://progress.test")
    arguments = [
        "--preset",
        preset.name,
        "--version",
        recipe.RUN_ARTIFACT_VERSION,
        "--run",
        "--wait",
        "--marina-url",
        "http://127.0.0.1:55777",
        "--max-concurrent",
        "3",
        *set_args,
    ]
    client, job = fake_iris(monkeypatch, "train-coordinator")
    default_module = snowball_model.__name__
    second_entry = "experiments.post_training.mochi_rl"
    mode, coordinator_module = "valid", default_module
    entries = {}

    def invoke_entry(name, patch):
        module_spec = find_spec(name)
        assert module_spec is not None and module_spec.loader is not None
        patch.setattr(sys.modules["__main__"], "__spec__", module_spec)
        return model_entry_fixture.main(entries[name], arguments)

    def submit(**kwargs):
        command = kwargs["entrypoint"].command
        assert command == ["python", "-m", coordinator_module, *arguments]
        if coordinator_module == default_module and mode == "valid" and not native_calls:
            opened = subprocess.run(
                [sys.executable, "-m", snowball_model.__name__, "--help"],
                cwd=recipe.REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            assert "--preset" in opened.stdout and "--wait" in opened.stdout
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
            child.setattr(
                "experiments.post_training.rl_publication.publish_result",
                lambda *args: pytest.fail("Coordinator contacted Marina"),
            )
            info = JobInfo(task_id=JobName.from_wire("/test-user/train-coordinator/0"))
            with iris_ctx_scope(IrisContext.from_job_info(info)):
                code = invoke_entry(coordinator_module, child)
            assert code == {"valid": 0, "failed-following": 3}.get(mode, 4)
        set_job_info(None)
        return SimpleNamespace(job_id=job)

    client.submit.side_effect = submit
    client.job.return_value.state_only.side_effect = lambda: JobState.SUCCEEDED if mode == "valid" else JobState.FAILED

    policy_uri = "s3://marin-test/models/python-policy"
    policy_path = tmp_path / "python-policy"
    raw_load = SkyRLRun.raw_load
    monkeypatch.setattr(SkyRLRun, "raw_load", lambda path: raw_load(str(policy_path) if path == policy_uri else path))

    def module(
        name, *, entry_module=None, use_init=True, forward_env=("HF_TEST_KEY",), policy_forward=None, following=False
    ):
        entries[name] = {
            "entry_module": entry_module,
            "use_init": use_init,
            "policy_forward": policy_forward,
            "forward_env": forward_env,
            "source": str(source),
            "policy_path": str(policy_path),
            "policy_uri": policy_uri,
            "following": following,
        }

    @click.command()
    def entry():
        with monkeypatch.context() as driver:
            raise click.exceptions.Exit(invoke_entry(default_module, driver))

    def invoke_with_preflight_stub(command, args=()):
        # Native preflight has its own tests. Use the real composer to create
        # recorded fixtures and inspect the native launch in this test.
        config = earlier.native.config
        with monkeypatch.context() as preflight:
            preflight.setattr(gate, "compose_configs", lambda paths, **_: [config for _ in paths])
            return CliRunner().invoke(command, args)

    for override in (None, second_entry):
        coordinator_module = override or default_module
        module(default_module, entry_module=override)
        if override is not None:
            module(override)
        result = invoke_with_preflight_stub(entry)
        assert result.exit_code == 0, (result.output, result.exception)
        assert "best_checkpoint_step: 2" in result.output and f"{preset.evaluations[0].name}@3/scored" in result.output
        assert "http://127.0.0.1:55777/" in result.output and "test-initializer-credential" not in result.output
    assert len(native_calls) == 1
    assert selection_files() == before_launch
    assert published[-1].optimizer_steps == published[-1].planned_updates == 3
    assert (published[-1].model, published[-1].issue) == ("snowball", 73)

    for mode in (
        "initializer-mismatch",
        "environment-mismatch",
        "overlapping-environment-mismatch",
        "dependency-environment-mismatch",
    ):
        coordinator_module = second_entry
        module(
            default_module,
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

    mode, coordinator_module = "failed-following", default_module
    module(coordinator_module, following=True)
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and "post_run_error:" in result.output, (result.output, result.exception)
    assert "RL has a committed result" in result.output

    completed_id = published[-1].attempt_id
    assert completed_id != earlier.attempt_id
    evaluations = read_evaluations(str(source / "exports"), attempt_id=completed_id, evaluation="eval/")
    replace_evaluations(
        source,
        [
            row.model_copy(
                update={
                    "answers": [answer.model_copy(update={"error_treatment": "passthrough"}) for answer in row.answers]
                }
            )
            for row in evaluations
            if row.step != 2
        ],
    )
    (source / "exports/best_checkpoint.json").unlink()
    with edit_receipt(source) as receipt:
        receipt["result"]["model"] = None
    step = built.runs[0]
    write_skyrl_run_record(
        source, step.name, step.version, fingerprint=step.fingerprint(), global_step=None, attempt_id=completed_id
    )
    planning_args = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, *set_args]
    result = invoke_with_preflight_stub(
        snowball.command(table, argv=tuple(planning_args), init=initializer, forward_env=("HF_TEST_KEY",)),
        planning_args,
    )
    assert result.exit_code == 1, (result.output, result.exception)
    [status] = printed_status(result.stdout)["runs"]
    assert status["best_checkpoint_step"] == "none eligible"
    facts = {fact["name"]: fact for fact in status["evaluation_invariants"]}
    assert not facts[f"{preset.evaluations[0].name}/step-2/answers_complete"]["holds"]
    assert not facts[f"{preset.evaluations[0].name}/task_coverage"]["holds"]

    mode, coordinator_module = "valid", default_module
    module(coordinator_module)
    resumed = invoke_with_preflight_stub(entry)
    assert resumed.exit_code == 1 and len(native_calls) == 1, (resumed.output, resumed.exception)
    assert "best_checkpoint_step: none eligible" in resumed.output

    replace_evaluations(source, evaluations)

    live_preset = replace(preset, checkpoints="none", checkpoint_every=None, export="none")
    live = snowball.build_experiment(
        live_preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=settings
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
    assert not facts[f"{preset.evaluations[0].name}/step-2/answers_complete"]["holds"]

    mode, coordinator_module = "failed-native", default_module
    module(default_module)
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "failed"
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)


@pytest.mark.parametrize("model,sft_updates,rl_updates", [("mochi", 524, 72), ("snowball", 360, 36)])
def test_sft_mix_arm_uses_curriculum_validation_and_step_budget(
    model, sft_updates, rl_updates, tmp_path, monkeypatch, selection, mochi_sft_presets
):
    monkeypatch.setenv("MARIN_PREFIX", SFT_ARM_TEST_PREFIX)
    fake_s3_checkpoint(monkeypatch, SFT_ARM_TEST_PREFIX, tmp_path / "cloud")
    monkeypatch.setattr(snowball, "read_skyrl_attempt", lambda _path: None)
    monkeypatch.setattr(snowball, "read_skyrl_metrics", lambda _path: {})
    monkeypatch.setattr(recipe, "artifact_prefix", lambda _cluster: SFT_ARM_TEST_PREFIX)
    sft = mochi_sft_presets["periodic"] if model == "mochi" else MIXTURE_262K_32NODE
    table = MOCHI if model == "mochi" else snowball_model.SNOWBALL
    tokenizer = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
    data = LmDataConfig(
        tokenizer=tokenizer,
        components={"v5": DatasetComponent(cache_dir="/regional/v5", flat_cache=True)},
        train_weights={"v5": 1.0},
        auto_build_caches=False,
    )
    root = tmp_path / "v5"
    root.mkdir()
    manifest = "mix.json"
    (root / manifest).write_text(json.dumps(draccus.encode(data)))
    plans = []
    monkeypatch.setattr("marin.experiment.cli._print_plan", plans.extend)
    arguments = [
        "--preset",
        "train-long",
        "--sft",
        sft.name,
        "--sft-data-mix",
        "tests/sft-v5@2026.11.25",
        "--sft-data-uri",
        str(root),
        "--sft-data-file",
        manifest,
        "--sft-data-tokenizer",
        tokenizer,
        "--version",
        "2026.11.25.2",
        "--plan",
    ]
    result = CliRunner().invoke(snowball.command(table, argv=tuple(arguments), sft_presets=lambda: (sft,)), arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    [run] = plans
    handles = graph_handles([run])
    [training] = [
        step
        for step in handles
        if "mix-" in step.name
        and step.name.startswith(("grug/", "users/", "checkpoints/"))
        and any(dep.name == "tests/sft-v5" for dep in step.deps)
    ]
    config = materialized_config(training, str(tmp_path))
    if model == "mochi":
        assert config.stop_step == config.trainer.trainer.num_train_steps == 393491 + sft_updates
        assert config.data.mixture_start_step == 393491
        assert config.optimizer.decay == 52
        train_config = config
    else:
        train_config = config.train_config
        assert train_config.trainer.num_train_steps == sft_updates
    assert train_config.data.components == data.components
    launch = yaml.safe_load(materialized_config(run, str(tmp_path)).launch_config_yaml)
    trainer = launch["skyrl"]["trainer"]
    assert trainer["max_steps"] == rl_updates
    assert trainer["eval_interval"] == 3
    assert trainer["checkpoint_keep_every"] == 3
    callbacks = {item["type"]: item for item in trainer["callbacks"]}
    assert callbacks["checkpoint"]["save_steps"] == callbacks["evaluation"]["eval_steps"] == 3
    assert callbacks["best_checkpoint"]["expected_tasks"] == 200
    assert callbacks["best_checkpoint"]["selection_window"] == 3
    assert trainer["hf_export_checkpoint"] == "best"
    assert (
        launch["inputs"]["train_data"][0]["uri"].rsplit("/", 1)[0]
        == launch["inputs"]["validation_data"][0]["uri"].rsplit("/", 1)[0]
    )
    execution = skyrl_api.skyrl_execution(run)
    assert execution.coordinator_timeout_hours == execution.job_timeout_seconds == 0
    environment = {key: "fixture" for key in execution.required_env_vars}
    _, request = _coordinator_request([run], "experiments.post_training.snowball_rl", (), Path.cwd(), environment)
    assert request.timeout is None
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))
    native = native_json()(
        """
import json,sys
from omegaconf import OmegaConf
from skyrl_train.trainer import RayPPOTrainer
config = json.load(open(sys.argv[1]))['skyrl']
trainer = RayPPOTrainer.__new__(RayPPOTrainer)
trainer.cfg = OmegaConf.create(config)
trainer.train_dataset = range(100000)
trainer._configure_training_schedule()
print(json.dumps({'stop':trainer.total_training_steps,'available':trainer.available_training_steps}))
""",
        [str(path)],
    )
    assert native["stop"] == rl_updates < native["available"]


def test_sft_only_arm_exports_without_starting_rl(tmp_path, monkeypatch, selection):
    monkeypatch.setenv("MARIN_PREFIX", SFT_ARM_TEST_PREFIX)
    fake_s3_checkpoint(monkeypatch, SFT_ARM_TEST_PREFIX, tmp_path / "cloud")
    monkeypatch.setattr(snowball, "read_skyrl_attempt", lambda _path: None)
    monkeypatch.setattr(snowball, "read_stored_skyrl_attempt", lambda _path: None)
    monkeypatch.setattr(snowball, "read_skyrl_metrics", lambda _path: {})
    monkeypatch.setattr(recipe, "artifact_prefix", lambda _cluster: SFT_ARM_TEST_PREFIX)
    tokenizer = MIXTURE_262K_32NODE.recipe.model.tokenizer_cache_key()
    (tmp_path / "data.json").write_text(json.dumps(draccus.encode(LmDataConfig(tokenizer=tokenizer))))
    plans = []
    monkeypatch.setattr("marin.experiment.cli._print_plan", plans.extend)
    arguments = (
        "--preset",
        "train-long-agentic",
        "--sft",
        MIXTURE_262K_32NODE.name,
        "--sft-data-mix",
        "tests/sft-only@2026.11.25",
        "--sft-data-uri",
        str(tmp_path),
        "--sft-data-file",
        "data.json",
        "--sft-data-tokenizer",
        tokenizer,
        "--sft-steps",
        "1679",
        "--through",
        "sft",
        "--version",
        "2026.11.25.7",
    )
    command = snowball.command(snowball_model.SNOWBALL, argv=arguments, sft_presets=lambda: (MIXTURE_262K_32NODE,))
    result = CliRunner().invoke(command, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    execution = recipe.execution_config(snowball_model.SNOWBALL.by_name["train-long-agentic"])
    environment = {key: "fixture" for key in execution.required_env_vars}
    _, request = _coordinator_request(
        plans, "experiments.post_training.snowball_rl", arguments, Path.cwd(), environment, execution=execution
    )
    for key, value in request.environment.env_vars.items():
        monkeypatch.setenv(key, value)
    built = []
    monkeypatch.setattr(snowball, "get_job_info", lambda: SimpleNamespace())
    monkeypatch.setattr(snowball, "build_artifacts", lambda *steps, **kwargs: built.extend(steps))
    result = CliRunner().invoke(command, (*arguments, "--run"))
    assert result.exit_code == 0, (result.output, result.exception)
    [export] = built
    assert export.name.startswith("checkpoints/snowball/")
    assert export.version == "2026.11.25.7"
    assert skyrl_api.skyrl_execution(export) is None
    assert all(skyrl_api.skyrl_execution(step) is None for step in graph_handles([export]))
    changed = tuple("1680" if value == "1679" else value for value in arguments)
    refused = CliRunner().invoke(command, (*changed, "--run"))
    assert refused.exit_code == 4, (refused.output, refused.exception)
    assert len(built) == 1


def test_data_mix_changes_get_separate_sft_outputs(tmp_path):
    source = ArtifactStep.adopt("tests/sft-mix", "2026.11.25.1", str(tmp_path))
    first = PreparedData(source, "v5.json", MIXTURE_262K_32NODE.recipe.model.tokenizer_cache_key())
    second = replace(first, step=replace(source, version="2026.11.25.2"), file="v7.json")
    spec = MIXTURE_262K_32NODE.recipe
    v5 = spec.with_data_mix(first).trained(MIXTURE_262K_32NODE.resources).step
    v7 = spec.with_data_mix(second).trained(MIXTURE_262K_32NODE.resources).step
    assert v5.path() != v7.path()


def test_curriculum_holdout_is_seeded_disjoint_and_records_its_population(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "outputs"))
    root = tmp_path / "curriculum"
    root.mkdir()
    tags = TaskStrata(family="nonagentic", source="fixture", aai_category="general").model_dump()
    rows = []
    for index in range(620):
        identifier = f"task-{index}"
        task = TaskSpec(
            id=identifier,
            source=Source(dataset="fixture", revision="1", row=identifier, importer_revision="1"),
            context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {index}."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=ScriptGrader(
                argv=("python", "/grade.py"),
                placement=GraderMachine(
                    environment=EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="numpy==2.3.5\n"
                    )
                ),
            ),
        )
        rows.append({TASK_SPEC_COLUMN: task.model_dump_json(), **tags})
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), root / TRAIN_FILENAME)
    (root / SUMMARY_FILENAME).write_text(
        json.dumps({"recipe": {"heldout": []}, "selection_strata": {"name": "curriculum", "version": "2026.11.25"}})
    )
    source = ArtifactStep.adopt("tests/curriculum", "2026.11.25", str(root))

    def ids(values):
        return {json.loads(row[TASK_SPEC_COLUMN])["id"] for row in values}

    results = []
    for seed in (17, 18):
        step = curriculum_holdout_step(source, TrainingHoldout(count=200, seed=seed))
        run(step)
        directory = Path(step.path())
        train = pq.read_table(directory / TRAIN_FILENAME).to_pylist()
        val = pq.read_table(directory / VALIDATION_FILENAME).to_pylist()
        assert len(train) == 420 and len(val) == 200
        assert ids(train).isdisjoint(ids(val))
        assert ids(train) | ids(val) == ids(rows)
        summary = json.loads((directory / SUMMARY_FILENAME).read_text())
        assert summary["training_holdout"]["count"] == 200
        assert summary["training_holdout"]["seed"] == seed
        assert summary["training_holdout"]["population"] == 620
        assert set(summary["validation_task_ids"]) == ids(val)
        results.append(ids(val))
    assert results[0] != results[1]
    reversed_root = tmp_path / "reversed"
    reversed_root.mkdir()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows)), schema=TASK_SCHEMA), reversed_root / TRAIN_FILENAME)
    (reversed_root / SUMMARY_FILENAME).write_bytes((root / SUMMARY_FILENAME).read_bytes())
    repeated = curriculum_holdout_step(
        ArtifactStep.adopt("tests/reversed", "2026.11.25", str(reversed_root)), TrainingHoldout(count=200, seed=17)
    )
    run(repeated)
    assert ids(pq.read_table(Path(repeated.path()) / VALIDATION_FILENAME).to_pylist()) == results[0]
    assert len(pq.read_table(root / TRAIN_FILENAME)) == 620


def test_best_checkpoint_uses_three_evaluation_mean_and_restores_window(tmp_path):
    result = native_json()(
        """
import asyncio,json,sys
from pathlib import Path
from types import SimpleNamespace
from skyrl_train.callbacks.base import TrainerControl,TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback
root=Path(sys.argv[1])
saved=[]
step=0
async def save_checkpoints(): saved.append(step)
trainer=SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(root),export_path=str(root/'exports'))),save_checkpoints=save_checkpoints)
params=dict(initial_model_uri='sft-final',expected_tasks=200,minimum_scored_tasks=198,selection_window=3)
callback=BestCheckpointCallback(**params)
async def main():
 global callback,step
 windows=[]
 for step,score in zip((3,6,9,12,15,18),(.9,.1,.2,.8,.8,.1)):
  if step==15: callback=BestCheckpointCallback(**params)
  metrics={'eval/all/num_attempted':200,'eval/all/num_scored':200,'eval/all/avg_verifier_score':score}
  await callback.on_evaluate_async(TrainerState(step,0,18,18),TrainerControl(),trainer=trainer,metrics=metrics)
  windows.append(metrics.get('eval/selection_score'))
 selected=json.loads((root/'best_checkpoint.json').read_text())
 print(json.dumps({'selected':selected,'saved':saved,'windows':windows,'export':json.loads((root/'exports/best_checkpoint.json').read_text())}))
asyncio.run(main())
""",
        [str(tmp_path)],
    )
    assert result["selected"]["step"] == 15
    assert result["selected"]["accuracy"] == pytest.approx(0.6)
    assert result["windows"][:2] == [None, None]
    assert result["windows"][2:] == pytest.approx([0.4, 1.1 / 3, 0.6, 1.7 / 3])
    assert result["saved"] == [9, 15]
    assert result["export"] == result["selected"]
