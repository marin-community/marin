# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from importlib.machinery import ModuleSpec
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
from skyrlrecipe.documents import merge_mappings, set_path

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_tests import model_entry_fixture
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
    write_artifact_record,
    write_published_artifact,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import native_json
from experiments.post_training.snowball_sft import OPENTHOUGHTS_32K_4NODE
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, VALIDATION_FILENAME

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def printed_status(output):
    # CliRunner merges stderr into result.output; status is on result.stdout.
    marker = output.rfind("\npreset:")
    return yaml.safe_load(output[marker + 1 if marker >= 0 else 0 :].split("\n# ", 1)[0])


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
        arguments = ["--preset", preset.name, "--version", preset.revision, "--plan"]
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
        if preset.name == "eval-short":
            status = printed_status(result.stdout.partition("\nStepSpec(")[0])
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
    assert gate.comparison_config(recorded) == gate.comparison_config(planned)


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


def test_checkpoint_cohort_coverage_controls_best_checkpoint(selection, tmp_path):
    table = snowball_model.SNOWBALL
    preset = table.by_name["train-long-checkpoint-eval"]
    spec = recipe.run_spec(preset, name="checkpoint-evaluation-coverage", model_name=table.model)
    step = recipe.run_step(preset, spec)
    path = tmp_path / "checkpoint-evaluation.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
    results = native_json()(
        """
import asyncio, json, sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from cloud.iris.launch_config import load_launch_config
from skyrl_train.callbacks.base import CallbackHandler, TrainerState, TrainerControl
from skyrl_train.callbacks.builtin import create_callbacks_from_config

path = Path(sys.argv[1])
config = load_launch_config(path)
assert not config.skyrl.generator.enable_http_endpoint
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
        [str(path)],
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


def test_mochi_three_rack_launch_evaluates_without_model_exports(selection, tmp_path, mochi_sft_presets):
    preset = MOCHI.by_name["train-long-3rack"]
    (step,) = snowball.build_experiment(preset, table=MOCHI).runs
    path = tmp_path / "mochi.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)

    (config,) = gate.compose_configs([str(path)], validate_driver=True)
    producer = mochi_sft_presets["periodic"].model()
    assert producer.relative_path is not None
    export = StoragePath(producer.step.path("s3://marin-us-east-02a/marin")) / producer.relative_path
    assert config["inputs"]["model"]["uri"] == str(export)

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


def test_speed_short_opt_out_preserves_other_generator_settings_and_inputs(selection, tmp_path):
    table = snowball_model.SNOWBALL
    preset = table.by_name["speed-short"]
    [default_run] = snowball.build_experiment(preset, table=table).runs
    [plain_run] = snowball.build_experiment(preset, table=table, settings=("generator.speculative_decoding=null",)).runs
    default = yaml.safe_load(materialized_config(default_run, str(tmp_path)).launch_config_yaml)
    plain = yaml.safe_load(materialized_config(plain_run, str(tmp_path)).launch_config_yaml)

    assert default_run.name != plain_run.name
    assert default["skyrl"]["generator"].pop("speculative_decoding")["method"] == "dspark"
    assert plain["skyrl"]["generator"].pop("speculative_decoding", None) is None
    assert default["skyrl"]["generator"] == plain["skyrl"]["generator"]
    assert default["inputs"] == plain["inputs"]


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


@pytest.mark.parametrize("recorded_runs", [2], indirect=True, ids=["one-task-per-family"])
def test_train_long_reenters_the_model_module_and_reports_checkpoint_evaluations(
    recorded_runs, preset_table, tmp_path, monkeypatch
):
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(
        preset,
        checkpoint_every=2,
        evaluations=(replace(preset.evaluations[0], every=2),),
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
    assert preset.held_out_tasks is not None
    training_selection = recipe.selection_artifact(preset.tasks)
    write_artifact_record(
        Path(training_selection.path()),
        training_selection.name,
        training_selection.version,
        Artifact,
        config={"seed": 42},
    )
    published_selections = Path(recipe.SELECTION_ROOT)

    def selection_files():
        return {
            path: (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in (published_selections, *published_selections.rglob("*"))
        }

    before_launch = selection_files()
    built = snowball.build_experiment(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=3",)
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    _json(source / "exports/best_checkpoint.json", {"step": 2})
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
            global_step=3,
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
        "trainer.max_steps=3",
    ]
    client, job = fake_iris(monkeypatch, "train-coordinator")
    mode, coordinator_module = "valid", "first_model_entry"
    entries = {}

    def invoke_entry(name, patch):
        patch.setattr(sys.modules["__main__"], "__spec__", ModuleSpec(name, loader=None))
        return model_entry_fixture.main(entries[name], arguments)

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
    client.job.return_value.wait.side_effect = lambda *args, **kwargs: SimpleNamespace(
        state=JobState.SUCCEEDED if mode == "valid" else JobState.FAILED
    )

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
            raise click.exceptions.Exit(invoke_entry("first_model_entry", driver))

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
        assert "best_checkpoint_step: 2" in result.output and f"{preset.evaluations[0].name}@3/scored" in result.output
        assert "http://127.0.0.1:55777/" in result.output and "test-initializer-credential" not in result.output
    assert len(native_calls) == 1
    assert selection_files() == before_launch
    assert published[-1].optimizer_steps == published[-1].planned_updates == 3
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
    module(coordinator_module, following=True)
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
            if row.step != 2
        ],
    )
    (source / "exports/best_checkpoint.json").unlink()
    with edit_receipt(source) as receipt:
        receipt["result"]["model"] = None
    step = built.runs[0]
    write_skyrl_run_record(source, step.name, step.version, fingerprint=step.fingerprint(), global_step=None)
    planning_args = ["--preset", preset.name, "--version", preset.revision, "--set", "trainer.max_steps=3"]
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

    mode, coordinator_module = "valid", "first_model_entry"
    module(coordinator_module)
    resumed = invoke_with_preflight_stub(entry)
    assert resumed.exit_code == 1 and len(native_calls) == 1, (resumed.output, resumed.exception)
    assert "best_checkpoint_step: none eligible" in resumed.output

    replace_evaluations(source, evaluations)

    live_preset = replace(preset, checkpoints="none", checkpoint_every=None, export="none")
    live = snowball.build_experiment(
        live_preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=3",)
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

    mode, coordinator_module = "failed-native", "first_model_entry"
    module("first_model_entry")
    with edit_receipt(source) as receipt:
        receipt["result"]["state"] = "failed"
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)
