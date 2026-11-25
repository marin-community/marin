# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from skyrlrecipe.documents import merge_mappings

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    controls,
    fake_iris,
    fake_s3_checkpoint,
    write_published_artifact,
)
from experiments.post_training.skyrl_test_support import native_json
from experiments.post_training.snowball_sft import OPENTHOUGHTS_32K_4NODE
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, VALIDATION_FILENAME

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def test_preset_table_rejects_duplicate_names():
    table = snowball_model.SNOWBALL
    with pytest.raises(ValueError, match="Duplicate preset name 'eval-short'"):
        replace(table, presets=(*table.presets, table.by_name["eval-short"]))


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
