# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import click
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import (
    AttemptRecord,
    RunIndexIdentity,
    read_attempt,
    read_evaluations,
    write_attempt,
)
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.cli import RUN_INDEX_ENV
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from skyrlrecipe.documents import merge_mappings

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_index
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
    FakeIris,
    FakeMarina,
    append_metrics,
    cli,
    cohort_from_fixtures,
    compose_steps,
    controls,
    copy_evaluations,
    fake_s3,
    fake_s3_checkpoint,
    printed_status,
    replace_evaluations,
    set_receipt_fields,
    write_hf_export,
    write_published_artifact,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import read_skyrl_metrics, run_native_test
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, VALIDATION_FILENAME

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


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

    mochi = MOCHI.by_name["train-long-3rack"]
    [mochi_step] = snowball.build_experiment(mochi, table=MOCHI).runs
    launch_path = tmp_path / f"{mochi.name}.yaml"
    launch_path.write_text(materialized_config(mochi_step, str(tmp_path)).launch_config_yaml)
    launch_paths[mochi.name] = str(launch_path)
    composed = gate.compose_configs(list(launch_paths.values()), validate_driver=True)
    configs = dict(zip(launch_paths, composed, strict=True))
    for name in ("speed-short", "data-short", "eval-short"):
        assert configs[name]["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"

    for name, key in (("speed-short", "evaluation_data"), ("learning-medium", "validation_data")):
        held_out = recipe.selection_artifact(table.by_name[name].held_out_tasks)
        sources = configs[name]["inputs"][key]
        (source,) = sources["eval_lite"] if key == "evaluation_data" else sources
        assert (source["uri"], source["relative_path"]) == (held_out.path(), FULL_VALIDATION_FILENAME)
    callbacks = [row["type"] for row in configs["learning-medium"]["skyrl"]["trainer"]["callbacks"]]
    assert callbacks.index("evaluation") < callbacks.index("best_checkpoint") < callbacks.index("hf_model_save")

    # Published train-long controls still contain defaults removed by the TaskSpec migration.
    planned = configs["train-long"]
    recorded = deepcopy(planned)
    recorded["skyrl"]["data"].update(task_cache_dir="~/.cache/marinskyrl/tasks", terminal_bench_data=[])
    del recorded["skyrl"]["environment"]["task_sessions"]["agent_harnesses"]
    del recorded["skyrl"]["trajectory_runner"]["max_concurrent_training_machine_tasks"]
    assert gate.config_version(recorded) == gate.config_version(planned)

    mochi = MOCHI.by_name["train-long-3rack"]
    config = configs[mochi.name]
    assert config["skyrl"]["generator"].get("speculative_decoding") is None
    model = config["inputs"]["model"]
    assert model["uri"] == MOCHI_STEP_400407_EXPORT
    assert model["tokenizer_uri"] == MOCHI_TOKENIZER_URI
    assert model["tokenizer_revision"] == MOCHI_TOKENIZER_REVISION

    (source,) = config["inputs"]["validation_data"]
    held_out = recipe.selection_artifact(mochi.held_out_tasks)
    assert (source["uri"], source["relative_path"]) == (held_out.path(), VALIDATION_FILENAME)
    assert not config["run"]["export_hf"]
    callbacks = {row["type"] for row in config["skyrl"]["trainer"]["callbacks"]}
    assert "evaluation" in callbacks
    assert callbacks.isdisjoint({"checkpoint", "best_checkpoint", "hf_model_save"})

    docs = (recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
    section = docs.split("`mochi-inference.yaml`", 1)[1]
    fragment = yaml.safe_load(section.split("```yaml\n", 1)[1].split("```", 1)[0])
    assert merge_mappings(config["skyrl"], fragment) == config["skyrl"]


@pytest.mark.parametrize(("mode", "exit_code"), [("--plan", 0), ("--run", 4)])
def test_insufficient_controls_restates_the_exact_base_version(recorded_runs, mode, exit_code, monkeypatch):
    table = snowball_model.SNOWBALL
    expected_version = "missing-base-version"
    monkeypatch.setattr(snowball, "_base_preset", lambda *_: ("commit-one", expected_version))

    preset = table.by_name["learning-medium"]
    result = cli(table, preset.name, mode, entry_module=snowball_model.__name__)
    assert result.exit_code == exit_code, (result.output, result.exception)
    if mode == "--plan":
        status = printed_status(result.stdout.partition("\nStepSpec(")[0])
        assert len(status["runs"]) == 1
        assert status["preset_version"]
    else:
        assert f"at base version {expected_version}: 0/5 completed main controls" in result.output


@pytest.mark.parametrize(
    "mode, problem, reason",
    [
        ("--plan", "batch", "lcm_dp_size=8"),
        ("--run", "batch", "lcm_dp_size=8"),
        ("--plan", "cpu", "rollout"),
    ],
)
def test_launcher_refuses_invalid_driver_settings(selection, preset_table, monkeypatch, mode, problem, reason):
    iris = FakeIris(monkeypatch)
    preset = snowball_model.SNOWBALL.by_name["train-long"]
    layout, settings = preset.recipe.layout, (
        "trainer.policy_mini_batch_size=2",
        "generator.n_samples_per_prompt=8",
        "trainer.train_batch_size=2",
    )
    if problem == "cpu":
        layout = replace(layout, host_cpu=32)
        settings = (
            "trajectory_runner.rollout_workers.num_workers=5",
            "trajectory_runner.rollout_workers.cpus_per_worker=8",
        )
    recipe_config = preset.recipe.config.with_settings(settings)
    table = preset_table(replace(preset, recipe=replace(preset.recipe, layout=layout, config=recipe_config)))
    result = cli(table, preset.name, mode, entry_module=snowball_model.__name__)
    assert result.exit_code == 4 and reason in result.output, (result.output, result.exception)
    if problem == "batch":
        assert json.loads(result.stdout)["verdict"] == "invalid"
    assert not iris.requests


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def export_model(output):
        write_hf_export(Path(output) / "hf")
        return Artifact(path=output)

    producer = apply("tests/computed-init", export_model, version="2026.11.17", output=OUT)
    train = snowball_model.SNOWBALL.by_name["train-long"]
    initializer = ArtifactHfModel(producer)
    [planned] = snowball.build_experiment(train, init=initializer, table=snowball_model.SNOWBALL).runs
    assert producer in planned.deps
    planned.lower()
    assert not Path(producer.path()).exists()
    run(producer)
    config = yaml.safe_load(materialized_config(planned, str(tmp_path)).launch_config_yaml)
    best = next(row for row in config["skyrl"]["trainer"]["callbacks"] if row["type"] == "best_checkpoint")
    assert best["initial_model_uri"] == config["inputs"]["model"]["uri"] == str(Path(producer.path()) / "hf")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    checkpoint = StoragePath("s3://marin-test/models/checkpoint")
    export = checkpoint / "hf"
    checkpoint_dir = tmp_path / "model"
    write_hf_export(checkpoint_dir / "hf")
    fake_s3_checkpoint(monkeypatch, str(checkpoint), checkpoint_dir)
    initializer = ArtifactStep.adopt("tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint)

    speed = snowball_model.SNOWBALL.by_name["speed-short"]
    steps = (
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
    configs = compose_steps(tmp_path, steps)
    assert configs[0]["inputs"]["model"]["uri"] == str(export)
    assert configs[0]["iris"]["wandb_entity"] == "test-team"
    assert (configs[0]["inputs"]["model"]["tokenizer_uri"], configs[0]["inputs"]["model"]["tokenizer_revision"]) == (
        "tests/own-tokenizer",
        "tokenizer-revision",
    )
    callbacks = configs[1]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[1]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[1]["skyrl"]["trainer"]["max_prompt_length"] == 24576
    assert (
        configs[2]["inputs"]["validation_data"][0]["identity"]
        .partition(":")[0]
        .endswith("@" + recipe.selection_artifact(speed.tasks).version)
    )


def test_checkpoint_cohort_coverage_controls_best_checkpoint(selection, tmp_path):
    table = snowball_model.SNOWBALL
    preset = table.by_name["train-long-checkpoint-eval"]
    spec = recipe.run_spec(preset, name="checkpoint-evaluation-coverage", model_name=table.model)
    step = recipe.run_step(preset, spec)
    path = tmp_path / "checkpoint-evaluation.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
    results = run_native_test("cloud.iris.tests.native_checkpoint_coverage", [str(path)])
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
    write_hf_export(root / "hf")
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

    if table is snowball_model.SNOWBALL:
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
    table = snowball_model.SNOWBALL
    paths = (*repeated.runs[:2], run(learning, "trainer.policy.optimizer_config.lr=2e-6"))
    provenance = replace(Provenance.capture(), branch=None)
    configs = compose_steps(tmp_path, paths, prefix=recipe.artifact_prefix(learning.recipe.layout.cluster))
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
    original = deepcopy(summary["recipe"])
    for fault, message in (("heldout", "different development panel"), ("exclusions", "full held-out selection")):
        summary["recipe"] = original | {fault: [{"version": "other"}] if fault == "heldout" else []}
        _json(summary_path, summary)
        with pytest.raises(click.BadParameter, match=message):
            name(data, tasks=selection)
    summary["recipe"] = original | {
        "exclusions": [{"uri": str(new_task_set / FULL_VALIDATION_FILENAME), "version": "2026.11.05"}]
    }
    _json(summary_path, summary)
    for preset in (data, snowball_model.SNOWBALL.by_name["data-medium"]):
        assert name(replace(preset, held_out_tasks=f"{task_set_name}@2026.11.05"), tasks=selection)


@pytest.mark.parametrize("case, expected", [("guard", 1), ("missing", 3), ("native-failed", 2), ("data-short", 0)])
def test_wait_reports_native_and_gate_exit_codes(
    recorded_runs, selection, monkeypatch, preset_table, cache_native_compositions, case, expected
):
    table = snowball_model.SNOWBALL
    preset = table.by_name["data-short" if case == "data-short" else "speed-medium"]
    preset_table(preset)
    tasks = selection if case == "data-short" else None
    if preset.gated:
        controls(recorded_runs, preset.name)
    experiment = snowball.build_experiment(preset, table=table, tasks=tasks)
    step = experiment.runs[0]
    iris = FakeIris(monkeypatch, name="dashboard-coordinator")
    job = JobName.root(iris.owner, iris.name)
    source = recorded_runs(
        preset.name,
        step.name,
        inputs=experiment.inputs,
        fingerprint=step.fingerprint(),
        coordinator=str(job),
        state="failed" if case == "native-failed" else "succeeded",
        speed=150 if case == "guard" else 101,
    )
    if preset.gated:
        cohort = cohort_from_fixtures(source)
        monkeypatch.setattr(snowball, "_base_preset", lambda *_: ("commit-one", cohort.version))
    if case in {"guard", "data-short"} and preset.gated:
        build_gate = gate.gate_step(
            preset, experiment.runs, inputs=experiment.inputs, cohort=cohort, group_name=experiment.group_name
        )

        def complete():
            run(gate.completed_gate(build_gate))
            return JobState.SUCCEEDED

        iris.state_for_wait = complete
    else:
        iris.state_for_wait = lambda: JobState.FAILED if case == "native-failed" else JobState.SUCCEEDED
    rows = FakeMarina(monkeypatch, initial=rl_index.list_attempts()).history
    args = ["--preset", preset.name, "--version", recipe.RUN_ARTIFACT_VERSION, "--run", "--wait"]
    if tasks:
        args += ["--tasks", SELECTION_IDENTITY]
    result = CliRunner().invoke(snowball.command(table, argv=tuple(args), entry_module=snowball_model.__name__), args)
    assert result.exit_code == expected, (result.output, result.exception)
    if case in {"data-short", "guard"}:
        [row] = rows
        assert row.optimizer_steps == (0 if case == "data-short" else preset.updates)
        if case == "data-short":
            assert row.evaluation_data_identity
        else:
            assert row.gate_result.verdict == "fail"


@pytest.mark.parametrize("interrupted", [False, True], ids=["completed", "interrupted"])
def test_detached_ungated_launch_indexes_prepared_then_terminal_attempt(
    recorded_runs, preset_table, monkeypatch, interrupted
):
    nominal = snowball_model.SNOWBALL.by_name["train-long"]
    preset = replace(nominal.with_updates(2), checkpoints="none", export="none", checkpoint_every=None, evaluations=())
    table = preset_table(preset)
    module = "experiments.post_training.rl_tests.archive_preset_fixture"
    monkeypatch.setenv("MARIN_RL_TEST_SELECTION_ROOT", recipe.SELECTION_ROOT)
    git_output = snowball.git_output
    source_ref, tree = [git_output(["rev-parse", arg], cwd=recipe.REPO_ROOT) for arg in ("HEAD", "HEAD^{tree}")]
    monkeypatch.setattr(
        Provenance, "capture", lambda *a, **kw: Provenance("tree-one", source_ref, False, "branch", "test")
    )
    monkeypatch.setattr(
        snowball, "git_output", lambda args, **kw: tree if args[0] == "merge-base" else git_output(args, **kw)
    )
    experiment = snowball.build_experiment(preset, table=table)
    source = recorded_runs(
        preset.name,
        experiment.runs[0].name,
        inputs=experiment.inputs,
        attempt_id="native-old",
        retry_attempt="foreign-attempt",
    )
    receipt = json.loads((source / "terminal.json").read_text())
    launch = receipt["config"]
    if interrupted:
        base_version = "fixture-base"
        monkeypatch.setattr(snowball, "_base_preset", lambda *_: (tree, base_version))
    else:
        base_version = gate.config_version(gate.compose_preset(nominal))
    iris = FakeIris(monkeypatch, name="index-coordinator")
    job = JobName.root(iris.owner, iris.name)
    monkeypatch.setattr(snowball, "_PUBLICATION_POLL_SECONDS", 0)
    publications, followers = FakeMarina(monkeypatch).history, []

    def run_local(argv, **kwargs):
        if argv[0] == "gh":
            return subprocess.CompletedProcess(argv, 0, "124")
        if argv[:2] == ["git", "archive"]:
            assert argv[2] == tree
        if argv[1:4] == ["-m", module, "_preset-config"]:
            assert argv[4:] == [preset.name]
            assert (kwargs["cwd"] / "vendor/skyrl/cloud/iris/config_cli.py").is_file()
            assert str(kwargs["cwd"]) in kwargs["env"]["PYTHONPATH"]
        return subprocess.run(argv, **kwargs)

    monkeypatch.setattr(
        snowball,
        "subprocess",
        SimpleNamespace(
            run=run_local,
            SubprocessError=subprocess.SubprocessError,
            DEVNULL=subprocess.DEVNULL,
            STDOUT=subprocess.STDOUT,
            Popen=lambda argv, **kw: followers.append(argv[3:]),
        ),
    )
    attempt = "native-indexed"
    export = launch["artifacts"]["export_root"]

    def prepare(job, kwargs):
        identity = json.loads(kwargs["environment"].env_vars[RUN_INDEX_ENV])
        index = RunIndexIdentity.model_validate(identity[launch["run"]["id"]])
        config = deepcopy(launch)
        config["run"].update(attempt_id=attempt, coordinator_iris_job_id=str(job))
        write_attempt(
            export,
            AttemptRecord(
                run_id=config["run"]["id"],
                attempt_id=attempt,
                config=config,
                index=index,
                metrics_started_at=receipt["result"]["started_at"],
            ),
        )

    def finish():
        if not publications:
            return JobState.RUNNING
        if interrupted:
            return JobState.FAILED
        launch["run"].update(attempt_id=attempt, coordinator_iris_job_id=str(job))
        write_receipt(source, receipt)
        metrics = read_skyrl_metrics(str(source), attempt_id="native-old")
        for row in metrics.values():
            row["attempt_id"] = attempt
        append_metrics(source, metrics)
        write_attempt(
            export, AttemptRecord(run_id=launch["run"]["id"], attempt_id=attempt, elapsed_excluding_eval_seconds=123.0)
        )
        return JobState.SUCCEEDED

    iris.on_submit, iris.state_for_wait = prepare, finish
    result = cli(table, preset.name, "--run", entry_module=module)
    assert result.exit_code == 0, (result.output, result.exception)
    assert not publications
    [follower] = followers
    with monkeypatch.context() as timeout:
        clock = iter((0.0, 1_000_000.0))
        timeout.setattr(snowball, "time", SimpleNamespace(monotonic=lambda: next(clock)))
        timeout.setattr(iris, "state_for_wait", lambda: JobState.RUNNING)
        assert snowball.main(table, argv=follower) == 3
    assert read_attempt(export, attempt).interruption is None
    assert [row.status for row in publications] == ["running"]
    publications.clear()
    assert snowball.main(table, argv=follower) == int(interrupted)
    running, finished = publications
    assert running.status == "running"
    if interrupted:
        stored = read_attempt(export, attempt)
        assert stored.result is None and stored.interruption == "coordinator_terminated"
        assert finished.status == "failed" and finished.failed_facts == ["coordinator_interrupted"]
        return
    assert finished.status == "succeeded"
    for row in publications:
        assert (row.attempt_id, row.run_id, row.pr_number) == (attempt, finished.run_id, 124)
        assert (row.base_preset_version, row.base_commit) == (base_version, tree)
        assert row.source_revision == f"base={source_ref} tree=tree-one"
    assert finished.preset_version == gate.config_version(launch) != base_version
    assert finished.artifact_uri == str(source) and finished.elapsed_excluding_eval_seconds == 123
    assert parse_qs(urlsplit(finished.grafana_url).query)["var-job"] == [receipt["result"]["iris_job_id"]]


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
    write_attempt(str(source / "exports"), earlier.model_copy(update={"issue": 73}))
    staged = source.with_name("before-native-execution")
    source.rename(staged)
    write_attempt(str(source / "exports"), AttemptRecord(run_id=earlier.run_id, attempt_id="earlier", issue=73))
    native_calls = []

    def native(command, *, env_vars=None):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        stored = json.loads((source / "terminal.json").read_text())
        response = stored["result"]
        response["model"] = {
            "global_step": 3,
            "policy_export_uri": str(source / "policy"),
            "tokenizer_uri": "tests/python-tokenizer",
            "tokenizer_revision": "python-tokenizer-revision",
            "checkpoint_root": str(source / "checkpoints"),
            "terminal_manifest_uri": str(source / "terminal.json"),
        }
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
    published = FakeMarina(monkeypatch).history
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
    iris = FakeIris(monkeypatch, name="train-coordinator")
    default_module = snowball_model.__name__
    second_entry = "experiments.post_training.mochi_rl"
    mode, coordinator_module = "valid", default_module
    entries = {}

    def invoke_entry(name, patch):
        module_spec = find_spec(name)
        assert module_spec is not None and module_spec.loader is not None
        patch.setattr(sys.modules["__main__"], "__spec__", module_spec)
        return model_entry_fixture.main(entries[name], arguments)

    def submit(job, kwargs):
        command = kwargs["entrypoint"].command
        assert command == ["python", "-m", coordinator_module, *arguments]
        environment = kwargs["environment"].env_vars
        if mode != "overlapping-environment-mismatch":
            assert environment["HF_TEST_KEY"] == "test-initializer-credential"
        assert "UNRELATED_SECRET" not in environment
        if mode == "failed-native":
            return
        with monkeypatch.context() as child:
            for key, value in environment.items():
                child.setenv(key, value)
            task = f"{job}/0"
            child.setenv("IRIS_TASK_ID", task)
            child.setattr(
                "experiments.post_training.rl_publication.publish_result",
                lambda *args: pytest.fail("Coordinator contacted Marina"),
            )
            info = JobInfo(task_id=JobName.from_wire(task))
            with iris_ctx_scope(IrisContext.from_job_info(info)):
                code = invoke_entry(coordinator_module, child)
            assert code == {"valid": 0, "failed-following": 3}.get(mode, 4)
        set_job_info(None)

    iris.on_submit = submit
    iris.state_for_wait = lambda: JobState.SUCCEEDED if mode == "valid" else JobState.FAILED

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
    assert result.exit_code == 3 and "post_run_error:" in result.output
    assert "RL has a committed result" in result.output

    completed_id = published[-1].attempt_id
    assert completed_id != earlier.attempt_id
    evaluations = read_evaluations(str(source / "exports"), attempt_id=completed_id, evaluation="eval/")
    replace_evaluations(source, [row for row in evaluations if row.step != 2])
    (source / "exports/best_checkpoint.json").unlink()
    receipt = json.loads((source / "terminal.json").read_text())
    set_receipt_fields(receipt, {"result.model": None})
    write_receipt(source, receipt)
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

    mode, coordinator_module = "valid", default_module
    module(coordinator_module)
    resumed = invoke_with_preflight_stub(entry)
    assert resumed.exit_code == 1 and len(native_calls) == 1, (resumed.output, resumed.exception)
    assert "best_checkpoint_step: none eligible" in resumed.output

    mode, coordinator_module = "failed-native", default_module
    module(default_module)
    set_receipt_fields(receipt, {"result.state": "failed"})
    write_receipt(source, receipt)
    result = invoke_with_preflight_stub(entry)
    assert result.exit_code == 3 and published[-1].status == "failed", (result.output, result.exception)
