# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import click
import pytest
import yaml
from fsspec.implementations.memory import MemoryFileSystem
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import skyrl_execution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl, rl_compose, rl_gate
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_measurements import read_run_inputs
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    FakeIris,
    cli,
    fake_s3,
    printed_status,
    write_published_artifact,
)
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME


@pytest.fixture
def table(request):
    return {"snowball": SNOWBALL, "mochi": MOCHI}[request.param]


def plan(table, name, *options, **kwargs):
    if table.by_name[name].tasks_input == "required":
        options = ("--tasks", SELECTION_IDENTITY, *options)
    return cli(table, name, "--plan", *options, **kwargs)


def test_agentic_plan_binds_native_tasks_and_keeps_attempts(selection, tmp_path, monkeypatch):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    settings = ("context_budget.max_new_tokens_per_turn=24576", "trainer.train_batch_size=128")
    run = rl.build_experiment(preset, table=SNOWBALL, tasks=selection, settings=settings)
    config = rl_gate.compose_preset(preset, run.inputs)
    buffer = config["skyrl"]["trainer"]["rollout_buffer"]
    assert (buffer["max_in_flight"], buffer["groups_per_family"]) == (32, {"agentic": 64, "nonagentic": 64})
    (step,) = run.runs
    assert set(skyrl_execution(step).required_env_vars) == {
        "DAYTONA_RL_API_KEY",
        "DAYTONA_EVAL_API_KEY",
        "WANDB_API_KEY",
    }
    summary_path = Path(selection.path()) / rl_compose.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["harness_graded_datasets"] = ["long-loop-smartie-swesmith"]
    summary_path.write_text(json.dumps(summary))
    [converted] = rl.build_experiment(preset, table=SNOWBALL, tasks=selection).runs
    document = yaml.safe_load(materialized_config(converted, str(tmp_path)).launch_config_yaml)
    (training,) = document["inputs"]["train_data"]
    assert training["identity"].startswith(f"{selection.name}-pi@")
    assert training["relative_path"] == "train.parquet"
    summary["harness_graded_datasets"] = []
    summary_path.write_text(json.dumps(summary))

    # Mochi exposes the per-run task attempts even when its temporary run has no policy export.
    preset = MOCHI.by_name["train-medium-agentic"]
    marker = StoragePath(preset.model.step.path()) / MODEL_MANIFEST_FILENAME
    fake_s3(monkeypatch, str(marker).removeprefix("s3://"))
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    monkeypatch.setattr(rl_compose, "marin_temp_bucket", lambda *_args, **_kwargs: str(tmp_path))
    result = plan(MOCHI, preset.name)
    assert result.exit_code == 0, (result.output, result.exception)
    [status] = printed_status(result.stdout)["runs"]
    assert f"rl-data/task-health/attempts/by-run/{status['name']}@{status['version']}" in result.stdout
    [mochi] = rl.build_experiment(preset, table=MOCHI, tasks=selection).runs
    assert not yaml.safe_load(materialized_config(mochi, str(tmp_path)).launch_config_yaml)["run"]["export_hf"]


@pytest.mark.parametrize(
    "change, error",
    [
        (BatchMix({"agentic": Fraction(1, 3), "nonagentic": Fraction(2, 3)}), "whole number of groups"),
        (BatchMix({"agentic": Fraction(1, 2), "math": Fraction(1, 2)}), "training selection has"),
        ("fixture-unmapped", "Unsupported training datasets"),
        ("shellbox", "SWESmith training requires the Pi harness and a task machine"),
    ],
)
def test_agentic_plan_refuses_invalid_selection_and_batch_mix(selection, change, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    settings = ()
    if isinstance(change, BatchMix):
        preset = replace(preset, batch_mix=change)
    else:
        summary_path = Path(selection.path()) / rl_compose.SUMMARY_FILENAME
        summary = json.loads(summary_path.read_text())
        summary["harness_graded_datasets"] = ["long-loop-smartie-swesmith" if change == "shellbox" else change]
        summary_path.write_text(json.dumps(summary))
        if change == "shellbox":
            settings = ("environment.task_sessions.task_spec.harness_by_dataset.long-loop-smartie-swesmith=shellbox",)
    with pytest.raises((ValueError, click.BadParameter), match=error):
        rl_compose.run_spec(preset, name="trial", tasks=selection, settings=settings)


@pytest.fixture
def stored_checkpoint(tmp_path):
    root = tmp_path / "checkpoints" / "previous" / "2026.11.19.1"
    return write_native_checkpoint(root, {})


def write_native_checkpoint(root, launch_document):
    checkpoint = root / "scratch" / "global_step_4"
    checkpoint.mkdir(parents=True)
    launch_document.setdefault("artifacts", {})["checkpoint_root"] = str(checkpoint.parent)
    (root / RESOLVED_LAUNCH_FILENAME).write_text(yaml.safe_dump({"config": launch_document}))
    (checkpoint.parent / "latest_ckpt_global_step.txt").write_text("4")
    return checkpoint


@pytest.fixture
def mochi_with_incomplete_default(tmp_path):
    preset = MOCHI.by_name["train-long"]
    model = preset.model
    root = tmp_path / "model"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    model = replace(model, step=ArtifactStep.adopt(model.step.name, model.step.version, str(root)))
    return replace(MOCHI, presets=(replace(preset, model=model),))


@pytest.mark.parametrize(
    ("table", "explicit_init"),
    [("snowball", False), ("mochi", True)],
    indirect=["table"],
    ids=["recorded-default", "mochi-init"],
)
def test_resume_inherits_the_source_runs_initializer(
    selection, table, explicit_init, tmp_path, monkeypatch, mochi_with_incomplete_default
):
    if table is MOCHI:
        monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
        preset = replace(mochi_with_incomplete_default.by_name["train-long"], checkpoints="latest")
        table = replace(mochi_with_incomplete_default, presets=(preset,))
    preset = table.by_name["train-long"]
    model_name = "models/alternative"
    model_uri = "gs://fixture-models/alternative"
    tokenizer = {"tokenizer_uri": "gs://fixture-models/tokenizer", "tokenizer_revision": "tokenizer-pin"}
    init = None
    if explicit_init:
        version = rl_compose.RUN_ARTIFACT_VERSION
        write_published_artifact(tmp_path, model_name, version, Artifact, source=model_uri, config=tokenizer)
        monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/hf"])
        init = rl_compose.published_model(f"{model_name}@{version}", preset=preset)
    [source] = rl.build_experiment(preset, table=table, init=init).runs
    source_config = materialized_config(source, str(tmp_path))
    source_launch = yaml.safe_load(source_config.launch_config_yaml)
    source_model = source_launch["inputs"]["model"]
    if not explicit_init:
        changed = ArtifactStep.adopt("models/changed-default", "2026.11.20.1", "gs://fixture-models/changed-default")
        preset = replace(preset, model=replace(preset.model, step=changed))
    runs = []

    def collect(step):
        runs.append(step)
        return step

    table = replace(table, presets=(replace(preset, then=collect),))
    root = Path(source_config.output.resolved_config_uri).parent
    checkpoint = write_native_checkpoint(root, source_launch)
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/newer-hf"])
    resume_prefix = tmp_path
    if table.model == "mochi":
        resume_prefix = tmp_path / "temporary"
        # A partial scratch artifact must not hide the launch receipt in durable storage.
        (resume_prefix / source.name / source.version).mkdir(parents=True)
        monkeypatch.delenv("MARIN_PREFIX")
        monkeypatch.setattr(rl_compose, "marin_temp_bucket", lambda *_args, **_kwargs: str(resume_prefix))

    resume_args = ("--resume", f"{source.name}@{source.version}")
    composed = []
    if not explicit_init:
        compose_preset = rl_gate.compose_preset

        def record_composition(preset, inputs):
            config = compose_preset(preset, inputs)
            composed.append(config)
            return config

        monkeypatch.setattr(rl_gate, "compose_preset", record_composition)
    planned = plan(table, preset.name, *resume_args)
    assert planned.exit_code == 0, (planned.output, planned.exception)
    assert f"  model: {source_model['identity']}\n" in planned.output
    assert f"  starting_checkpoint: {source.name}/resume/step-4@{source.version}:" in planned.output
    [resumed] = runs
    resumed_config = materialized_config(resumed, str(resume_prefix))
    inputs = yaml.safe_load(resumed_config.launch_config_yaml)["inputs"]
    assert inputs["model"] == source_model
    assert inputs["starting_checkpoint"]["uri"] == str(checkpoint)
    if not explicit_init:
        (planned_config,) = composed
        worker = deepcopy(planned_config)
        runtime = worker["skyrl"].setdefault("runtime", {})
        runtime.update(
            model_source_identity=inputs["model"]["identity"],
            starting_checkpoint_identity=inputs["starting_checkpoint"]["identity"],
        )
        assert rl_gate.config_version(worker) == rl_gate.config_version(planned_config)
        runtime["starting_checkpoint_identity"] = "different-checkpoint"
        with pytest.raises(rl_gate.InvalidComparison, match="starting_checkpoint_identity"):
            rl_gate.config_version(worker)
        return
    assert all(dep.name != model_name for dep in resumed.lower().deps)
    monkeypatch.setattr(rl, "run_skyrl", lambda _config, **_kwargs: None)
    resumed.run(resumed_config)
    inputs = read_run_inputs(str(Path(resumed_config.output.resolved_config_uri).parent))
    assert {key: getattr(inputs.resume.model, key) for key in tokenizer} == tokenizer
    assert (inputs.resume.artifact_name, inputs.resume.artifact_version, inputs.resume.step) == (
        source.name,
        source.version,
        4,
    )


@pytest.mark.parametrize("source", (None, "same-root", "temporary"), ids=("fresh", "resume", "resume-temp-source"))
def test_mochi_plan_refuses_an_incomplete_default_export(
    selection, stored_checkpoint, source, mochi_with_incomplete_default, tmp_path, monkeypatch
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    table = mochi_with_incomplete_default
    preset = table.by_name["train-long"]
    model = preset.model
    args = ()
    if source is not None:
        (stored_checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
        args += ("--resume", "checkpoints/previous@2026.11.19.1", "--checkpoints", "latest")
    if source == "temporary":
        monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "destination"))
        monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: str(tmp_path / "durable"))
        monkeypatch.setattr(rl_compose, "marin_temp_bucket", lambda *_args, **_kwargs: str(tmp_path))
    result = plan(table, preset.name, *args, sft_presets=lambda: pytest.fail("loaded unused SFT presets"))
    assert result.exit_code == 4, (result.output, result.exception)
    assert f"{model.step.name}@{model.step.version}" in result.output
    assert "--sft PRESET" in result.output and "--init NAME@VERSION" in result.output


@pytest.mark.usefixtures("source_provenance", "cache_native_compositions")
@pytest.mark.parametrize("model, durable", [("snowball", True), ("mochi", False), ("mochi", True)])
def test_sft_composition_reenters_its_own_module(selection, monkeypatch, model, durable, request, tmp_path):
    table = MOCHI if model == "mochi" else SNOWBALL
    preset = table.by_name["train-long"]
    sft = (
        request.getfixturevalue("mochi_sft_presets")["bounded" if durable else "periodic"]
        if model == "mochi"
        else MIXTURE_262K_32NODE
    )
    module = f"experiments.post_training.{model}_rl"
    options = (
        "--preset train-long --version 2026.11.19.1 --run --checkpoints latest --checkpoint-every 5 --export latest"
    )
    argv = [*shlex.split(options), "--sft", sft.name, "--priority", "production"]
    published, scratch = "s3://marin-us-east-08a/marin", "s3://marin-us-east-08a/tmp/ttl=3d"
    prefix = published if durable else scratch
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setenv("MARIN_TEMP_PREFIX", scratch)
    monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: published)
    if durable:
        monkeypatch.setenv("MARIN_PREFIX", published)
    else:
        monkeypatch.delenv("MARIN_PREFIX", raising=False)
    fake_s3(monkeypatch)
    monkeypatch.setattr("rigging.filesystem.buckets._s3_filesystem", lambda *_args: MemoryFileSystem())
    monkeypatch.setattr(rl, "_follow_detached", lambda *_args: str(tmp_path / "follower.log"))
    graph = []

    def capture(step):
        graph.append(step)
        return step

    def observe_submit(_job, submission):
        assert submission["entrypoint"].command == ["python", "-m", module, *argv]
        environment = submission["environment"].env_vars
        assert environment.get("MARIN_PREFIX") == (prefix if model == "mochi" else None)
        run = graph[-1]
        layout = preset.recipe.layout
        assert (
            submission["timeout"].to_seconds()
            == (layout.coordinator_timeout_hours + (20 if model == "snowball" else 0)) * 3600
        )
        init = sft.model(coordinator_cluster=layout.cluster, coordinator_prefix=prefix if model == "mochi" else None)
        producer = next(step for step in run.deps if step.name == init.step.name)
        assert producer.fingerprint() == init.step.fingerprint()
        config = yaml.safe_load(materialized_config(run, prefix).launch_config_yaml)
        assert config["inputs"]["model"]["uri"] == f"{producer.path(prefix)}/{init.relative_path}"
        assert config["skyrl"]["trainer"]["callbacks"][0] == {"type": "checkpoint", "save_steps": 5}
        if model == "mochi" and durable:
            standalone = sft.recipe.trained(sft.resources, prefix=prefix).step
            assert environment["GIT_COMMIT"] == materialized_config(standalone, prefix)["source_revision"]
            assert producer.fingerprint() == standalone.fingerprint()
            from experiments.grug.moe_hero_ep import sft_recipe  # noqa: PLC0415

            with monkeypatch.context() as coordinator:
                coordinator.setenv("GIT_COMMIT", environment["GIT_COMMIT"])
                coordinator.setattr(sft_recipe, "get_git_commit", lambda: pytest.fail("coordinator has no checkout"))
                replay = sft.model(coordinator_cluster=layout.cluster, coordinator_prefix=prefix)
            assert replay.step.fingerprint() == producer.fingerprint()

    FakeIris(monkeypatch, on_submit=observe_submit)
    table = replace(table, presets=(replace(preset, then=capture),))
    assert rl.main(table, argv=argv, entry_module=module, sft_presets=lambda: (sft,)) == 0
    assert len(graph) == 1


@pytest.mark.parametrize(
    "model, name, options, error",
    [
        ("mochi", "train-long", "", "must use s3 storage"),
        ("mochi", "train-long", "--resume checkpoints/previous@2026.11.19.1", "--resume requires checkpointing"),
        ("snowball", "train-long", "--resume checkpoints/previous@2026.11.19.1", "source run's retention period"),
        ("snowball", "train-long", "--resume run@1 --init model@1", "cannot be combined with --init"),
        ("snowball", "learning-medium", "--control test-batch --priority production", "default run options"),
        ("snowball", "train-long", "--checkpoint-every 3", "fall on checkpoint steps"),
        ("snowball", "learning-medium", "--checkpoints latest", "gated presets do not accept"),
        ("snowball", "learning-medium", "--sft unknown", "uses default-model controls"),
        ("snowball", "speed-medium", "--wait", "--wait"),
        ("snowball", "speed-short", "--init fixture-model@2026.11.17", "does not accept --init"),
        ("snowball", "data-short", "--init fixture-model@2026.11.17", "does not accept --init"),
        ("snowball", "learning-medium", "", "AI-judge"),
        ("snowball", "train-long", "--init model@1", "Python init and --init"),
        ("snowball", "train-long", "compare", "Python init"),
        ("snowball", "speed-medium", "--set trainer.hf_hub_repo_id=org/model", "use the preset's declared owned keys"),
    ],
)
def test_command_refusals(selection, model, name, options, error, request, monkeypatch, capsys, tmp_path):
    table = MOCHI if model == "mochi" else SNOWBALL
    if "retention period" in error:
        request.getfixturevalue("stored_checkpoint")
    if error == "does not accept --init":
        write_published_artifact(tmp_path, "fixture-model", "2026.11.17", Artifact, source=str(tmp_path / "model"))
    if error == "AI-judge":
        summary = Path(rl_compose.selection_artifact(table.by_name[name].tasks).path()) / rl_compose.SUMMARY_FILENAME
        content = json.loads(summary.read_text())
        content["grader_kind_counts"]["train"]["remote_judge"] = 1
        summary.write_text(json.dumps(content))
    if options == "compare":
        assert rl.main(table, argv=["compare"], init=table.by_name[name].model) == 4
        assert error in capsys.readouterr().err
        return
    if model == "mochi" and not options:
        monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: "s3://marin-us-east-02a/marin")
        monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central2/fixture")
        monkeypatch.setattr(rl, "read_skyrl_attempt", lambda _path: None)
        storage_exists = StoragePath.exists

        def check_storage(path):
            assert path.scheme != "gs", "planning read from an incompatible storage backend"
            return storage_exists(path)

        monkeypatch.setattr(StoragePath, "exists", check_storage)
    init = table.by_name[name].model if error == "Python init and --init" else None
    result = plan(
        table, name, *shlex.split(options), init=init, sft_presets=lambda: pytest.fail("loaded refused SFT presets")
    )
    assert result.exit_code == 4 and error in result.output, (result.output, result.exception)
