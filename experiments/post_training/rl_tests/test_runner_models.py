# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from importlib.machinery import ModuleSpec
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner
from fray.types import ANY_REGION
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, apply, artifact_identity, materialized_config
from marin.rl.skyrl import IrisSkyRLExecution, skyrl_execution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path

from experiments.post_training import mochi_rl, rl, rl_compose, rl_gate, snowball_rl
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_measurements import read_run_inputs
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_tests.fixtures import SELECTION_IDENTITY, write_published_artifact
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE


@pytest.fixture
def table(request):
    return {"snowball": SNOWBALL, "mochi": MOCHI}[request.param]


def plan(table, name, *options, version=None, sft_presets=None):
    preset = table.by_name[name]
    args = ("--preset", name, "--version", version or rl_compose.RUN_ARTIFACT_VERSION, "--plan")
    if preset.tasks_input == "required":
        args += ("--tasks", SELECTION_IDENTITY)
    args += options
    return CliRunner().invoke(rl.command(table, argv=args, sft_presets=sft_presets), args)


@pytest.mark.parametrize("cadence, period", [(None, None), (2, 4)])
def test_retention_composes_periodic_saves_and_evaluation(selection, tmp_path, cadence, period):
    preset = replace(
        MOCHI.by_name["train-long"], checkpoints="latest", checkpoint_every=cadence, keep_every=period, export="none"
    )
    spec = rl_compose.run_spec(preset, name="periodic-retention", model_name=MOCHI.model)
    path = tmp_path / "launch.yaml"
    path.write_text(materialized_config(rl_compose.run_step(preset, spec), str(tmp_path)).launch_config_yaml)
    (config,) = rl_gate.compose_configs([str(path)])
    trainer = config["skyrl"]["trainer"]
    assert (trainer["checkpoint_retention"], trainer.get("checkpoint_keep_every")) == ("latest", period)
    checkpoint, evaluation = trainer["callbacks"]
    assert (checkpoint["type"], evaluation["type"]) == ("checkpoint", "evaluation")
    assert checkpoint["save_steps"] == (cadence or preset.updates + 1)
    assert not config["run"]["export_hf"]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"checkpoints": "latest"}, "export must be none or match checkpoints"),
        ({"export": "none", "evaluations": ()}, "requires an evaluation"),
        ({"checkpoint_every": None}, 'checkpoints="best" requires checkpoint_every'),
        ({"checkpoints": "none", "export": "none"}, 'checkpoints="none" requires'),
        ({"keep_every": 0}, "keep_every must be positive"),
    ],
    ids=[
        "export-mismatch",
        "best-without-evaluation",
        "best-without-cadence",
        "none-with-cadence",
        "zero-keep-every",
    ],
)
def test_checkpoint_retention_refuses_incompatible_presets(changes, error):
    base = replace(MOCHI.by_name["train-long"], checkpoints="best", checkpoint_every=2, export="best")
    with pytest.raises(ValueError, match=error):
        replace(base, **changes)


def test_mixed_candidate_uses_the_selected_tasks_and_preserves_authored_concurrency(selection, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    preset = SNOWBALL.by_name["train-medium-agentic"]
    settings = ("context_budget.max_new_tokens_per_turn=24576", "trainer.train_batch_size=128")
    run = rl.build_experiment(preset, table=SNOWBALL, tasks=selection, settings=settings)
    assert run.inputs is not None
    config = rl_gate.validate_candidate(preset, run.inputs, table=SNOWBALL)["skyrl"]
    buffer = config["trainer"]["rollout_buffer"]
    assert (buffer["max_in_flight"], buffer["groups_per_family"]) == (32, {"agentic": 64, "nonagentic": 64})
    (step,) = run.runs
    execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
    assert set(execution.required_env_vars) == {
        "DAYTONA_RL_API_KEY",
        "DAYTONA_EVAL_API_KEY",
        "WANDB_API_KEY",
    }


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("environment.task_sessions.daytona.ttl_minutes", 60, "ttl_minutes"),
        ("trainer.algorithm.group_admission.stall_timeout", 9000, "stall_timeout"),
        ("trainer.placement.policy_num_nodes", 10, "recipe settings are derived by the runner"),
        ("trainer.hf_export_checkpoint", "best", "recipe settings are derived by the runner"),
        ("trainer.rollout_buffer.groups_per_family", {"agentic": 32}, "recipe settings are derived by the runner"),
    ],
)
def test_recipe_refuses_invalid_authored_settings(selection, path, value, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    patch = {}
    set_path(patch, path, value)
    with pytest.raises(ValueError, match=error):
        config = preset.recipe.config.merge(RecipePatch.from_document(patch))
        preset = replace(preset, recipe=replace(preset.recipe, config=config))
        rl_compose.run_spec(preset, name="recipe-test", tasks=selection)


@pytest.mark.parametrize(
    ("shares", "error"),
    [
        ({"agentic": Fraction(1, 4), "nonagentic": Fraction(1, 4)}, "sum to 1"),
        ({"agentic": Fraction(1, 3), "nonagentic": Fraction(2, 3)}, "whole number of groups"),
        ({"agentic": Fraction(1, 2), "math": Fraction(1, 2)}, "training selection has"),
    ],
)
def test_batch_mix_refuses_invalid_counts(selection, shares, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    with pytest.raises((ValueError, click.BadParameter), match=error):
        rl_compose.run_spec(replace(preset, batch_mix=BatchMix(shares)), name="mix-test", tasks=selection)


@pytest.mark.parametrize(
    ("table", "name", "settings", "version", "error"),
    [
        (
            "snowball",
            "train-medium-agentic",
            ("environment.task_sessions.agent_harnesses.pi.unused=true",),
            None,
            None,
        ),
        (
            "snowball",
            "train-medium-agentic",
            ("generator.engine_init_kwargs.enable_return_routed_experts=false",),
            "dev",
            "moe_router_replay requires generator.engine_init_kwargs.enable_return_routed_experts=true",
        ),
    ],
    indirect=["table"],
    ids=("native-agent-option", "disabled-route-capture"),
)
def test_plan_print_rejects_invalid_native_config(table, name, settings, version, error, selection):
    options = tuple(arg for setting in settings for arg in ("--set", setting))
    result = plan(table, name, *options, version=version)
    assert result.exit_code == 4, (result.output, result.exception)
    issue = json.loads(result.stdout)
    assert issue["verdict"] == "invalid"
    if error is not None:
        assert error in issue["reason"]


@pytest.mark.parametrize(
    ("dataset", "settings", "error"),
    [
        ("fixture-unmapped", (), "Unsupported training datasets with harness graders: fixture-unmapped"),
        (
            "long-loop-smartie-swesmith",
            ("environment.task_sessions.task_spec.harness_by_dataset.long-loop-smartie-swesmith=shellbox",),
            "SWESmith training requires the Pi harness and a task machine",
        ),
    ],
)
def test_plan_refuses_a_wrapper_selection_without_native_conversion(selection, dataset, settings, error):
    summary_path = Path(selection.path()) / rl_compose.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["harness_graded_datasets"] = [dataset]
    summary_path.write_text(json.dumps(summary))
    preset = SNOWBALL.by_name["train-medium-agentic"]
    with pytest.raises(click.BadParameter, match=error):
        rl_compose.run_spec(preset, name="rejected-wrapper-selection", tasks=selection, settings=settings)


def test_plan_binds_the_converted_swesmith_selection(selection, tmp_path):
    summary_path = Path(selection.path()) / rl_compose.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["harness_graded_datasets"] = ["long-loop-smartie-swesmith"]
    summary_path.write_text(json.dumps(summary))
    preset = SNOWBALL.by_name["train-medium-agentic"]
    spec = rl_compose.run_spec(preset, name="converted-wrapper-selection", tasks=selection)
    config = yaml.safe_load(materialized_config(rl_compose.run_step(preset, spec), str(tmp_path)).launch_config_yaml)
    (training,) = config["inputs"]["train_data"]
    assert training["identity"].startswith(f"{selection.name}-pi@")
    assert training["relative_path"] == "train.parquet"


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
    (root / "hf").mkdir(parents=True)
    (root / "hf" / "config.json").write_text("{}")
    model = replace(model, step=ArtifactStep.adopt(model.step.name, model.step.version, str(root)))
    return replace(MOCHI, presets=(replace(preset, model=model),))


@pytest.mark.parametrize(
    ("table", "explicit_init"),
    [("snowball", False), ("snowball", True), ("mochi", True)],
    indirect=["table"],
    ids=["recorded-default", "snowball-init", "mochi-init"],
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
    start_args = ()
    if explicit_init:
        write_published_artifact(
            tmp_path, model_name, rl_compose.RUN_ARTIFACT_VERSION, Artifact, source=model_uri, config=tokenizer
        )
        monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/hf"])
        start_args = ("--init", f"{model_name}@{rl_compose.RUN_ARTIFACT_VERSION}")
    runs = []

    def after_rl(run) -> Artifact:
        return Artifact(path=run.path)

    def collect(step):
        runs.append(step)
        return apply("tests/after-rl", after_rl, version="2026.11.19.1", run=step)

    preset = replace(preset, then=collect)
    table = replace(table, presets=(preset,))
    started = plan(table, preset.name, *start_args)
    assert started.exit_code == 0, (started.output, started.exception)
    (source,) = runs
    source_config = materialized_config(source, str(tmp_path))
    source_launch = yaml.safe_load(source_config.launch_config_yaml)
    source_model = source_launch["inputs"]["model"]
    if explicit_init:
        assert source_model["uri"] == f"{model_uri}/hf"
    else:
        changed = ArtifactStep.adopt(
            "models/changed-default", rl_compose.RUN_ARTIFACT_VERSION, "gs://fixture-models/changed-default"
        )
        table = replace(table, presets=(replace(preset, model=replace(preset.model, step=changed)),))
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
    planned = plan(table, preset.name, *resume_args)
    assert planned.exit_code == 0, (planned.output, planned.exception)
    assert f"  model: {source_model['identity']}\n" in planned.output
    assert f"  starting_checkpoint: {source.name}/resume/step-4@{source.version}:" in planned.output
    _, resumed = runs
    resumed_config = materialized_config(resumed, str(resume_prefix))
    inputs = yaml.safe_load(resumed_config.launch_config_yaml)["inputs"]
    assert inputs["model"] == source_model
    assert inputs["starting_checkpoint"]["uri"] == str(checkpoint)
    if not explicit_init:
        launch = tmp_path / "resumed.yaml"
        launch.write_text(resumed_config.launch_config_yaml)
        (plan_document,) = rl_gate.compose_configs([str(launch)])
        worker_document = deepcopy(plan_document)
        runtime = worker_document["skyrl"].setdefault("runtime", {})
        runtime["model_source_identity"] = inputs["model"]["identity"]
        runtime["starting_checkpoint_identity"] = inputs["starting_checkpoint"]["identity"]
        assert rl_gate.config_version(worker_document) == rl_gate.config_version(plan_document)
        runtime["starting_checkpoint_identity"] = "different-checkpoint"
        with pytest.raises(rl_gate.InvalidComparison, match="starting_checkpoint_identity"):
            rl_gate.config_version(worker_document)
        return
    assert all(dep.name != model_name for dep in resumed.lower().deps)
    monkeypatch.setattr(rl, "run_skyrl", lambda _config, **_kwargs: None)
    resumed.run(resumed_config)
    inputs = read_run_inputs(str(Path(resumed_config.output.resolved_config_uri).parent))
    assert inputs.options is not None
    assert (inputs.options.checkpoints, inputs.options.checkpoint_every, inputs.options.export) == (
        ("latest", None, "none") if table.model == "mochi" else ("best", 10, "best")
    )
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


@pytest.mark.usefixtures("source_provenance")
@pytest.mark.parametrize(
    ("model", "override"),
    [("snowball", True), ("mochi", False), ("mochi", True)],
    ids=("snowball", "mochi-ttl", "mochi-durable"),
)
def test_sft_composition_reenters_its_own_module(selection, monkeypatch, model, override, request):
    module = f"experiments.post_training.{model}_rl"
    if model == "snowball":
        sft = MIXTURE_262K_32NODE
    else:
        sft = request.getfixturevalue("mochi_sft_presets")["bounded" if override else "periodic"]
    argv = (
        "--preset",
        "train-long",
        "--sft",
        sft.name,
        "--version",
        "2026.11.19.1",
        "--run",
        "--checkpoints",
        "latest",
        "--checkpoint-every",
        "5",
        "--export",
        "latest",
        "--priority",
        "production",
    )
    durable = "s3://marin-us-east-08a/marin"
    temporary = "s3://marin-us-east-08a/tmp/ttl=3d"
    prefix = durable if override else temporary
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    if model == "mochi" and override:
        git_output = rl.git_output

        def checkout_output(args, *other, **kwargs):
            return "f" * 40 if args == ["rev-parse", "HEAD"] else git_output(args, *other, **kwargs)

        monkeypatch.setattr(rl, "git_output", checkout_output)
    if override:
        monkeypatch.setenv("MARIN_PREFIX", durable)
    else:
        monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: durable)
    monkeypatch.setattr(rl, "_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", [module, *argv])
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", ModuleSpec(module, loader=None))
    submitted = []

    def submit(entry_module, handles, _max_concurrent, env, *, argv):
        assert entry_module == module
        assert env["MARIN_PREFIX"] == prefix
        (run,) = handles
        preset = (MOCHI if model == "mochi" else SNOWBALL).by_name["train-long"]
        execution = skyrl_execution(run)
        assert execution is not None
        layout = preset.recipe.layout
        sft_hours = 20 if model == "snowball" else 0
        assert execution.coordinator_timeout_hours == layout.coordinator_timeout_hours + sft_hours
        assert execution.job_timeout_seconds == layout.job_timeout_seconds
        initializer = sft.model(
            coordinator_cluster=preset.recipe.layout.cluster,
            coordinator_prefix=prefix if model == "mochi" else None,
        )
        producer = next(step for step in run.deps if step.name == initializer.step.name)
        assert producer.fingerprint() == initializer.step.fingerprint()
        training = (
            producer
            if model == "snowball"
            else next(dependency for dependency in producer.deps if "train_resources" in dependency.runtime_args)
        )
        resources = training.runtime_args["train_resources"]
        assert resources.target_cluster is None
        assert tuple(resources.regions or ()) == ((ANY_REGION,) if model == "mochi" else ())
        config = materialized_config(run, env["MARIN_PREFIX"])
        document = yaml.safe_load(config.launch_config_yaml)
        assert document["inputs"]["model"]["identity"] == artifact_identity(initializer.step)
        assert document["inputs"]["model"]["uri"] == f"{producer.path(prefix)}/{initializer.relative_path}"
        assert document["inputs"]["model"]["tokenizer_uri"] is None
        assert document["iris"]["priority"] == "production"
        assert document["run"]["export_hf"]
        trainer = document["skyrl"]["trainer"]
        assert trainer["checkpoint_retention"] == "latest"
        assert trainer["callbacks"][0] == {"type": "checkpoint", "save_steps": 5}
        assert "best_checkpoint" not in [callback["type"] for callback in trainer["callbacks"]]
        if model == "mochi":
            assert document["artifacts"]["checkpoint_root"] == f"{run.path(prefix)}/checkpoints"
            assert document["artifacts"]["export_root"] == f"{run.path(prefix)}/exports"
            if override:
                assert training.output_path_prefix == producer.output_path_prefix == prefix
                standalone = sft.recipe.trained(sft.resources, prefix=prefix)
                standalone_revision = materialized_config(standalone.step, prefix)["source_revision"]
                assert env.get("GIT_COMMIT") == standalone_revision
                assert producer.fingerprint() == standalone.step.fingerprint()
                from experiments.grug.moe_hero_ep import sft_recipe  # noqa: PLC0415 - only the Mochi case loads SFT

                with monkeypatch.context() as coordinator:
                    coordinator.setenv("GIT_COMMIT", env["GIT_COMMIT"])
                    coordinator.setattr(sft_recipe, "get_git_commit", lambda: pytest.fail("coordinator has no checkout"))
                    replay = sft.model(coordinator_cluster=preset.recipe.layout.cluster, coordinator_prefix=prefix)
                assert replay.step.fingerprint() == producer.fingerprint()
        submitted.append((run.name, argv))
        return JobName.from_wire("/fixture/sft-then-rl")

    monkeypatch.setattr(rl, "submit_or_run", submit)
    entry = snowball_rl.main if model == "snowball" else mochi_rl.main
    for _ in range(2):
        assert entry() == 0
    assert submitted == [(submitted[0][0], argv)] * 2


@pytest.mark.parametrize(
    ("table", "name", "options", "error", "mode"),
    [
        ("mochi", "train-long", ("--checkpoints", "best"), 'checkpoints="best" requires checkpoint_every', None),
        ("mochi", "train-long", (), "must use s3 storage", "cross-backend"),
        (
            "mochi",
            "train-long",
            ("--resume", "checkpoints/previous@2026.11.19.1"),
            "--resume requires checkpointing",
            None,
        ),
        (
            "snowball",
            "train-long",
            ("--resume", "checkpoints/previous@2026.11.19.1"),
            "source run's retention period",
            "expired",
        ),
        ("snowball", "train-long", ("--resume", "run@1", "--init", "model@1"), "cannot be combined with --init", None),
        (
            "snowball",
            "learning-medium",
            ("--control", "--priority", "production"),
            "controls use only the preset's pinned run options",
            None,
        ),
        ("snowball", "train-long", ("--checkpoint-every", "3"), "fall on checkpoint steps", None),
        ("snowball", "train-long", ("--sft", "unknown", "--init", "model@1"), "--sft requires a fresh candidate", None),
        ("snowball", "train-long", ("--sft", "unknown", "--resume", "run@1"), "--sft requires a fresh candidate", None),
        ("snowball", "learning-medium", ("--checkpoints", "latest"), "gated presets do not accept", None),
        ("snowball", "learning-medium", ("--checkpoint-every", "1"), "gated presets do not accept", None),
        ("snowball", "learning-medium", ("--export", "none"), "gated presets do not accept", None),
        ("snowball", "learning-medium", ("--sft", "unknown"), "uses default-model controls", None),
        ("snowball", "speed-short", ("--sft", "unknown"), "does not accept --sft", None),
        (
            "snowball",
            "train-medium-agentic",
            ("--set", "trainer.hf_export_checkpoint=best"),
            "owned by the preset",
            None,
        ),
        (
            "snowball",
            "train-medium-agentic",
            ("--set", "trainer.placement.policy_num_nodes=10"),
            "owned by the preset",
            None,
        ),
        (
            "snowball",
            "train-medium-agentic",
            ("--set", 'trainer.rollout_buffer.groups_per_family={"agentic":32}'),
            "owned by the preset",
            None,
        ),
    ],
    indirect=["table"],
)
def test_command_refusals(selection, table, name, options, error, mode, request, monkeypatch):
    if mode == "expired":
        request.getfixturevalue("stored_checkpoint")
    if mode == "cross-backend":
        monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: "s3://marin-us-east-02a/marin")
        monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central2/fixture")
        monkeypatch.setattr(rl, "check_cache", lambda _path: True)
        monkeypatch.setattr(rl, "read_skyrl_attempt", lambda _path: None)
        storage_exists = StoragePath.exists

        def check_storage(path):
            if path.scheme == "gs":
                raise AssertionError("planning read from an incompatible storage backend")
            return storage_exists(path)

        monkeypatch.setattr(StoragePath, "exists", check_storage)
    result = plan(
        table,
        name,
        *options,
        sft_presets=lambda: pytest.fail("loaded refused SFT presets"),
    )
    assert result.exit_code == 4 and error in result.output, (result.output, result.exception)


def test_mochi_agentic_plan_collects_attempts_with_exports_disabled(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "check_cache", lambda _path: True)
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    scratch = tmp_path / "temporary"
    monkeypatch.setattr(rl_compose, "marin_temp_bucket", lambda *_args, **_kwargs: str(scratch))
    result = plan(MOCHI, "train-medium-agentic")
    assert result.exit_code == 0, (result.output, result.exception)
    assert "rl-data/task-health/attempts/" in result.output
    assert f"  root: {scratch}\n" in result.output
    assert "  source: preset's 3-day storage\n" in result.output
    assert "coordinator time limit exceeds storage lifetime" in result.output
