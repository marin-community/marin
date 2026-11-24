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
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import AttemptRecord, read_attempt, read_evaluations, write_attempt
from fray.types import ANY_REGION
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, apply, artifact_identity, materialized_config
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, IrisSkyRLExecution, SkyRLRun, skyrl_execution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path
from skyrlrecipe.model_staging import model_metadata_subpath, tokenizer_metadata_subpath

from experiments.post_training import rl, rl_compose, rl_gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_measurements import read_run_inputs
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    edit_receipt,
    fake_iris,
    fake_publication,
    fake_s3,
    printed_status,
    replace_evaluations,
    shrink_recorded_panel,
    write_artifact_record,
    write_json,
    write_published_artifact,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.skyrl_test_support import native_json
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE


@pytest.fixture
def table(request):
    return {"snowball": SNOWBALL, "mochi": MOCHI}[request.param]


def plan(table, name, *options, version=None, sft_presets=None):
    preset = table.by_name[name]
    args = ("--preset", name, "--version", version or preset.revision, "--plan")
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
        "DAYTONA_TARGET",
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
            ("terminal_bench.installed_agent.agent.kwargs.unused=true",),
            None,
            "unused",
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
    ids=("installed-agent-kwargs", "disabled-route-capture"),
)
def test_plan_print_rejects_invalid_native_config(table, name, settings, version, error, selection):
    options = tuple(arg for setting in settings for arg in ("--set", setting))
    result = plan(table, name, *options, version=version)
    assert result.exit_code == 4, (result.output, result.exception)
    assert "native configuration cannot be composed" in result.output and error in result.output


@pytest.mark.parametrize(
    ("dataset", "harness_override", "exit_code"),
    [
        ("long-loop-smartie-swesmith", None, 0),
        ("fixture-unmapped", None, 4),
        ("long-loop-smartie-swesmith", "shellbox", 4),
    ],
)
def test_plan_requires_harness_grading_mapping(selection, dataset, harness_override, exit_code):
    summary_path = Path(selection.path()) / rl_compose.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["harness_graded_datasets"] = [dataset]
    summary_path.write_text(json.dumps(summary))
    options = (
        ("--set", f"environment.task_sessions.task_spec.harness_by_dataset.{dataset}={harness_override}")
        if harness_override
        else ()
    )
    result = plan(SNOWBALL, "train-medium-agentic", *options, version="dev")
    assert result.exit_code == exit_code, (result.output, result.exception)
    if exit_code:
        assert "harbor_trial in harness_by_dataset" in result.output and dataset in result.output


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
        write_published_artifact(tmp_path, model_name, preset.revision, Artifact, source=model_uri, config=tokenizer)
        monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/hf"])
        start_args = ("--init", f"{model_name}@{preset.revision}")
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
        changed = ArtifactStep.adopt("models/changed-default", preset.revision, "gs://fixture-models/changed-default")
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
        assert rl_gate.comparison_config(worker_document) == rl_gate.comparison_config(plan_document)

        model_path = f"/native-tmp/{model_metadata_subpath(source_model['uri'], source_model['identity'])}"
        for tokenizer_uri, revision in (
            (source_model["tokenizer_uri"], source_model["tokenizer_revision"]),
            (None, None),
        ):
            declared = deepcopy(plan_document)
            staged = deepcopy(worker_document)
            for document in (declared, staged):
                document["inputs"]["model"].update(tokenizer_uri=tokenizer_uri, tokenizer_revision=revision)
            tokenizer_path = model_path
            if tokenizer_uri:
                tokenizer_key = tokenizer_metadata_subpath(tokenizer_uri, revision)
                tokenizer_path = f"/native-tmp/{tokenizer_key}"
            for role in ("policy", "ref"):
                staged["skyrl"]["trainer"][role]["model"].update(
                    path=model_path, tokenizer_path=tokenizer_path, tokenizer_revision=None
                )
            assert rl_gate.comparison_config(staged) == rl_gate.comparison_config(declared)
            ref_tokenizer = "skyrl.trainer.ref.model.tokenizer_path"
            set_path(staged, ref_tokenizer, "/native-tmp/marinskyrl/model_metadata/" + "0" * 64)
            with pytest.raises(rl_gate.InvalidComparison, match=ref_tokenizer):
                rl_gate.comparison_config(staged)

        runtime["starting_checkpoint_identity"] = "different-checkpoint"
        with pytest.raises(rl_gate.InvalidComparison, match="starting_checkpoint_identity"):
            rl_gate.comparison_config(worker_document)
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
    monkeypatch.delitem(sys.modules, module)
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
    for _ in range(2):
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        assert exit_.value.code == 0
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


@pytest.mark.usefixtures("source_provenance")
@pytest.mark.timeout(180)
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
    shrink_recorded_panel(preset.held_out_tasks)
    training_selection = rl_compose.selection_artifact(preset.tasks)
    write_artifact_record(
        Path(training_selection.path()),
        training_selection.name,
        training_selection.version,
        Artifact,
        config={"seed": 42},
    )
    published_selections = Path(rl_compose.SELECTION_ROOT)

    def selection_files():
        return {
            path: (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in (published_selections, *published_selections.rglob("*"))
        }

    before_launch = selection_files()
    built = rl.build_experiment(
        preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=3",)
    )
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint(), inputs=built.inputs)
    write_json(source / "exports/best_checkpoint.json", {"step": 2})
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
                rl_gate, "compose_configs", lambda paths, **_: [{"runtime": {"training_type": "async"}} for _ in paths]
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
        rl.command(table, argv=tuple(planning_args), init=initializer, forward_env=("HF_TEST_KEY",)),
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
    live = rl.build_experiment(
        live_preset, table=table, init=initializer, forward_env=("HF_TEST_KEY",), settings=("trainer.max_steps=3",)
    )
    assert live.runs[0].name != built.runs[0].name
    with monkeypatch.context() as live_record:
        live_record.setattr(snowball_model, "SNOWBALL", replace(table, presets=(live_preset,)))
        recorded_runs(preset.name, live.runs[0].name, fingerprint=live.runs[0].fingerprint(), inputs=live.inputs)
    live_args = [*planning_args, "--checkpoints", "none", "--checkpoint-every", "0", "--export", "none"]
    result = invoke_with_preflight_stub(
        rl.command(table, argv=tuple(live_args), init=initializer, forward_env=("HF_TEST_KEY",)), live_args
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
