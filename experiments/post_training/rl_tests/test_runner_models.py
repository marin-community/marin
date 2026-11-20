# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner
from fray.types import ANY_REGION
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity, materialized_config
from marin.rl.skyrl import ArtifactHfModel, IrisSkyRLExecution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path

from experiments.post_training import rl, rl_compose, rl_gate
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_measurements import read_run_inputs
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_tests.fixtures import SELECTION_IDENTITY, write_published_artifact
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE


@pytest.fixture
def table(request):
    return {"snowball": SNOWBALL, "mochi": MOCHI}[request.param]


def plan(table, name, *options, version=None, then=None, sft_presets=None):
    preset = table.by_name[name]
    args = ("--preset", name, "--version", version or preset.revision, "--plan")
    if preset.tasks_input == "required":
        args += ("--tasks", SELECTION_IDENTITY)
    args += options
    return CliRunner().invoke(rl.command(table, argv=args, then=then, sft_presets=sft_presets), args)


@pytest.fixture
def mochi_sft_preset(monkeypatch):
    from experiments.grug.moe_hero_ep import launch_hero_sft, sft_recipe  # noqa: PLC0415
    from experiments.post_training.mochi_sft import MIXTURE_262K_176NODE  # noqa: PLC0415

    monkeypatch.setattr(sft_recipe, "checkpoint_step", lambda _path: 393491)
    monkeypatch.setattr(launch_hero_sft, "special_token_init", lambda _step: None)
    return MIXTURE_262K_176NODE


@pytest.mark.parametrize("mode, cadence, period", [("none", None, None), ("latest", None, None), ("latest", 2, 4)])
def test_retention_composes_periodic_saves_and_evaluation(selection, tmp_path, mode, cadence, period):
    preset = replace(
        MOCHI.by_name["train-long"], checkpoints=mode, checkpoint_every=cadence, keep_every=period, export="none"
    )
    spec = rl_compose.run_spec(preset, name="periodic-retention", model_name=MOCHI.model)
    path = tmp_path / "launch.yaml"
    path.write_text(materialized_config(rl_compose.run_step(preset, spec), str(tmp_path)).launch_config_yaml)
    (config,) = rl_gate.compose_configs([str(path)])
    trainer = config["skyrl"]["trainer"]
    assert (trainer.get("checkpoint_retention", "none"), trainer.get("checkpoint_keep_every")) == (mode, period)
    assert (trainer["ckpt_interval"], trainer["max_ckpts_to_keep"], trainer["hf_export_checkpoint"]) == (0, 2, "latest")
    callbacks = ["checkpoint", "evaluation"] if mode == "latest" else ["evaluation"]
    assert [callback["type"] for callback in trainer["callbacks"]] == callbacks
    if mode == "latest":
        assert trainer["callbacks"][0]["save_steps"] == (cadence or 10_001)
    assert not config["run"]["export_hf"]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"checkpoints": "latest"}, "export must be none or match checkpoints"),
        ({"export": "none", "evaluations": ()}, "requires an evaluation"),
        ({"checkpoint_every": None}, 'checkpoints="best" requires checkpoint_every'),
        ({"checkpoints": "none", "export": "none"}, 'checkpoints="none" requires'),
        ({"keep_every": 0}, "keep_every must be positive"),
        ({"keep_every": -1}, "keep_every must be positive"),
    ],
    ids=[
        "export-mismatch",
        "best-without-evaluation",
        "best-without-cadence",
        "none-with-cadence",
        "zero-keep-every",
        "negative-keep-every",
    ],
)
def test_checkpoint_retention_refuses_incompatible_presets(changes, error):
    base = replace(MOCHI.by_name["train-long"], checkpoints="best", checkpoint_every=2, export="best")
    with pytest.raises(ValueError, match=error):
        replace(base, **changes)


def test_run_options_use_preset_checkpoint_validation():
    result = plan(MOCHI, "train-long", "--checkpoints", "best")
    assert result.exit_code == 4 and 'checkpoints="best" requires checkpoint_every' in result.output


@pytest.mark.parametrize(
    ("table", "revisions"),
    [
        (
            "mochi",
            {
                "train-medium": {
                    "2026.11.19.1": "8b5a970e",
                    "2026.11.20.1": "674dadce",
                    "2026.11.20.2": "15b0f019",
                    "2026.11.20.3": "83cf1ff6",
                    "2026.11.20.4": "08ba7390",
                },
                "train-long": {
                    "2026.11.19.1": "db854d78",
                    "2026.11.20.1": "b6144722",
                    "2026.11.20.2": "d1f75f26",
                    "2026.11.20.3": "491fdfff",
                    "2026.11.20.4": "e105d46c",
                },
                "train-long-3rack": {
                    "2026.11.19.1": "a0e17c94",
                    "2026.11.20.1": "bc381480",
                    "2026.11.20.2": "bc381480",
                    "2026.11.20.3": "4c67c254",
                    "2026.11.20.4": "9734d621",
                },
                "train-medium-agentic": {
                    "2026.11.19.1": "3f203da5",
                    "2026.11.20.1": "5f0a0fa1",
                    "2026.11.20.2": "d25249f4",
                    "2026.11.20.3": "56f7ec8d",
                    "2026.11.20.4": "1617b1d3",
                },
                "train-long-agentic": {
                    "2026.11.19.1": "bef4849f",
                    "2026.11.20.1": "9c4ea508",
                    "2026.11.20.2": "335aff0c",
                    "2026.11.20.3": "9f1754cd",
                    "2026.11.20.4": "f29b9e5f",
                    "2026.11.20.5": "86d2efb3",
                    "2026.11.20.6": "a3e2befe",
                },
                "train-long-agentic-3rack": {
                    "2026.11.19.1": "87a6ba8f",
                    "2026.11.20.1": "15da2779",
                    "2026.11.20.2": "0fcb3537",
                    "2026.11.20.3": "fbfd8b6a",
                    "2026.11.20.4": "e1eba83b",
                    "2026.11.20.5": "13b9c0ac",
                    "2026.11.20.6": "b8f3bba7",
                },
            },
        ),
        (
            "snowball",
            {
                "train-medium-agentic": {
                    "2026.11.19.1": "756f58ef",
                    "2026.11.20.1": "b557a4f0",
                    "2026.11.20.3": "e7e10480",
                    "2026.11.20.4": "f76d6b51",
                },
                "train-long-agentic": {
                    "2026.11.19.1": "72ec851f",
                    "2026.11.20.1": "6335c530",
                    "2026.11.20.3": "1cab698a",
                    "2026.11.20.4": "7123c960",
                },
            },
        ),
    ],
    indirect=["table"],
    ids=("mochi", "snowball-agentic"),
)
def test_new_presets_match_revisioned_native_configs(table, revisions, tmp_path, monkeypatch, request):
    if table.model == "mochi":
        produced = request.getfixturevalue("mochi_sft_preset").model()
        for preset in table.presets:
            model, prefix = preset.model, rl_compose.artifact_prefix(preset.recipe.layout.cluster)
            assert isinstance(model, ArtifactHfModel)
            assert (model.step.name, model.step.version, model.step.path(prefix)) == (
                produced.step.name,
                produced.step.version,
                produced.step.path(prefix),
            )
            assert model.relative_path == produced.relative_path and model.tokenizer_uri is None

    selection = request.getfixturevalue("selection")
    configs = assert_revisioned_native_configs(
        table, revisions, tmp_path=tmp_path, monkeypatch=monkeypatch, mixed_tasks=selection
    )
    if table.model == "mochi":
        documents = dict(zip(revisions, configs, strict=True))
        for preset in table.presets:
            document = documents[preset.name]
            assert not document["run"]["export_hf"]
            assert not any(
                callback["type"] in {"checkpoint", "best_checkpoint", "hf_model_save"}
                for callback in document["skyrl"]["trainer"]["callbacks"]
            )
            if preset.evaluations:
                (evaluation,) = preset.evaluations
                (validation,) = documents[preset.name]["inputs"]["validation_data"]
                assert (evaluation.name, validation["relative_path"]) == ("eval_lite_100", "validation.parquet")
        docs = (rl_compose.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
        fragment = docs.split("## Mochi inference on GB200", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
        served = rl_gate._config_values(documents["train-long-3rack"]["skyrl"])
        for path, value in rl_gate._config_values(yaml.safe_load(fragment)).items():
            assert served[path] == value, path


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
    ("case", "value", "error"),
    [
        ("recipe", ("environment.task_sessions.daytona.ttl_minutes", 60), "ttl_minutes"),
        ("recipe", ("trainer.algorithm.group_admission.stall_timeout", 9000), "stall_timeout"),
        ("owner", ("trainer.hf_export_checkpoint", "best"), "owned by the preset"),
        pytest.param(
            "owner",
            ("trainer.placement.policy_num_nodes", 10),
            "owned by the preset",
            id="authored-placement",
        ),
        ("owner", ("trainer.rollout_buffer.groups_per_family", {"agentic": 32}), "owned by the preset"),
        ("mix", {"agentic": Fraction(1, 4), "nonagentic": Fraction(1, 4)}, "sum to 1"),
        ("mix", {"agentic": Fraction(1, 3), "nonagentic": Fraction(2, 3)}, "whole number of groups"),
        ("mix", {"agentic": Fraction(1, 2), "math": Fraction(1, 2)}, "training selection has"),
        ("evaluation", 3, "fall on checkpoint steps"),
        ("prefix", None, "storage prefix"),
        ("gated", None, "already built"),
    ],
)
def test_runner_refuses_incompatible_launches(selection, tmp_path, monkeypatch, case, value, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    if case == "owner":
        key, setting = value
        with pytest.raises(click.BadParameter, match=error):
            rl_compose.run_spec(
                preset, name="settings-test", tasks=selection, settings=(f"{key}={json.dumps(setting)}",)
            )
        error = "recipe settings are derived by the runner"
    with pytest.raises((ValueError, click.BadParameter), match=error):
        if case in ("recipe", "owner"):
            key, setting = value
            patch = {}
            set_path(patch, key, setting)
            config = preset.recipe.config.merge(RecipePatch.from_document(patch))
            rl_compose.run_spec(
                replace(preset, recipe=replace(preset.recipe, config=config)), name="recipe-test", tasks=selection
            )
        elif case == "mix":
            rl_compose.run_spec(replace(preset, batch_mix=BatchMix(value)), name="mix-test", tasks=selection)
        elif case == "evaluation":
            replace(preset, evaluations=(replace(preset.evaluations[0], every=value),))
        else:

            def export_artifact(path):
                return Artifact(path=path)

            monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "other") if case == "prefix" else str(tmp_path))
            init = ArtifactHfModel(apply("models/not-built", export_artifact, version="2026.11.19", path=OUT))
            preset = SNOWBALL.by_name["learning-medium" if case == "gated" else "train-long"]
            rl.build_experiment(preset, table=SNOWBALL, init=init)


@pytest.mark.parametrize(
    ("table", "name", "settings", "version", "error"),
    [
        (
            "mochi",
            "train-medium",
            (
                "trainer.policy.megatron_config.tensor_model_parallel_size=2",
                # With TP=2, CP=2 keeps TP * PP * CP at 48 policy GPUs.
                "trainer.policy.megatron_config.context_parallel_size=2",
                # Expert-block weight sync requires TP=1.
                "generator.weight_sync_transport=broadcast",
            ),
            None,
            "MuonH fused-matrix updates require tensor parallel size 1",
        ),
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
    ids=("muonh", "installed-agent-kwargs", "disabled-route-capture"),
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


def test_resume_pins_the_native_checkpoint(selection, tmp_path, stored_checkpoint):
    preset = SNOWBALL.by_name["train-long"]
    args = ("--resume", "checkpoints/previous@2026.11.19.1")
    expired = plan(SNOWBALL, preset.name, *args)
    assert expired.exit_code == 4 and "source run's retention period" in expired.output
    (stored_checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    runs = []
    planned = plan(SNOWBALL, preset.name, *args, then=lambda model: runs.append(model.step) or ())
    assert planned.exit_code == 0 and "-resume-step-4" in planned.output, (planned.output, planned.exception)
    assert f"  model: {artifact_identity(preset.model.step)}\n" in planned.output
    launch = materialized_config(runs[0], str(tmp_path)).launch_config_yaml
    inputs = yaml.safe_load(launch)["inputs"]
    assert inputs["model"]["uri"] == preset.model.step.path(str(tmp_path))
    assert inputs["starting_checkpoint"]["uri"] == str(stored_checkpoint)
    assert "step-4@2026.11.19.1" in inputs["starting_checkpoint"]["identity"]
    resume = rl_compose.recovery_checkpoint(args[1], preset=preset)
    with pytest.raises(click.BadParameter, match="cannot be combined with --init"):
        rl.build_experiment(preset, table=SNOWBALL, resume=resume, init=preset.model)


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

    def collect(model):
        runs.append(model.step)
        return ()

    started = plan(table, preset.name, *start_args, then=collect)
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

    resume_args = ("--resume", f"{source.name}@{source.version}")
    planned = plan(table, preset.name, *resume_args, then=collect)
    assert planned.exit_code == 0, (planned.output, planned.exception)
    assert f"  model: {source_model['identity']}\n" in planned.output
    assert f"  starting_checkpoint: {source.name}/resume/step-4@{source.version}:" in planned.output
    _, resumed = runs
    resumed_config = materialized_config(resumed, str(tmp_path))
    inputs = yaml.safe_load(resumed_config.launch_config_yaml)["inputs"]
    assert inputs["model"] == source_model
    assert inputs["starting_checkpoint"]["uri"] == str(checkpoint)
    if not explicit_init:
        refused = plan(
            table,
            preset.name,
            *resume_args,
            "--set",
            "trainer.train_batch_size=2",
            "--set",
            "trainer.policy_mini_batch_size=2",
            "--set",
            "generator.n_samples_per_prompt=8",
        )
        assert refused.exit_code == 4 and "lcm_dp_size=16" in refused.output, (refused.output, refused.exception)
        return
    assert all(dep.name != model_name for dep in resumed.lower().deps)
    monkeypatch.setattr(rl, "run_skyrl", lambda _config, **_kwargs: None)
    resumed.run(resumed_config)
    inputs = read_run_inputs(str(Path(resumed_config.output.resolved_config_uri).parent))
    assert inputs.options is not None
    assert (inputs.options.checkpoints, inputs.options.checkpoint_every, inputs.options.export) == (
        ("latest", None, "latest") if table.model == "mochi" else ("best", 10, "best")
    )
    assert {key: getattr(inputs.resume.model, key) for key in tokenizer} == tokenizer
    assert (inputs.resume.artifact_name, inputs.resume.artifact_version, inputs.resume.step) == (
        source.name,
        source.version,
        4,
    )


@pytest.mark.parametrize("resume", (False, True), ids=("fresh", "resume"))
def test_mochi_plan_refuses_an_incomplete_default_export(
    selection, stored_checkpoint, resume, mochi_with_incomplete_default, tmp_path, monkeypatch
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    table = mochi_with_incomplete_default
    preset = table.by_name["train-long"]
    model = preset.model
    args = ()
    if resume:
        (stored_checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
        args += ("--resume", "checkpoints/previous@2026.11.19.1", "--checkpoints", "latest")
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
    sft = MIXTURE_262K_32NODE if model == "snowball" else request.getfixturevalue("mochi_sft_preset")
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
        initializer = sft.model()
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
        assert document["run"]["sft_preset"] == sft.name
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
        submitted.append((run.name, argv))
        return JobName.from_wire("/fixture/sft-then-rl")

    monkeypatch.setattr(rl, "submit_or_run", submit)
    for _ in range(2):
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        assert exit_.value.code == 0
    assert submitted == [(submitted[0][0], argv)] * 2


@pytest.mark.parametrize(
    ("name", "extra", "error"),
    [
        ("train-long", ("--init", "model@1"), "--sft requires a fresh candidate"),
        ("train-long", ("--resume", "run@1"), "--sft requires a fresh candidate"),
        ("learning-medium", (), "uses default-model controls"),
        ("speed-short", (), "does not accept --sft"),
    ],
)
def test_training_graph_rejects_incompatible_sft_options(name, extra, error):
    result = plan(
        SNOWBALL, name, "--sft", "unknown", *extra, sft_presets=lambda: pytest.fail("loaded refused SFT presets")
    )
    assert result.exit_code == 4 and error in result.output, (result.output, result.exception)


@pytest.mark.parametrize(
    ("table", "name", "allowed"),
    [("snowball", "train-long", True), ("snowball", "learning-medium", False), ("mochi", "train-medium", True)],
    indirect=["table"],
    ids=("exporting", "gated", "implicit-terminal-export"),
)
def test_runner_composes_after_an_export(selection, tmp_path, table, name, allowed, monkeypatch):
    if table is MOCHI:
        monkeypatch.setattr(rl, "check_cache", lambda _path: True)

    def read_policy(policy_output):
        return Artifact(path=policy_output)

    def evaluate(model):
        if table is MOCHI:
            document = yaml.safe_load(materialized_config(model.step, str(tmp_path)).launch_config_yaml)
            assert document["run"]["export_hf"]
            trainer = document["skyrl"]["trainer"]
            assert trainer["checkpoint_retention"] == "latest"
        return (apply("runs/post-rl-evaluation", read_policy, version="2026.11.19", policy_output=model.step),)

    result = CliRunner().invoke(
        rl.command(table, argv=(), then=evaluate), ["--version", "2026.11.19.1", "--preset", name]
    )
    assert result.exit_code == (0 if allowed else 4), (result.output, result.exception)
    expected = "runs/post-rl-evaluation" if allowed else "requires one ungated run with a model export"
    assert expected in result.output


def test_mochi_agentic_plan_collects_attempts_with_exports_disabled(selection, monkeypatch):
    monkeypatch.setattr(rl, "check_cache", lambda _path: True)
    result = plan(MOCHI, "train-medium-agentic")
    assert result.exit_code == 0, (result.output, result.exception)
    assert "rl-data/task-health/attempts/" in result.output
    status = yaml.safe_load(result.output.split("\n\nStepSpec", 1)[0])
    assert status["run_options"]["checkpoints"] == status["run_options"]["export"] == "none"
