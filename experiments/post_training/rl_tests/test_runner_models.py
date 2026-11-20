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
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity, materialized_config
from marin.rl.skyrl import ArtifactHfModel, IrisSkyRLExecution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path

from experiments.post_training import rl, rl_gate, rl_recipe
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_scoring import read_gate_inputs
from experiments.post_training.rl_tests.fixtures import SELECTION_IDENTITY, write_published_artifact
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE


@pytest.fixture
def table(request):
    return {"snowball": SNOWBALL, "mochi": MOCHI}[request.param]


@pytest.fixture
def mochi_sft_preset(monkeypatch):
    from experiments.grug.moe_hero_ep import launch_hero_sft, sft_recipe  # noqa: PLC0415
    from experiments.post_training.mochi_sft import MIXTURE_262K_176NODE  # noqa: PLC0415

    monkeypatch.setattr(sft_recipe, "checkpoint_step", lambda _path: 393491)
    monkeypatch.setattr(launch_hero_sft, "special_token_init", lambda _step: None)
    return MIXTURE_262K_176NODE


def test_mochi_default_matches_production_sft_export(mochi_sft_preset):
    produced = mochi_sft_preset.model()
    for preset in MOCHI.presets:
        model = preset.model
        assert isinstance(model, ArtifactHfModel)
        prefix = rl_recipe.artifact_prefix(preset.recipe.layout.cluster)
        assert (model.step.name, model.step.version, model.step.path(prefix)) == (
            produced.step.name,
            produced.step.version,
            produced.step.path(prefix),
        )
        assert model.relative_path == produced.relative_path and model.tokenizer_uri is None


def test_checkpoint_retention_composes_with_the_save_and_evaluation_callbacks(selection, tmp_path):
    template = MOCHI.by_name["train-long"]
    cases = (
        ("none", None, "none", ["evaluation"]),
        ("latest", 4, "none", ["checkpoint", "evaluation"]),
        ("best", None, "best", ["checkpoint", "evaluation", "best_checkpoint", "hf_model_save"]),
    )
    paths = []
    for mode, keep_every, export, _ in cases:
        preset = replace(
            template,
            checkpoints=mode,
            checkpoint_every=None if mode == "none" else 2,
            keep_every=keep_every,
            export=export,
        )
        spec = rl_recipe.run_spec(preset, name=f"retention-{mode}", model_name=MOCHI.model)
        launch = materialized_config(rl_recipe.run_step(preset, spec), str(tmp_path)).launch_config_yaml
        path = tmp_path / f"{mode}.yaml"
        path.write_text(launch)
        paths.append(str(path))

    for (mode, keep_every, export, callbacks), config in zip(cases, rl_gate.compose_configs(paths), strict=True):
        trainer = config["skyrl"]["trainer"]
        assert trainer.get("checkpoint_retention", "none") == mode
        assert trainer.get("checkpoint_keep_every") == keep_every
        assert trainer["ckpt_interval"] == 0
        assert trainer["max_ckpts_to_keep"] == 2
        assert [callback["type"] for callback in trainer["callbacks"]] == callbacks
        assert config["run"]["export_hf"] == (export == "best")
        assert trainer["hf_export_checkpoint"] == ("best" if export == "best" else "latest")
        if mode != "none":
            assert trainer["callbacks"][0]["save_steps"] == 2


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"checkpoints": "latest"}, "export must be none or match checkpoints"),
        ({"export": "none", "evaluations": ()}, "requires an evaluation"),
        ({"checkpoints": "latest", "export": "none", "checkpoint_every": None}, 'checkpoints="latest" requires'),
        ({"checkpoint_every": None}, 'checkpoints="best" requires checkpoint_every'),
        ({"checkpoints": "none", "export": "none"}, 'checkpoints="none" requires'),
        ({"keep_every": 0}, "keep_every must be positive"),
        ({"keep_every": -1}, "keep_every must be positive"),
    ],
    ids=[
        "export-mismatch",
        "best-without-evaluation",
        "latest-without-cadence",
        "best-without-cadence",
        "none-with-cadence",
        "zero-keep-every",
        "negative-keep-every",
    ],
)
def test_checkpoint_retention_refuses_incompatible_presets(changes, error):
    with pytest.raises(ValueError, match=error):
        replace(MOCHI.by_name["train-long"], **changes)


@pytest.mark.parametrize(
    ("table", "revisions"),
    [
        (
            "mochi",
            {
                "train-medium": {"2026.11.19.1": "8b5a970e", "2026.11.20.1": "674dadce"},
                "train-long": {"2026.11.19.1": "db854d78", "2026.11.20.1": "b6144722"},
                "train-long-3rack": {"2026.11.19.1": "a0e17c94", "2026.11.20.1": "bc381480"},
                "train-medium-agentic": {"2026.11.19.1": "3f203da5", "2026.11.20.1": "5f0a0fa1"},
                "train-long-agentic": {"2026.11.19.1": "bef4849f", "2026.11.20.1": "9c4ea508"},
                "train-long-agentic-3rack": {"2026.11.19.1": "87a6ba8f", "2026.11.20.1": "15da2779"},
            },
        ),
        (
            "snowball",
            {
                "train-medium-agentic": {"2026.11.19.1": "756f58ef", "2026.11.20.1": "b557a4f0"},
                "train-long-agentic": {"2026.11.19.1": "72ec851f", "2026.11.20.1": "6335c530"},
            },
        ),
    ],
    indirect=["table"],
    ids=("mochi", "snowball-agentic"),
)
def test_new_presets_match_revisioned_native_configs(table, revisions, selection, tmp_path, monkeypatch):
    with monkeypatch.context() as pins:
        configs = assert_revisioned_native_configs(
            table, revisions, tmp_path=tmp_path, monkeypatch=pins, mixed_tasks=selection
        )
    if table.model == "mochi":
        documents = dict(zip(revisions, configs, strict=True))
        docs = (rl_recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
        fragment = docs.split("## Mochi inference on GB200", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
        served = rl_gate._config_values(documents["train-long-3rack"]["skyrl"])
        for path, value in rl_gate._config_values(yaml.safe_load(fragment)).items():
            assert served[path] == value, path


def test_mixed_candidate_uses_the_selected_tasks_and_preserves_authored_concurrency(selection, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    preset = SNOWBALL.by_name["train-medium-agentic"]
    settings = ("context_budget.max_new_tokens_per_turn=24576", "trainer.train_batch_size=128")
    run = rl.build_run(preset, table=SNOWBALL, tasks=selection, settings=settings)
    assert run.inputs is not None
    config = rl_gate.validate_candidate(preset, run.inputs, table=SNOWBALL)["skyrl"]
    # The Snowball campaign recipe sets a fixed cap of 32 generating groups.
    assert config["trainer"]["rollout_buffer"]["max_in_flight"] == 32
    assert config["trainer"]["rollout_buffer"]["groups_per_family"] == {"agentic": 64, "nonagentic": 64}
    execution = next(value for value in run.runs[0].runtime_args.values() if isinstance(value, IrisSkyRLExecution))
    assert set(execution.required_env_vars) == {
        "DAYTONA_RL_API_KEY",
        "DAYTONA_EVAL_API_KEY",
        "DAYTONA_TARGET",
        "WANDB_API_KEY",
    }


def test_snowball_plan_refuses_explicitly_disabled_route_capture(selection):
    args = ("--preset", "train-medium-agentic", "--version", "dev", "--plan", "--tasks", SELECTION_IDENTITY)
    args += ("--set", "generator.engine_init_kwargs.enable_return_routed_experts=false")
    invalid = CliRunner().invoke(rl.command(SNOWBALL, argv=args), args)
    assert invalid.exit_code == 4, (invalid.output, invalid.exception)
    assert "moe_router_replay requires generator.engine_init_kwargs.enable_return_routed_experts=true" in invalid.output


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
            rl_recipe.run_spec(preset, name="settings-test", tasks=selection, settings=(f"{key}={json.dumps(setting)}",))
        error = "recipe settings are derived by the runner"
    with pytest.raises((ValueError, click.BadParameter), match=error):
        if case in ("recipe", "owner"):
            key, setting = value
            patch = {}
            set_path(patch, key, setting)
            config = preset.recipe.config.merge(RecipePatch.from_document(patch))
            rl_recipe.run_spec(
                replace(preset, recipe=replace(preset.recipe, config=config)), name="recipe-test", tasks=selection
            )
        elif case == "mix":
            rl_recipe.run_spec(replace(preset, batch_mix=BatchMix(value)), name="mix-test", tasks=selection)
        elif case == "evaluation":
            replace(preset, evaluations=(replace(preset.evaluations[0], every=value),))
        else:

            def export_artifact(path):
                return Artifact(path=path)

            monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "other") if case == "prefix" else str(tmp_path))
            init = ArtifactHfModel(apply("models/not-built", export_artifact, version="2026.11.19", path=OUT))
            preset = SNOWBALL.by_name["learning-medium" if case == "gated" else "train-long"]
            rl.build_run(preset, table=SNOWBALL, init=init)


@pytest.mark.parametrize(
    ("table", "name", "settings", "error"),
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
            "MuonH fused-matrix updates require tensor parallel size 1",
        ),
        ("snowball", "train-medium-agentic", ("terminal_bench.installed_agent.agent.kwargs.unused=true",), "unused"),
    ],
    indirect=["table"],
    ids=("muonh", "installed-agent-kwargs"),
)
def test_plan_print_rejects_invalid_native_config(table, name, settings, error, selection):
    preset = table.by_name[name]
    args = ["--preset", name, "--version", preset.revision, "--plan"]
    for setting in settings:
        args += ["--set", setting]
    if preset.tasks_input == "required":
        args += ["--tasks", SELECTION_IDENTITY]
    result = CliRunner().invoke(rl.command(table, argv=tuple(args)), args)
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
    summary_path = Path(selection.path()) / rl_recipe.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["harness_graded_datasets"] = [dataset]
    summary_path.write_text(json.dumps(summary))
    args = ["--preset", "train-medium-agentic", "--version", "dev", "--plan", "--tasks", SELECTION_IDENTITY]
    if harness_override:
        args += ["--set", f"environment.task_sessions.task_spec.harness_by_dataset.{dataset}={harness_override}"]
    result = CliRunner().invoke(rl.command(SNOWBALL, argv=tuple(args)), args)
    assert result.exit_code == exit_code, (result.output, result.exception)
    if exit_code:
        assert "harbor_trial in harness_by_dataset" in result.output and dataset in result.output


@pytest.fixture
def stored_checkpoint(tmp_path):
    root = tmp_path / "checkpoints" / "previous" / "2026.11.19.1"
    checkpoint = root / "scratch" / "global_step_4"
    checkpoint.mkdir(parents=True)
    (root / RESOLVED_LAUNCH_FILENAME).write_text(
        yaml.safe_dump({"config": {"artifacts": {"checkpoint_root": str(checkpoint.parent)}}})
    )
    (checkpoint.parent / "latest_ckpt_global_step.txt").write_text("4")
    return checkpoint


@pytest.mark.parametrize("recorded_model", (False, True), ids=("legacy", "changed-default"))
def test_resume_pins_the_native_checkpoint(selection, tmp_path, stored_checkpoint, recorded_model):
    table = SNOWBALL
    preset = SNOWBALL.by_name["train-long"]
    original_model = preset.model.step
    checkpoint = stored_checkpoint
    with pytest.raises(click.BadParameter, match="source run's retention period"):
        rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    if recorded_model:
        (original,) = rl.build_run(preset, table=table).runs
        source_config = materialized_config(original, str(tmp_path))
        source_model = yaml.safe_load(source_config.launch_config_yaml)["inputs"]["model"]
        launch = checkpoint.parent.parent / RESOLVED_LAUNCH_FILENAME
        document = yaml.safe_load(launch.read_text())
        document["config"]["inputs"] = {"model": source_model}
        launch.write_text(yaml.safe_dump(document))
        changed = ArtifactStep.adopt("models/changed-default", preset.revision, "gs://fixture-models/changed-default")
        preset = replace(preset, model=replace(preset.model, step=changed))
        table = replace(table, presets=(preset,))
    resume = rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    resumed = rl.build_run(preset, table=table, resume=resume)
    config = materialized_config(resumed.runs[0], str(tmp_path)).launch_config_yaml
    native_inputs = yaml.safe_load(config)["inputs"]
    source = native_inputs["starting_checkpoint"]
    assert source["uri"] == str(checkpoint)
    assert "step-4@2026.11.19.1" in source["identity"]
    assert native_inputs["model"]["uri"] == original_model.path(str(tmp_path))
    args = ("--preset", preset.name, "--version", preset.revision, "--resume", "checkpoints/previous@2026.11.19.1")
    planned = CliRunner().invoke(rl.command(table, argv=args), args)
    assert planned.exit_code == 0 and "-resume-step-4" in planned.output, (planned.output, planned.exception)
    assert f"  model: {artifact_identity(original_model)}\n" in planned.output
    if recorded_model:
        invalid_batch = (
            *args,
            "--set",
            "trainer.train_batch_size=2",
            "--set",
            "trainer.policy_mini_batch_size=2",
            "--set",
            "generator.n_samples_per_prompt=8",
        )
        refused = CliRunner().invoke(rl.command(table, argv=invalid_batch), invalid_batch)
        assert refused.exit_code == 4 and "lcm_dp_size=16" in refused.output, (refused.output, refused.exception)
    with pytest.raises(click.BadParameter, match="cannot be combined with --init"):
        rl.build_run(preset, table=table, resume=resume, init=preset.model)


@pytest.fixture
def mochi_with_incomplete_default(tmp_path):
    preset = MOCHI.by_name["train-long"]
    model = preset.model
    root = tmp_path / "model"
    (root / "hf").mkdir(parents=True)
    (root / "hf" / "config.json").write_text("{}")
    model = replace(model, step=ArtifactStep.adopt(model.step.name, model.step.version, str(root)))
    return replace(MOCHI, presets=(replace(preset, model=model),))


@pytest.mark.parametrize("table", ("snowball", "mochi"), indirect=True)
def test_resume_inherits_the_source_runs_initializer(
    selection, table, tmp_path, monkeypatch, mochi_with_incomplete_default
):
    if table is MOCHI:
        table = mochi_with_incomplete_default
    preset = table.by_name["train-long"]
    model_name = "models/alternative"
    model_uri = "gs://fixture-models/alternative"
    tokenizer = {"tokenizer_uri": "gs://fixture-models/tokenizer", "tokenizer_revision": "tokenizer-pin"}
    write_published_artifact(tmp_path, model_name, preset.revision, Artifact, source=model_uri, config=tokenizer)
    monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/hf"])
    monkeypatch.setattr(rl, "run_skyrl", lambda _config, **_kwargs: None)
    runner = CliRunner()
    runs = []

    def collect(model):
        runs.append(model.step)
        return ()

    args = ("--preset", preset.name, "--version", preset.revision, "--plan")
    start_args = (*args, "--init", f"{model_name}@{preset.revision}")
    started = runner.invoke(rl.command(table, argv=start_args, then=collect), start_args)
    assert started.exit_code == 0, (started.output, started.exception)
    (source,) = runs
    source_config = materialized_config(source, str(tmp_path))
    source.run(source_config)
    launch_file = Path(source_config.output.resolved_config_uri)
    source_launch = yaml.safe_load(source_config.launch_config_yaml)
    source_model = source_launch["inputs"]["model"]
    assert source_model["uri"] == f"{model_uri}/hf"
    checkpoint_root = tmp_path / "checkpoints-from-source"
    checkpoint = checkpoint_root / "global_step_4"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text("4")
    source_launch["artifacts"]["checkpoint_root"] = str(checkpoint_root)
    launch_file.write_text(yaml.safe_dump({"config": source_launch}))
    monkeypatch.setattr("marin.rl.skyrl.discover_hf_checkpoints", lambda path: [f"{path}/newer-hf"])

    resume_args = (*args, "--resume", f"{source.name}@{source.version}")
    planned = runner.invoke(rl.command(table, argv=resume_args, then=collect), resume_args)
    assert planned.exit_code == 0, (planned.output, planned.exception)
    assert f"  model: {source_model['identity']}\n" in planned.output
    assert f"  starting_checkpoint: {source.name}/resume/step-4@{source.version}:" in planned.output
    _, resumed = runs
    resumed_config = materialized_config(resumed, str(tmp_path))
    assert yaml.safe_load(resumed_config.launch_config_yaml)["inputs"]["model"] == source_model
    assert all(dep.name != model_name for dep in resumed.lower().deps)
    resumed.run(resumed_config)
    inputs = read_gate_inputs(str(Path(resumed_config.output.resolved_config_uri).parent))
    assert {key: getattr(inputs.resume.model, key) for key in tokenizer} == tokenizer
    assert (inputs.resume.artifact_name, inputs.resume.artifact_version, inputs.resume.step) == (
        source.name,
        source.version,
        4,
    )


@pytest.mark.parametrize("resume", (False, True), ids=("fresh", "resume"))
def test_mochi_plan_refuses_an_incomplete_default_export(
    selection, stored_checkpoint, resume, mochi_with_incomplete_default
):
    table = mochi_with_incomplete_default
    preset = table.by_name["train-long"]
    model = preset.model
    table = replace(table, presets=(replace(preset, export="none"),))
    args = ("--preset", preset.name, "--version", preset.revision, "--plan")
    if resume:
        (stored_checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
        args += ("--resume", "checkpoints/previous@2026.11.19.1")
    result = CliRunner().invoke(rl.command(table, argv=args), args)
    assert result.exit_code == 4, (result.output, result.exception)
    assert f"{model.step.name}@{model.step.version}" in result.output
    assert "mochi_sft_then_rl.py" in result.output and "--init NAME@VERSION" in result.output


@pytest.mark.usefixtures("source_provenance")
@pytest.mark.parametrize("model", ("snowball", "mochi"))
def test_sft_composition_reenters_its_own_module(selection, monkeypatch, model, request):
    module = f"experiments.post_training.{model}_sft_then_rl"
    sft = MIXTURE_262K_32NODE if model == "snowball" else request.getfixturevalue("mochi_sft_preset")
    argv = ("--preset", "train-long", "--version", "2026.11.19.1", "--run")
    prefix = "s3://fixture-models/shared"
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setattr(rl_recipe, "artifact_prefix", lambda _cluster: prefix)
    monkeypatch.setattr(rl, "_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", [module, *argv])
    submitted = []

    def submit(entry_module, handles, _max_concurrent, _env, *, argv):
        assert entry_module == module
        (run,) = handles
        initializer = sft.model()
        producer = next(step for step in run.deps if step.name == initializer.step.name)
        assert producer.fingerprint() == initializer.step.fingerprint()
        document = yaml.safe_load(materialized_config(run, prefix).launch_config_yaml)
        assert document["inputs"]["model"]["uri"] == f"{producer.path(prefix)}/{initializer.relative_path}"
        assert document["inputs"]["model"]["tokenizer_uri"] is None
        submitted.append((run.name, argv))
        return JobName.from_wire("/fixture/sft-then-rl")

    monkeypatch.setattr(rl, "submit_or_run", submit)
    for _ in range(2):
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        assert exit_.value.code == 0
    assert submitted == [(submitted[0][0], argv)] * 2


@pytest.mark.parametrize(
    ("table", "name", "allowed"),
    [("snowball", "train-long", True), ("snowball", "learning-medium", False), ("mochi", "train-medium", False)],
    indirect=["table"],
    ids=("exporting", "gated", "no-export"),
)
def test_runner_composes_after_an_export(selection, table, name, allowed):
    def read_policy(policy_output):
        return Artifact(path=policy_output)

    def evaluate(model):
        return (apply("runs/post-rl-evaluation", read_policy, version="2026.11.19", policy_output=model.step),)

    result = CliRunner().invoke(
        rl.command(table, argv=(), then=evaluate), ["--version", "2026.11.19.1", "--preset", name]
    )
    assert result.exit_code == (0 if allowed else 4), (result.output, result.exception)
    expected = "runs/post-rl-evaluation" if allowed else "requires one ungated run with a model export"
    assert expected in result.output
