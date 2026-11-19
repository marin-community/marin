# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
from dataclasses import replace
from fractions import Fraction

import click
import pytest
import yaml
from click.testing import CliRunner
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, apply, materialized_config
from marin.rl.skyrl import ArtifactHfModel, IrisSkyRLExecution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path
from skyrlrecipe.locality import RoleTopologyMode

from experiments.post_training import rl, rl_gate, rl_recipe
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.rl_tests.fixtures import SELECTION_IDENTITY
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_SFT


@pytest.mark.parametrize(
    ("table", "revisions"),
    [
        (
            MOCHI,
            {
                "train-medium": {"2026.11.19.1": "f5a97828"},
                "train-long": {"2026.11.19.1": "4ce8b138"},
                "train-long-3rack": {"2026.11.19.1": "349c287f"},
                "train-long-agentic": {"2026.11.19.1": "ffd07b8d"},
                "train-long-agentic-3rack": {"2026.11.19.1": "34ec44c4"},
            },
        ),
        (
            SNOWBALL,
            {
                "train-medium-agentic": {"2026.11.19.1": "355e9079"},
                "train-long-agentic": {"2026.11.19.1": "b0fba56c"},
            },
        ),
    ],
    ids=("mochi", "snowball-agentic"),
)
def test_new_presets_match_revisioned_native_configs(table, revisions, selection, tmp_path, monkeypatch):
    with monkeypatch.context() as pins:
        configs = assert_revisioned_native_configs(
            table, revisions, tmp_path=tmp_path, monkeypatch=pins, mixed_tasks=selection
        )
    documents = dict(zip(revisions, configs, strict=True))
    agentic = documents["train-long-agentic"]["skyrl"]
    assert agentic["trainer"]["hf_export_checkpoint"] == "best"
    if table is MOCHI:
        for preset in table.presets:
            spec = rl_recipe.run_spec(
                preset, name="layout-check", tasks=selection if preset.batch_mix else None, model_name=table.model
            )
            execution = next(
                value
                for value in rl_recipe.run_step(preset, spec).runtime_args.values()
                if isinstance(value, IrisSkyRLExecution)
            )
            assert execution.role_topology == RoleTopologyMode.NVLINK_DOMAIN
        docs = (rl_recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
        fragment = docs.split("## Mochi inference on GB200", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
        served = rl_gate._config_values(documents["train-long-3rack"]["skyrl"])
        for path, value in rl_gate._config_values(yaml.safe_load(fragment)).items():
            assert served[path] == value, path


def test_mixed_candidate_uses_the_selected_tasks_and_preserves_authored_concurrency(selection):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    settings = ("context_budget.max_new_tokens_per_turn=24576", "trainer.train_batch_size=128")
    inputs = rl_gate.gate_inputs(preset.name, settings, SELECTION_IDENTITY)
    config = rl_gate.validate_candidate(preset, inputs, table=SNOWBALL)["skyrl"]
    assert config["trainer"]["rollout_buffer"]["max_in_flight"] == 32
    assert config["trainer"]["rollout_buffer"]["groups_per_family"] == {"agentic": 64, "nonagentic": 64}
    spec = rl_recipe.run_spec(
        preset,
        name="test-agent-settings",
        tasks=selection,
        settings=settings,
    )
    execution = next(
        value
        for value in rl_recipe.run_step(preset, spec).runtime_args.values()
        if isinstance(value, IrisSkyRLExecution)
    )
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
        ("owner", ("trainer.hf_export_checkpoint", "best"), "derived by the runner"),
        ("owner", ("trainer.rollout_buffer.groups_per_family", {"agentic": 32}), "derived by the runner"),
        ("mix", {"agentic": Fraction(1, 4), "nonagentic": Fraction(1, 4)}, "sum to 1"),
        ("mix", {"agentic": Fraction(1, 3), "nonagentic": Fraction(2, 3)}, "whole number of groups"),
        ("mix", {"agentic": Fraction(1, 2), "math": Fraction(1, 2)}, "training selection has"),
        ("checkpoint", None, "best export requires checkpointing"),
        ("evaluation", 3, "fall on checkpoint steps"),
        ("prefix", None, "storage prefix"),
        ("gated", None, "already built"),
    ],
)
def test_runner_refuses_incompatible_launches(selection, tmp_path, monkeypatch, case, value, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    with pytest.raises((ValueError, click.BadParameter), match=error):
        if case in ("recipe", "owner"):
            key, setting = value
            patch = {}
            set_path(patch, key, setting)
            config = preset.recipe.config.merge(RecipePatch.from_document(patch))
            if case == "owner":
                with pytest.raises(click.BadParameter, match=error):
                    rl_recipe.run_spec(
                        preset, name="settings-test", tasks=selection, settings=(f"{key}={json.dumps(setting)}",)
                    )
            rl_recipe.run_spec(
                replace(preset, recipe=replace(preset.recipe, config=config)), name="recipe-test", tasks=selection
            )
        elif case == "mix":
            rl_recipe.run_spec(replace(preset, batch_mix=BatchMix(value)), name="mix-test", tasks=selection)
        elif case == "checkpoint":
            replace(preset, checkpoint_every=None)
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
            MOCHI,
            "train-medium",
            (
                "trainer.policy.megatron_config.tensor_model_parallel_size=2",
                "trainer.policy.megatron_config.context_parallel_size=2",
                "generator.weight_sync_transport=broadcast",
            ),
            "MuonH fused-matrix updates require tensor parallel size 1",
        ),
        (SNOWBALL, "train-medium-agentic", ("terminal_bench.installed_agent.agent.kwargs.unused=true",), "unused"),
    ],
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


def test_resume_pins_the_native_checkpoint(selection, tmp_path):
    preset = SNOWBALL.by_name["train-long"]
    root = tmp_path / "checkpoints" / "previous" / "2026.11.19.1"
    checkpoint = root / "scratch" / "global_step_4"
    checkpoint.mkdir(parents=True)
    (root / RESOLVED_LAUNCH_FILENAME).write_text(
        yaml.safe_dump({"config": {"artifacts": {"checkpoint_root": str(checkpoint.parent)}}})
    )
    (checkpoint.parent / "latest_ckpt_global_step.txt").write_text("4")
    with pytest.raises(click.BadParameter, match="source run's retention period"):
        rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    resume = rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    resumed = rl.build_run(preset, table=SNOWBALL, resume=resume)
    config = materialized_config(resumed.runs[0], str(tmp_path)).launch_config_yaml
    source = yaml.safe_load(config)["inputs"]["starting_checkpoint"]
    assert source["uri"] == str(checkpoint)
    assert "step-4@2026.11.19.1" in source["identity"]
    args = ("--preset", preset.name, "--version", preset.revision, "--resume", "checkpoints/previous@2026.11.19.1")
    planned = CliRunner().invoke(rl.command(SNOWBALL, argv=args), args)
    assert planned.exit_code == 0 and "-resume-step-4" in planned.output, (planned.output, planned.exception)
    with pytest.raises(click.BadParameter, match="cannot be combined with --init"):
        rl.build_run(preset, table=SNOWBALL, resume=resume, init=preset.model)


@pytest.mark.usefixtures("source_provenance")
def test_sft_composition_reenters_its_own_module(selection, monkeypatch):
    module = "experiments.post_training.snowball_sft_then_rl"
    argv = ("--preset", "train-long", "--version", "2026.11.19.1", "--no-issue", "--run")
    prefix = "s3://fixture-models/shared"
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setattr(rl_recipe, "artifact_prefix", lambda _cluster: prefix)
    monkeypatch.setattr(rl, "_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", [module, *argv])
    submitted = []

    def submit(entry_module, handles, _max_concurrent, _env, *, argv):
        assert entry_module == module
        (run,) = handles
        initializer = MIXTURE_SFT.model()
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
    [(SNOWBALL, "train-long", True), (SNOWBALL, "learning-medium", False), (MOCHI, "train-medium", False)],
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
