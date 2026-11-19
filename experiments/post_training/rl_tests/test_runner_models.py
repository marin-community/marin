# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from marin.execution.lazy import apply, materialized_config
from marin.rl.skyrl import IrisSkyRLExecution
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import set_path

from experiments.post_training import rl, rl_gate, rl_recipe
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import BatchMix
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_SFT


def test_model_presets_compose_with_their_role_plans(selection, tmp_path):
    named = [(MOCHI, name) for name in MOCHI.by_name]
    named.extend((SNOWBALL, name) for name in ("train-medium-agentic", "train-long-agentic"))
    paths = []
    for table, name in named:
        preset = table.by_name[name]
        tasks = selection if preset.batch_mix else None
        spec = rl_recipe.run_spec(preset, name=f"test-{table.model}-{name}", tasks=tasks, model_name=table.model)
        step = rl_recipe.run_step(preset, spec)
        path = tmp_path / f"{table.model}-{name}.yaml"
        path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
        paths.append(str(path))
    documents = dict(zip(((table.model, name) for table, name in named), rl_gate.compose_configs(paths), strict=True))
    expected_gpus = {
        ("mochi", "train-medium"): 60,
        ("mochi", "train-long"): 60,
        ("mochi", "train-long-3rack"): 192,
        ("mochi", "train-long-agentic"): 60,
        ("mochi", "train-long-agentic-3rack"): 192,
        ("snowball", "train-medium-agentic"): 88,
        ("snowball", "train-long-agentic"): 256,
    }
    for identity, config in documents.items():
        allocation = config["iris"]["allocation"]
        assert allocation["num_nodes"] * allocation["gpus_per_node"] == expected_gpus[identity]
        trainer = config["skyrl"]["trainer"]
        if identity[0] == "mochi":
            assert trainer["policy"]["optimizer_config"]["optimizer"] == "MuonH"
            assert trainer["epochs"] == trainer["max_steps"]
            assert config["inputs"]["model"]["uri"].endswith("/step-400407/2af3f828aa-cpu-a1")
            assert config["skyrl"]["generator"]["sampling_params"]["stop_token_ids"] == [128001, 128009]
            assert config["skyrl"]["generator"]["engine_init_kwargs"]["compilation_config"]["cudagraph_mode"] == (
                "PIECEWISE"
            )
        if "agentic" in identity[1]:
            half = trainer["train_batch_size"] // 2
            assert trainer["rollout_buffer"]["groups_per_family"] == {"agentic": half, "nonagentic": half}
            normalized = rl_gate.comparison_config(config)
            for family in ("agentic", "nonagentic"):
                assert normalized[f"skyrl.trainer.rollout_buffer.groups_per_family.{family}"] == half

    medium = documents["mochi", "train-medium"]
    assert not medium["run"]["export_hf"]
    assert [callback["type"] for callback in medium["skyrl"]["trainer"]["callbacks"]] == ["checkpoint"]

    docs = (rl_recipe.REPO_ROOT / "docs/dev-guide/skyrl.md").read_text()
    fragment = docs.split("## Mochi inference on GB200", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
    served = rl_gate._config_values(documents["mochi", "train-long-3rack"]["skyrl"])
    for path, value in rl_gate._config_values(yaml.safe_load(fragment)).items():
        assert served[path] == value, path


def test_recipe_derives_agent_limits_and_preserves_authored_concurrency(selection):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    spec = rl_recipe.run_spec(
        preset,
        name="test-agent-settings",
        tasks=selection,
        settings=("context_budget.max_new_tokens_per_turn=8192", "trainer.train_batch_size=128"),
    )
    config = yaml.safe_load(spec.config_yaml)
    assert config["trainer"]["rollout_buffer"]["max_in_flight"] == 32
    assert config["trainer"]["rollout_buffer"]["groups_per_family"] == {"agentic": 64, "nonagentic": 64}
    model_info = config["terminal_bench"]["installed_agent"]["agent"]["kwargs"]["model_info"]
    assert model_info == {"max_input_tokens": 102_400 - 8192, "max_output_tokens": 8192}
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
    ("key", "value", "error"),
    [
        ("environment.task_sessions.daytona.ttl_minutes", 60, "ttl_minutes"),
        ("trainer.algorithm.group_admission.stall_timeout", 9000, "stall_timeout"),
        ("terminal_bench.installed_agent.agent.kwargs.unused", True, "Unknown Pi recipe kwargs: unused"),
    ],
)
def test_preset_refuses_inconsistent_agent_recipe(key, value, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    patch = {}
    set_path(patch, key, value)
    config = preset.recipe.config.merge(RecipePatch.from_document(patch))
    with pytest.raises(ValueError, match=error):
        replace(preset, recipe=replace(preset.recipe, config=config))


def test_checkpoint_settings_are_owned_by_the_runner(selection):
    preset = SNOWBALL.by_name["train-long"]
    with pytest.raises(click.BadParameter, match="derived by the runner"):
        rl_recipe.run_spec(preset, name="test-checkpoints", settings=("trainer.hf_export_checkpoint=best",))
    patch = RecipePatch.from_document({"trainer": {"ckpt_interval": 100}})
    config = preset.recipe.config.merge(patch)
    with pytest.raises(ValueError, match=r"derived by the runner: trainer\.ckpt_interval"):
        rl_recipe.run_spec(replace(preset, recipe=replace(preset.recipe, config=config)), name="test-checkpoints")


@pytest.mark.parametrize(
    ("shares", "error"),
    [
        ({"agentic": Fraction(1, 4), "nonagentic": Fraction(1, 4)}, "sum to 1"),
        ({"agentic": Fraction(1, 3), "nonagentic": Fraction(2, 3)}, "whole number of groups"),
        ({"agentic": Fraction(1, 2), "math": Fraction(1, 2)}, "training selection has"),
    ],
)
def test_runner_refuses_incompatible_batch_mix(selection, shares, error):
    preset = SNOWBALL.by_name["train-medium-agentic"]
    with pytest.raises((ValueError, click.BadParameter), match=error):
        preset = replace(preset, batch_mix=BatchMix(shares))
        rl_recipe.run_spec(preset, name="test-groups", tasks=selection)


def test_resume_pins_the_native_checkpoint(selection, tmp_path):
    preset = SNOWBALL.by_name["train-long"]
    root = tmp_path / "checkpoints" / "previous" / "2026.11.19.1"
    checkpoint = root / "scratch" / "global_step_4"
    checkpoint.mkdir(parents=True)
    (root / RESOLVED_LAUNCH_FILENAME).write_text(
        yaml.safe_dump({"config": {"artifacts": {"checkpoint_root": str(checkpoint.parent)}}})
    )
    (checkpoint.parent / "latest_ckpt_global_step.txt").write_text("4")
    with pytest.raises(click.BadParameter, match="temporary checkpoints usually expire after 14 days"):
        rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer checkpoint")
    resume = rl_recipe.published_checkpoint("checkpoints/previous@2026.11.19.1", preset=preset)
    resumed = rl.build_run(preset, table=SNOWBALL, resume=resume)
    plain = rl.build_run(preset, table=SNOWBALL)
    assert resumed.runs[0].name != plain.runs[0].name
    config = materialized_config(resumed.runs[0], str(tmp_path)).launch_config_yaml
    source = yaml.safe_load(config)["inputs"]["starting_checkpoint"]
    assert source["uri"] == str(checkpoint)
    assert "step-4@2026.11.19.1" in source["identity"]
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


def test_runner_composes_after_an_export_and_refuses_other_runs(selection):
    def read_policy(policy_output):
        return Artifact(path=policy_output)

    def evaluate(model):
        return (apply("runs/post-rl-evaluation", read_policy, version="2026.11.19", policy_output=model.step),)

    command = rl.command(SNOWBALL, argv=(), then=evaluate)
    args = ["--version", "2026.11.19.1", "--preset"]
    result = CliRunner().invoke(command, [*args, "train-long"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "runs/post-rl-evaluation" in result.output
    for extra in (("learning-medium",), ("train-long", "--replicates", "2")):
        result = CliRunner().invoke(command, [*args, *extra])
        assert result.exit_code == 4
        assert "requires one ungated run with a model export" in result.output
    no_export = CliRunner().invoke(rl.command(MOCHI, argv=(), then=evaluate), [*args, "train-medium"])
    assert no_export.exit_code == 4
    assert "requires one ungated run with a model export" in no_export.output
