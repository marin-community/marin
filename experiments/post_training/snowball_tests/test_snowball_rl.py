# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import click
import pytest
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe
from experiments.post_training.skyrl_test_support import native_json

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = presets.PRESETS["learning-medium"]
    changed = snowball.build_run(
        learning,
        settings=(
            'trainer={"policy_mini_batch_size":64,"train_batch_size":64,"bf16":false,"algorithm":{"batch_invariant":true}}',
            "generator.model_dtype=float16",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
            "trainer.policy.optimizer_config.lr=2e-6",
            "generator.engine_init_kwargs.probe={}",
        ),
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
    checkpoint = tmp_path / "checkpoint"
    export = checkpoint / "hf"
    export.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (export / name).write_text("{}")
    initializer = ArtifactStep.adopt("tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint)
    revisioned = replace(presets.PRESETS["speed-short"], revision="2026.11.12.2")
    steps = (
        changed,
        snowball.build_run(promoted).runs[0],
        snowball.build_run(learning, init=initializer).runs[0],
        snowball.build_run(presets.PRESETS["train-long"], settings=("trainer.max_steps=2",)).runs[0],
        snowball.build_run(revisioned, control=True).runs[0],
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
        assert (
            trainer["policy"]["optimizer_config"]["weight_decay"],
            trainer["policy"]["optimizer_config"]["adam_betas"],
        ) == (0, [0.9, 0.999])
    trainer, generator = configs[0]["skyrl"]["trainer"], configs[0]["skyrl"]["generator"]
    assert (
        trainer["bf16"],
        generator["model_dtype"],
        trainer["algorithm"]["batch_invariant"],
        generator["engine_init_kwargs"]["probe"],
    ) == (False, "float16", True, {})
    assert configs[2]["inputs"]["model"]["uri"] == str(export)
    assert configs[2]["iris"]["wandb_entity"] == "test-team"
    callbacks = configs[3]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[3]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[4]["inputs"]["validation_data"][0]["identity"].partition(":")[0].endswith("@" + revisioned.revision)
    with pytest.raises(ValueError, match="require guards"):
        replace(learning, guards=())
    with pytest.raises(ValueError, match="evaluation"):
        replace(presets.PRESETS["train-long"], evaluations=())
    summary = Path(presets.selection_artifact(learning.tasks).path()) / presets.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(learning)


def test_harbor_coverage_controls_best_checkpoint(selection, tmp_path):
    step = snowball.build_run(presets.PRESETS["train-long-harbor"]).runs[0]
    path = tmp_path / "harbor.yaml"
    path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
    selected = native_json()(
        """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from cloud.iris.launch_config import load_launch_config
from skyrl_train.callbacks.base import CallbackHandler, TrainerState, TrainerControl
from skyrl_train.callbacks.builtin import create_callbacks_from_config
config = load_launch_config(sys.argv[1])
selected = []
for index, (native, cohorts) in enumerate(((99, (27, 27, 27)), (98, (30, 30, 30)), (100, (26, 30, 30)))):
    directory = Path(sys.argv[1]).parent / f"coverage-{index}"
    config.skyrl.trainer.ckpt_path = str(directory)
    metrics = {"eval/all/num_attempted": 100, "eval/all/num_scored": native, "eval/all/avg_verifier_score": 0.5}
    harbor = next(row for row in config.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
    for cohort, scored in zip(harbor.cohorts, cohorts, strict=True):
        prefix = f"eval/harbor/{cohort.name}/"
        metrics.update({prefix + "num_attempted": 30, prefix + "num_scored": scored, prefix + "reward_sum": scored / 2})
    handler = CallbackHandler(create_callbacks_from_config(config.skyrl))
    state = TrainerState(global_step=0, epoch=0, total_steps=100, num_steps_per_epoch=100)
    asyncio.run(handler.call_event_async("on_evaluate", state, TrainerControl(), metrics=metrics,
                                       trainer=SimpleNamespace(cfg=config.skyrl)))
    selected.append((directory / "best_checkpoint.json").exists())
print(json.dumps(selected))
""",
        [str(path)],
    )
    assert selected == [True, False, False]


def test_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    bindings = {binding for preset in presets.PRESETS.values() for binding in (preset.tasks, preset.eval_lite)}
    summaries = {
        binding: (Path(presets.selection_artifact(binding).path()) / presets.SUMMARY_FILENAME).read_text()
        for binding in bindings
    }
    monkeypatch.setattr(presets, "SELECTION_ROOT", "memory://snowball-calibration")
    for binding, summary in summaries.items():
        (StoragePath(presets.selection_artifact(binding).path()) / presets.SUMMARY_FILENAME).write_text(summary)
    revisions = {
        "speed-short": {"2026.11.13.1": "e492dd1e"},
        "data-short": {"2026.11.13.1": "f29bf661"},
        "speed-medium": {"2026.11.12.1": "1af00f32"},
        "learning-medium": {"2026.11.12.1": "5d01147c"},
        "data-medium": {"2026.11.12.1": "5d01147c"},
        "speed-long": {"2026.11.11.1": "ebab276e"},
        "train-long": {"2026.11.11.1": "05f0a068"},
        "train-long-harbor": {"2026.11.14.1": "2daffa92"},
    }
    paths = []
    for name in revisions:
        preset = presets.PRESETS[name]
        step = recipe.run_step(preset, recipe.run_spec(preset, name="fixture-default-" + name))
        path = tmp_path / f"{name}.yaml"
        path.write_text(materialized_config(step, "/snowball-pin-prefix").launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    actual = {
        name: fingerprint_hash(json.dumps(gate.comparison_config(config), sort_keys=True))
        for name, config in zip(revisions, configs, strict=True)
    }
    print(json.dumps(actual, sort_keys=True, indent=2))
    assert actual == {name: values.get(presets.PRESETS[name].revision) for name, values in revisions.items()}
    for name, config in zip(revisions, configs, strict=True):
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert config["iris"]["wandb_entity"] == "marin-community"
        assert config["iris"]["allocation"]["num_nodes"] * config["iris"]["allocation"]["gpus_per_node"] <= 40
        generating = name in ("speed-short", "data-short")
        assert (
            config["runtime"]["training_type"],
            trainer["rollout_buffer"]["max_staleness_steps"],
            trainer["algorithm"]["off_policy_correction"],
            trainer["rollout_buffer"]["batch_policy"],
        ) == ((None, 0, "none", "full_batch") if generating else ("async", 2, "tis", "full_batch"))
        if generating:
            assert config["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
            assert config["iris"]["allocation"]["num_nodes"] == 1
            assert config["skyrl"]["trajectory_runner"]["max_concurrent_tasks"] == (
                2048 if name == "speed-short" else 1024
            )
            assert trainer["eval_interval"] > 0 and config["inputs"]["validation_data"]
            sampling = generator["eval_sampling_params"]
            assert (sampling["temperature"], sampling["max_generate_length"], sampling["ignore_eos"]) == (
                1,
                4096 if name == "speed-short" else 8192,
                name == "speed-short",
            )
        else:
            assert generator["apply_overlong_filtering"] is False
        assert (trainer["bf16"], generator["model_dtype"], trainer["algorithm"]["batch_invariant"]) == (
            True,
            "bfloat16",
            False,
        )
        if name in ("speed-short", "learning-medium", "data-medium", "train-long"):
            profile = next(
                (
                    row["additional_evaluations"]["eval_lite"]
                    for row in trainer["callbacks"]
                    if "eval_lite" in row.get("additional_evaluations", {})
                ),
                None,
            )
            for source in config["inputs"].get("evaluation_data", {}).get("eval_lite", []):
                assert (
                    source["identity"].partition(":")[0]
                    == f"{presets.SELECTION_PREFIX}/{presets.PRESETS[name].eval_lite}"
                )
                assert source["relative_path"] == "full_validation.parquet"
            sampling = profile["sampling_params"] if profile else generator["eval_sampling_params"]
            assert (
                sampling["temperature"],
                sampling["ignore_eos"],
                sampling["max_generate_length"],
                profile["n_samples_per_prompt"] if profile else generator["eval_n_samples_per_prompt"],
            ) == (0, False, 8192, 1)
        if name in ("learning-medium", "data-medium"):
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_before_train"] is True
            assert config["run"]["export_hf"] and "hf_model_save" in [row["type"] for row in trainer["callbacks"]]
        if name == "speed-long":
            assert not config["inputs"]["validation_data"] and not config["run"]["export_hf"]
        if name == "train-long":
            kinds = [row["type"] for row in trainer["callbacks"]]
            assert kinds.index("evaluation") < kinds.index("best_checkpoint")
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_on_train_end"] is False


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = presets.PRESETS["learning-medium"]
    data = presets.PRESETS["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_run(
            replace(learning, accepts_settings=False), settings=("trainer.policy.optimizer_config.lr=2e-6",)
        )

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=settings, **kwargs).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    assert name(learning, "generator.engine_init_kwargs.probe={}") != name()
    repeated = snowball.build_run(learning, replicates=3)
    assert [step.name for step in snowball.build_run(learning, replicates=2).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    task_set_name = presets.DEFAULT_EVAL_LITE_NAME
    task_set_version = presets.DEFAULT_EVAL_LITE_VERSION
    new_task_set = tmp_path / task_set_name / "2026.11.05"
    new_task_set.mkdir(parents=True)
    (new_task_set / "summary.json").write_bytes(
        (tmp_path / task_set_name / task_set_version / "summary.json").read_bytes()
    )
    changed_task_set = replace(
        learning,
        eval_lite=f"{task_set_name}@2026.11.05",
    )
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=initializer)),
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
    pin = gate.control_pin(presets.PRESETS["learning-medium"])
    for index in range(pin.runs):
        recorded_runs("learning-medium", gate.control_name(pin.preset, pin.commit, index))
    first = snowball.build_run(learning, requirements=("eval_lite/mean>=0",))
    second = snowball.build_run(learning, requirements=("eval_lite/mean>=0.1",))
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
    with pytest.raises(click.BadParameter, match="different heldout"):
        snowball.build_run(data, tasks=selection)
    summary["recipe"]["heldout"] = heldout
    summary_path.write_text(json.dumps(summary))
    for current in (data, presets.PRESETS["data-medium"]):
        selected = replace(current, eval_lite=f"{task_set_name}@2026.11.05")
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(selected, tasks=selection)
        summary["recipe"]["exclusions"] = [
            {"uri": str(new_task_set / presets.FULL_VALIDATION_FILENAME), "version": "2026.11.05"}
        ]
        summary_path.write_text(json.dumps(summary))
        assert snowball.build_run(selected, tasks=selection).gate is not None
        summary["recipe"]["exclusions"] = original_exclusions
        summary_path.write_text(json.dumps(summary))
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (presets.selection_artifact(presets.PRESETS["learning-medium"].tasks), selection):
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(data, tasks=tasks)
