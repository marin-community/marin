# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import click
import pytest
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe


def _render(step, path, *, runtime=False):
    context = (
        StepContext.for_run(str(path.with_suffix("")), str(path.parent), runtime_args=step.runtime_args, deps=step.deps)
        if runtime
        else StepContext.for_fingerprint(step.runtime_args, step.deps)
    )
    config = step.build_config(context)
    path.write_text(config.launch_config_yaml)


pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.timeout(180)  # The real bootstrap installs its independent, locked CPU environment.
def test_presets_compose_and_preserve_measurement_contracts(selection, tmp_path, monkeypatch, native_json):
    measured = presets.PRESETS["speed-medium"]
    with pytest.raises(ValueError, match="measured window"):
        snowball.build_run(replace(measured, measured=(measured.updates + 1,)))
    paths = []
    for name, preset in presets.PRESETS.items():
        if name == "speed-short":
            monkeypatch.delenv("WANDB_ENTITY", raising=False)
        else:
            monkeypatch.setenv("WANDB_ENTITY", "test-team")
        run = snowball.build_run(
            preset,
            tasks=selection if preset.tasks_input == "required" else None,
            control=name == "speed-long",
        )
        for index, step in enumerate(run.runs):
            path = tmp_path / f"{name}-{index}.yaml"
            _render(step, path, runtime=True)
            paths.append(str(path))
    off_boundary = snowball.build_run(presets.PRESETS["train-long"], settings=("trainer.max_steps=2",)).runs[0]
    path = tmp_path / "off-boundary.yaml"
    _render(off_boundary, path, runtime=True)
    paths.append(str(path))
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    revisioned = snowball.build_run(replace(presets.PRESETS["speed-short"], revision="2026.11.12.2"), control=True).runs[
        0
    ]
    path = tmp_path / "revisioned-speed.yaml"
    _render(revisioned, path, runtime=True)
    paths.append(str(path))
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    changed = snowball.build_run(
        presets.PRESETS["learning-medium"],
        settings=(
            'trainer={"policy_mini_batch_size": 64, "train_batch_size": 64, '
            '"bf16": false, "algorithm": {"batch_invariant": true}}',
            "generator.model_dtype=float16",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
            "trainer.policy.optimizer_config.lr=2.0e-6",
            "generator.engine_init_kwargs.probe={}",
        ),
    ).runs[0]
    path = tmp_path / "changed-batch.yaml"
    _render(changed, path, runtime=True)
    paths.append(str(path))
    learning = presets.PRESETS["learning-medium"]
    gate = snowball.snowball_rl_gate
    for current, arguments in (
        (replace(learning, revision="2026.11.12.2"), {}),
        (replace(presets.PRESETS["data-medium"], revision="2026.11.12.2"), {"tasks": selection}),
    ):
        with pytest.raises(gate.InvalidComparison, match="no control pin") as error:
            snowball.build_run(current, **arguments)
        assert "--preset learning-medium --version 2026.11.12.2 --control" in str(error.value)
    task_set = tmp_path / presets.DEFAULT_EVAL_LITE_NAME / "2026.11.12.2"
    task_set.mkdir(parents=True)
    (task_set / "summary.json").write_bytes(
        (tmp_path / presets.DEFAULT_EVAL_LITE_NAME / presets.DEFAULT_EVAL_LITE_VERSION / "summary.json").read_bytes()
    )
    promoted = replace(
        learning,
        revision="2026.11.12.2",
        tasks="fixture-selection@2026.11.10",
        eval_lite="eval-lite@2026.11.12.2",
        model=replace(
            learning.model,
            step=ArtifactStep.adopt(
                "tests/promoted-model",
                "2026.11.12.2",
                "s3://fixture-model/v2",
                kind=LevanterCheckpoint,
            ),
            tokenizer_uri="fixture-tokenizer",
            tokenizer_revision="v2",
        ),
        recipe=replace(
            learning.recipe,
            config=learning.recipe.config.with_settings(
                (
                    "trainer.train_batch_size=64",
                    "trainer.policy_mini_batch_size=64",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                    'trainer.policy.optimizer_config={"lr": 2e-06}',
                )
            ),
        ),
    )
    shared = replace(
        promoted,
        focus=presets.Focus.DATA,
        tasks_input="required",
        accepts_settings=False,
        accepts_init=False,
        accepts_control=False,
        shared_control=promoted.name,
        data_settings=frozenset(),
    )
    monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, promoted.name: promoted, shared.name: shared})
    monkeypatch.setitem(
        gate.CONTROLS,
        promoted.name,
        tuple(replace(pin, revision=promoted.revision) for pin in gate.CONTROLS[promoted.name]),
    )
    summary_path = Path(selection.path()) / presets.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text())
    summary["recipe"]["exclusions"] = [
        {"uri": str(task_set / presets.FULL_VALIDATION_FILENAME), "version": task_set.name}
    ]
    summary_path.write_text(json.dumps(summary))
    for current, arguments in ((promoted, {"control": True}), (shared, {"tasks": selection})):
        step = snowball.build_run(current, **arguments).runs[0]
        path = tmp_path / f"promoted-{current.name}.yaml"
        _render(step, path, runtime=True)
        paths.append(str(path))
    checkpoint_root = tmp_path / "custom-checkpoint"
    export = checkpoint_root / "hf"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "tokenizer_config.json").write_text("{}")
    checkpoint = ArtifactStep.adopt(
        "tests/custom-checkpoint", "2026.11.10", str(checkpoint_root), kind=LevanterCheckpoint
    )
    step = snowball.build_run(promoted, init=checkpoint).runs[0]
    path = tmp_path / "custom-checkpoint.yaml"
    _render(step, path, runtime=True)
    paths.append(str(path))
    rows = native_json(
        """
import asyncio, json, sys
from types import SimpleNamespace
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.evaluation_profiles import evaluation_profile_config, evaluation_profiles

class InferenceService:
    model_name = "fixture"
    tokenizer = SimpleNamespace(
        eos_token_id=99,
        decode=lambda *args, **kwargs: "answer",
        encode=lambda *args, **kwargs: [1, 2],
    )

    def __init__(self, overrides, validate):
        self.overrides, self.validate = overrides, validate

    async def tokenize(self, request):
        return {"tokens": [1, 2]}

    async def chat_completion(self, request):
        original = request["json"]
        self.body = dict(original)
        apply_openai_sampling(self.body, self.overrides, self.validate)
        return {"choices": [{
            "message": {"role": "assistant", "content": "answer"},
            "token_ids": [3], "finish_reason": "stop",
            "logprobs": {"content": [{"logprob": -0.1}]} if self.body.get("logprobs") else None,
        }]}

async def native_request(parameters, overrides, validate):
    service = InferenceService(overrides, validate)
    output = await DirectModelClient(service).generate({
        "prompts": [[{"role": "user", "content": "question"}]],
        "chat_completion_params": [{}], "sampling_params": parameters,
    })
    assert output["response_ids"] == [[3]]
    if parameters.get("logprobs") is not None:
        assert output["response_logprobs"] == [[-0.1]]
    return {key: service.body.get(key) for key in (
        "temperature", "max_completion_tokens", "ignore_eos", "seed", "logprobs",
    )}

configs = [load_launch_config(path) for path in sys.argv[1:]]
def quality_request(config):
    if not config.inputs.get("evaluation_data"):
        return None
    profile = evaluation_profiles(config.skyrl)["eval_lite"]
    merged = evaluation_profile_config(config.skyrl, profile, "eval_lite")
    return asyncio.run(native_request(
        get_vllm_sampling_params(merged.generator.eval_sampling_params),
        OmegaConf.to_container(config.skyrl.generator.sampling_params, resolve=True), True))

print(json.dumps([{
    "path": path, "name": config.run.id, "entrypoint": config.runtime.entrypoint,
    "nodes": config.iris.allocation.num_nodes,
    "gpus": config.iris.allocation.num_nodes * config.iris.allocation.gpus_per_node,
    "eval_interval": config.skyrl.trainer.eval_interval,
    "validation": bool(config.inputs.validation_data),
    "admission": config.skyrl.trajectory_runner.max_concurrent_tasks,
    "batch": config.skyrl.trainer.train_batch_size, "samples": config.skyrl.generator.n_samples_per_prompt,
    "quality_wire": quality_request(config),
    "at_start": [entry.get("eval_before_train", False) for entry in config.skyrl.trainer.callbacks
                 if entry.type == "evaluation"],
    "overlong_filtering": config.skyrl.generator.apply_overlong_filtering,
    "training": config.runtime.training_type is not None,
    "mode": [config.runtime.training_type, config.skyrl.trainer.rollout_buffer.max_staleness_steps,
             config.skyrl.trainer.algorithm.off_policy_correction, config.skyrl.trainer.rollout_buffer.batch_policy],
    "callbacks": [entry.type for entry in config.skyrl.trainer.callbacks],
    "steps": config.skyrl.trainer.max_steps,
    "team": config.iris.wandb_entity,
    "export_hf": config.run.export_hf,
    "micro": config.skyrl.trainer.micro_train_batch_size_per_gpu,
    "forward_micro": config.skyrl.trainer.micro_forward_batch_size_per_gpu,
    "numerics": [config.skyrl.trainer.bf16, config.skyrl.generator.model_dtype,
                 config.skyrl.trainer.algorithm.batch_invariant],
    "learning_rate": config.skyrl.trainer.policy.optimizer_config.lr,
    "weight_decay": config.skyrl.trainer.policy.optimizer_config.weight_decay,
    "adam_betas": list(config.skyrl.trainer.policy.optimizer_config.adam_betas),
    "model_uri": config.inputs.model.uri,
    "tokenizer": [config.inputs.model.tokenizer_uri, config.inputs.model.tokenizer_revision],
    "engine_kwargs": OmegaConf.to_container(config.skyrl.generator.engine_init_kwargs),
    "validation_identity": config.inputs.validation_data[0].identity if config.inputs.validation_data else None,
    "train_uri": config.inputs.train_data[0].uri if config.inputs.train_data else None,
    "eval_uri": config.inputs.validation_data[0].uri if config.inputs.validation_data else None,
    "profile_inputs": (OmegaConf.to_container(config.inputs.evaluation_data)
                       if config.inputs.get("evaluation_data") else {}),
    "terminal_eval": [entry.eval_on_train_end for entry in config.skyrl.trainer.callbacks if entry.type == "evaluation"]
        if config.runtime.training_type is not None else [],
    "wire": asyncio.run(native_request(
        get_vllm_sampling_params(config.skyrl.generator.eval_sampling_params),
        OmegaConf.to_container(config.skyrl.generator.sampling_params, resolve=True),
        True,
    )) if config.inputs.validation_data else None,
} for path, config in zip(sys.argv[1:], configs, strict=True)]))
""",
        paths,
    )
    assert len(rows) == len(paths)
    greedy_wire = {
        "temperature": 0.0,
        "max_completion_tokens": presets.EVAL_REPLY_TOKENS,
        "ignore_eos": False,
        "seed": None,
        "logprobs": None,
    }
    for row in rows:
        assert row["gpus"] <= 40, row
        assert row["team"] == ("marin-community" if "speed-short-" in row["name"] else "test-team")
        name = row["name"]
        if "learning-medium-" in name or "data-medium-" in name:
            assert row["at_start"] == [True]
        if "speed-short-" in name or "data-short-" in name:
            assert row["entrypoint"] == "skyrl_train.entrypoints.main_generate", row
            assert row["nodes"] == 1, row
            assert row["eval_interval"] > 0 and row["validation"], row
            expected = 2048 if "speed-short-" in name else 1024
            assert row["admission"] == expected, row
        if "speed-short-" in name:
            assert row["quality_wire"] == greedy_wire
        if "train-long-" in name:
            assert row["terminal_eval"] == [row["steps"] % 10 != 0], row
        if "learning-medium-" in name or "data-medium-" in name:
            assert row["export_hf"]
            assert "hf_model_save" in row["callbacks"]
        if "speed-long-" in name:
            assert not row["validation"] and not row["export_hf"]
        assert row["mode"] == (
            ["async", 2, "tis", "full_batch"] if row["training"] else [None, 0, "none", "full_batch"]
        ), row
        if row["training"]:
            assert not row["overlong_filtering"], row
        if row["wire"] is not None:
            if row["training"]:
                assert row["wire"] == greedy_wire
            else:
                assert row["wire"]["temperature"] == 1.0
                assert row["wire"]["max_completion_tokens"] == (4096 if "speed-short-" in name else 8192)
                assert row["wire"]["ignore_eos"] == ("speed-short-" in name)
        for source in row["profile_inputs"].get("eval_lite", []):
            assert (
                source["identity"].partition(":")[0]
                == f"{presets.SELECTION_PREFIX}/{presets.PRESETS['speed-short'].eval_lite}"
            )
            assert source["relative_path"] == "full_validation.parquet"
        if row["training"] and row["validation"]:
            assert "evaluation" in row["callbacks"], row
            if "best_checkpoint" in row["callbacks"]:
                assert row["callbacks"].index("evaluation") < row["callbacks"].index("best_checkpoint"), row
    for row in rows[-4:]:
        assert (row["weight_decay"], row["adam_betas"]) == (0.0, [0.9, 0.999])
        assert (row["batch"], row["samples"], row["micro"], row["forward_micro"], row["learning_rate"]) == (
            64,
            8,
            2,
            2,
            2e-6,
        )
    assert rows[-4]["engine_kwargs"]["probe"] == {}
    assert rows[-4]["numerics"] == [False, "float16", True]
    assert all(row["numerics"] == [True, "bfloat16", False] for row in rows if row is not rows[-4])
    sampled = next(row for row in rows if row["path"] == str(tmp_path / "revisioned-speed.yaml"))
    assert sampled["validation_identity"].partition(":")[0].endswith("@2026.11.12.2")
    assert rows[0]["name"].endswith("2026.11.13.1")
    for row in rows[-3:-1]:
        assert row["name"].endswith("2026.11.12.2")
        assert row["model_uri"] == "s3://fixture-model/v2"
        assert row["tokenizer"] == ["fixture-tokenizer", "v2"]
        assert row["train_uri"].endswith("fixture-selection/2026.11.10")
        assert row["eval_uri"].endswith("eval-lite/2026.11.12.2")
    assert rows[-1]["model_uri"] == str(export)
    assert rows[-1]["tokenizer"] == ["fixture-tokenizer", "v2"]
    summary = Path(presets.selection_artifact(presets.PRESETS["learning-medium"].tasks).path()) / "summary.json"
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(presets.PRESETS["learning-medium"])


@pytest.mark.timeout(180)
def test_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch, native_json):
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
    }
    expected = {name: versions.get(presets.PRESETS[name].revision) for name, versions in revisions.items()}
    paths = []
    for name in expected:
        preset = presets.PRESETS[name]
        spec = recipe.run_spec(preset, name="fixture-default-" + name)
        step = recipe.run_step(preset, spec)
        path = tmp_path / f"{name}.yaml"
        context = StepContext.for_run(
            "/snowball-pin-output/" + step.name,
            "/snowball-pin-prefix",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        path.write_text(step.build_config(context).launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    actual = {
        name: fingerprint_hash(json.dumps(gate.comparison_config(config), sort_keys=True))
        for name, config in zip(expected, configs, strict=True)
    }
    print(json.dumps(actual, sort_keys=True, indent=2))
    assert actual == expected, "Fingerprint envelope changed: bump the changed preset revision and recollect controls."


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = presets.PRESETS["learning-medium"]
    data = presets.PRESETS["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_run(
            replace(learning, accepts_settings=False), settings=("trainer.policy.optimizer_config.lr=2e-6",)
        )
    with pytest.raises(click.BadParameter, match="owned"):
        snowball.build_run(
            replace(learning, owned_keys=learning.owned_keys | {"trainer.policy.optimizer_config.lr"}),
            settings=("trainer.policy.optimizer_config.lr=2e-6",),
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
    pin = snowball.snowball_rl_gate.control_pins(presets.PRESETS["learning-medium"])[0]
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
