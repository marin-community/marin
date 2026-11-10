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
from experiments.post_training import snowball_rl_presets as presets


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
    paths = []
    for name, preset in presets.PRESETS.items():
        if name == "speed-short":
            monkeypatch.delenv("WANDB_ENTITY", raising=False)
        else:
            monkeypatch.setenv("WANDB_ENTITY", "test-team")
        run = snowball.build_run(
            preset,
            tasks=selection if preset.focus == presets.Focus.DATA else None,
            control=name == "speed-long",
        )
        for index, step in enumerate(run.runs):
            path = tmp_path / f"{name}-{index}.yaml"
            _render(step, path, runtime=True)
            paths.append(str(path))
    off_boundary = snowball.build_run(
        presets.PRESETS["train-long"], settings=(presets.parse_setting("trainer.max_steps=2"),)
    ).runs[0]
    path = tmp_path / "off-boundary.yaml"
    _render(off_boundary, path, runtime=True)
    paths.append(str(path))
    changed = snowball.build_run(
        presets.PRESETS["learning-medium"],
        settings=tuple(
            map(
                presets.parse_setting,
                (
                    "trainer={train_batch_size: 64, policy_mini_batch_size: 64}",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                ),
            )
        ),
    ).runs[0]
    path = tmp_path / "changed-batch.yaml"
    _render(changed, path, runtime=True)
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
from skyrl_train.evaluation_profiles import evaluation_profile_config, parse_evaluation_profiles

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
    profile = parse_evaluation_profiles(config.skyrl.trainer.callbacks[0].additional_evaluations)["test_panel"]
    merged = evaluation_profile_config(config.skyrl, profile, "test_panel")
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
    "callbacks": [entry.type for entry in config.skyrl.trainer.callbacks],
    "steps": config.skyrl.trainer.max_steps,
    "team": config.iris.wandb_entity,
    "export_hf": config.run.export_hf,
    "micro": config.skyrl.trainer.micro_train_batch_size_per_gpu,
    "forward_micro": config.skyrl.trainer.micro_forward_batch_size_per_gpu,
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
        if row["training"]:
            assert not row["overlong_filtering"], row
        if row["wire"] is not None:
            if row["training"]:
                assert row["wire"] == greedy_wire
            else:
                assert row["wire"]["temperature"] == 1.0
                assert row["wire"]["max_completion_tokens"] == (4096 if "speed-short-" in name else 8192)
                assert row["wire"]["ignore_eos"] == ("speed-short-" in name)
        for source in row["profile_inputs"].get("test_panel", []):
            assert (
                source["identity"].partition(":")[0]
                == f"{presets.SELECTION_PREFIX}/{presets.EVALUATIONS['test_panel'].source}"
            )
            assert source["relative_path"] == "full_validation.parquet"
        if row["training"] and row["validation"]:
            assert "evaluation" in row["callbacks"], row
            if "best_checkpoint" in row["callbacks"]:
                assert row["callbacks"].index("evaluation") < row["callbacks"].index("best_checkpoint"), row
    assert (rows[-1]["batch"], rows[-1]["samples"], rows[-1]["micro"], rows[-1]["forward_micro"]) == (64, 8, 2, 2)
    with pytest.raises(click.BadParameter, match="owned by"):
        snowball.build_run(
            presets.PRESETS["learning-medium"],
            settings=(presets.parse_setting("generator.sampling_params={ignore_eos: true}"),),
        )
    summary = Path(presets.default_selection().path()) / "summary.json"
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(presets.PRESETS["learning-medium"])


@pytest.mark.timeout(180)
def test_gated_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch, native_json):
    summary = (Path(presets.default_selection().path()) / "summary.json").read_text()
    panel_name = presets.DEFAULT_TEST_PANEL_NAME
    panel_version = presets.DEFAULT_TEST_PANEL_VERSION
    panel_summary = (tmp_path / panel_name / panel_version / "summary.json").read_text()
    monkeypatch.setattr(presets, "SELECTION_ROOT", "memory://snowball-calibration")
    (
        StoragePath(presets.SELECTION_ROOT) / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION / "summary.json"
    ).write_text(summary)
    (StoragePath(presets.SELECTION_ROOT) / panel_name / panel_version / "summary.json").write_text(panel_summary)
    expected = {
        "2026.11.06.1": {
            "speed-short": ("3c26e593", "4983e280"),
            "data-short": ("f47923c5", "b8748fb5"),
            "speed-medium": ("e268a2c8", "f3822837"),
            "learning-medium": ("934e55eb", "4bdf2d67"),
            "speed-long": ("5b99bc3f", "fa196cc1"),
        }
    }[presets.PRESET_REVISION]
    paths = []
    for name in expected:
        step = snowball.build_run(presets.PRESETS[name], control=True).runs[0]
        path = tmp_path / f"{name}.yaml"
        _render(step, path)
        paths.append(str(path))
    candidate = snowball.build_run(presets.PRESETS["learning-medium"]).runs[0]
    candidate_path = tmp_path / "learning-medium-candidate.yaml"
    _render(candidate, candidate_path)
    paths.append(str(candidate_path))
    configs = native_json(
        """
import json, os, sys
from pathlib import Path
from omegaconf import OmegaConf
from cloud.iris.launch_config import load_launch_config
os.environ["HOME"] = "/snowball-pin-home"
configs = [load_launch_config(Path(path)) for path in sys.argv[1:]]
print(json.dumps([
    OmegaConf.to_container(config, resolve=True) for config in configs
], sort_keys=True))
""",
        paths,
    )
    actual = {
        name: (
            fingerprint_hash(
                json.dumps({"skyrl": config["skyrl"], "allocation": config["iris"]["allocation"]}, sort_keys=True)
            ),
            fingerprint_hash(json.dumps(config, sort_keys=True)),
        )
        for name, config in zip(expected, configs[:-1], strict=True)
    }
    assert actual == expected, "Fingerprint envelope changed: bump PRESET_REVISION and recollect controls."
    learning_control = configs[list(expected).index("learning-medium")]
    assert configs[-1]["skyrl"] == learning_control["skyrl"]
    assert configs[-1]["iris"]["allocation"] == learning_control["iris"]["allocation"]
    assert configs[-1]["run"]["export_hf"] == learning_control["run"]["export_hf"]


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = presets.PRESETS["learning-medium"]
    data = presets.PRESETS["data-short"]
    long = presets.PRESETS["train-long"]

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=tuple(map(presets.parse_setting, settings)), **kwargs).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    repeated = snowball.build_run(learning, replicates=3)
    assert [step.name for step in snowball.build_run(learning, replicates=2).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    panel_name = presets.DEFAULT_TEST_PANEL_NAME
    panel_version = presets.DEFAULT_TEST_PANEL_VERSION
    new_panel = tmp_path / panel_name / "2026.11.05"
    new_panel.mkdir(parents=True)
    (new_panel / "summary.json").write_bytes((tmp_path / panel_name / panel_version / "summary.json").read_bytes())
    changed_panel = replace(
        learning,
        evaluations=tuple(replace(evaluation, source=f"{panel_name}@2026.11.05") for evaluation in learning.evaluations),
    )
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=initializer)),
        (name(), name(changed_panel)),
    ]
    assert all(left != right for left, right in pairs), pairs
    assert name(
        learning, "trainer.policy.optimizer_config={lr: 0.000002}", "trainer.policy.optimizer_config={weight_decay: 0.1}"
    ) == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002", "trainer.policy.optimizer_config.weight_decay=0.1"
    )
    original = run(long)
    run(long, "trainer.policy.optimizer_config.weight_decay=0.2")
    assert run(long).fingerprint() == original.fingerprint()
    pin = snowball.snowball_rl_gate.control_pins("learning-medium")[0]
    for index in range(pin.runs):
        recorded_runs("learning-medium", presets.control_name(pin.preset, pin.commit, index))
    first = snowball.build_run(learning, requirements=("test_panel/mean>=0",))
    second = snowball.build_run(learning, requirements=("test_panel/mean>=0.1",))
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
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (presets.default_selection(), selection):
        with pytest.raises(click.BadParameter, match="full test panel"):
            snowball.build_run(data, tasks=tasks)
