# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest
from finestore.rl.run_records import (
    read_attempt,
)
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from rigging.provenance import Provenance
from skyrlrecipe.documents import get_path, set_path

from experiments.post_training import rl as snowball
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_measurements import RUN_INPUTS_FILENAME, RunInputs, RunOptions, read_run_inputs
from experiments.post_training.rl_tests.fixtures import (
    SELECTION_IDENTITY,
    cache_native_compositions,
    compare,
    comparisons,
    controls,
    edit_metrics,
    edit_receipt,
    fake_s3,
    fake_s3_checkpoint,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import write_json as _json

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


@pytest.mark.parametrize(
    "case",
    [
        "initializer",
        "initializer-hf",
        "priority",
        "stale-default",
        "control-commit",
        "unbound-data",
        "allocation",
        "cached-default",
        "zero-spread",
        "missing-guard",
        "omitted-default",
        "omitted-current-nondefault",
        "omitted-schema-nondefault",
        "omitted-candidate-default",
        "schema-timeout",
        "protocol-temperature",
        "evaluation-cadence",
        "malformed-seed",
    ],
)
def test_gate_compares_recorded_configuration_and_defaults(preset_table, recorded_runs, tmp_path, monkeypatch, case):
    preset = snowball_model.SNOWBALL.by_name["speed-medium" if case == "missing-guard" else "learning-medium"]
    setting = "trainer.policy.optimizer_config.lr=2e-6"
    settings = (
        ()
        if case
        in (
            "stale-default",
            "cached-default",
            "zero-spread",
            "omitted-default",
            "omitted-current-nondefault",
            "omitted-schema-nondefault",
            "omitted-candidate-default",
            "schema-timeout",
        )
        else (setting,)
    )
    inputs = RunInputs(preset_name=preset.name, settings=settings)
    if case.startswith("initializer"):
        name, version = "tests/custom-init", "2026.11.07"
        root = tmp_path / name / version
        if case == "initializer-hf":
            initializer = ArtifactHfModel(
                ArtifactStep.adopt(name, version, "s3://fixture-initializer", kind=Artifact),
                preset.model.tokenizer_uri,
                preset.model.tokenizer_revision,
            )
            fake_s3(monkeypatch, "fixture-initializer", "fixture-initializer/policy/tokenizer_config.json")
        else:
            write_skyrl_run_record(
                root,
                name,
                version,
                fingerprint="initializer",
                hf_model_uri="s3://fixture-initializer/policy",
                global_step=25,
                tokenizer_uri=preset.model.tokenizer_uri,
                tokenizer_revision=preset.model.tokenizer_revision,
                iris_job_id="/test-user/initializer",
            )
            initializer = SkyRLPolicyModel(ArtifactStep.adopt(name, version, str(root), kind=SkyRLRun))
        inputs = inputs.model_copy(update={"init": recipe.stored_initializer(initializer, preset=preset)})
    control_roots = controls(recorded_runs, preset.name)
    if case == "zero-spread":
        pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
        control_roots = [
            recorded_runs(
                preset.name, gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL), score=0.5
            )
            for index in range(pin.replicates)
        ]
    if case in ("stale-default", "omitted-current-nondefault"):
        changed = setting if case == "stale-default" else 'generator.chat_template_kwargs={"enable_thinking":false}'
        preset = replace(preset, recipe=replace(preset.recipe, config=preset.recipe.config.with_settings((changed,))))
        preset_table(preset)
    name = "users/test/config-" + case
    if case == "initializer-hf":
        with monkeypatch.context() as planning:
            planning.setattr(
                skyrl_api, "discover_hf_checkpoints", lambda *_: pytest.fail("gate planning read the initializer export")
            )
            planned = gate.validate_candidate(preset, inputs, table=snowball_model.SNOWBALL)
            assert planned["inputs"]["model"]["uri"].endswith("/<terminal-hf-export>")
        return
    if case == "priority":
        override = replace(
            preset,
            recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, priority="production")),
        )
        inputs = inputs.model_copy(update={"options": RunOptions.from_preset(override)})
        gate.validate_candidate(override, inputs, table=snowball_model.SNOWBALL)
        table = snowball_model.SNOWBALL
        with monkeypatch.context() as altered:
            altered.setattr(snowball_model, "SNOWBALL", replace(table, presets=(override,)))
            candidate = recorded_runs(preset.name, name, inputs=inputs)
        assert json.loads((candidate / "terminal.json").read_text())["config"]["iris"]["priority"] == "production"
    else:
        candidate = recorded_runs(preset.name, name, inputs=inputs)
    omissions = {
        "omitted-default": {
            "config.skyrl.generator.chat_template_kwargs": {},
            "config.inputs.starting_checkpoint": None,
            "config.iris.role_topology": None,
        },
        "omitted-current-nondefault": {"config.skyrl.generator.chat_template_kwargs": {}},
        "omitted-schema-nondefault": {"config.skyrl.trainer.policy.optimizer_config.weight_decay": 0.0},
        "omitted-candidate-default": {"config.skyrl.generator.chat_template_kwargs": {}},
        "schema-timeout": {"config.skyrl.generator.chat_template_kwargs": {}},
    }.get(case, {})
    if omissions:
        omitted = [candidate] if case == "omitted-candidate-default" else control_roots
        for old in omitted:
            with edit_receipt(old) as receipt:
                for path, original in omissions.items():
                    parent, key = path.rsplit(".", 1)
                    assert get_path(receipt, parent).pop(key) == original
    if case == "schema-timeout":
        cache_native_compositions(monkeypatch)
        original_subprocess = subprocess.run
        for diagnostic in ("timeout", "empty", "whitespace", "oserror"):
            with monkeypatch.context() as unavailable:
                unavailable.setattr(
                    Provenance,
                    "capture",
                    lambda *_, tag=diagnostic: Provenance(tag, "commit-one", False, "branch", "test"),
                )

                def unavailable_defaults(command, *args, diagnostic=diagnostic, **kwargs):
                    if any("OmegaConf.structured(launch_config.SkyRLLaunchConfig)" in str(value) for value in command):
                        if diagnostic == "timeout":
                            raise subprocess.TimeoutExpired(command, 120)
                        if diagnostic == "oserror":
                            raise FileNotFoundError("native composer executable missing")
                        return subprocess.CompletedProcess(command, 1 if diagnostic == "whitespace" else 0, "", "\n")
                    return original_subprocess(command, *args, **kwargs)

                unavailable.setattr(subprocess, "run", unavailable_defaults)
                report = compare(preset.name, "--candidate", name)
                assert (
                    report["verdict"] == "invalid" and "native configuration cannot be composed" in report["reason"]
                ), report
        return
    reasons = {
        "stale-default": ("control config differs",),
        "zero-spread": ("variation is zero",),
        "missing-guard": ("missing or non-finite guard metric",),
        "omitted-current-nondefault": ("control config differs",),
        "omitted-schema-nondefault": ("control config differs",),
        "omitted-candidate-default": ("skyrl.generator.chat_template_kwargs",),
    }.get(case, ())
    mutation = {
        "control-commit": (control_roots[0], "result.provenance.base_commit", "wrong-commit", "does not match its pin"),
        "unbound-data": (candidate, "config.skyrl.data.train_data", ["/unbound/train.parquet"], "skyrl.data.train_data"),
        "allocation": (
            candidate,
            "config.iris.allocation.num_nodes",
            preset.recipe.layout.workload.plan.num_nodes + 1,
            "iris.allocation.num_nodes",
        ),
        "protocol-temperature": (
            candidate,
            "config.skyrl.generator.eval_sampling_params.temperature",
            0.5,
            "evaluation protocol",
        ),
        "malformed-seed": (candidate, "config.skyrl.trainer.seed", None, "malformed recorded configuration", "seed"),
    }.get(case)
    if mutation is not None:
        root, path, value, *reasons = mutation
        with edit_receipt(root) as receipt:
            set_path(receipt, path, value)
    if case == "missing-guard":
        edit_metrics(candidate, lambda row: row.update({"policy/mismatch/staleness0/selected_tokens": 0}))
    if case == "evaluation-cadence":
        with edit_receipt(candidate) as receipt:
            evaluation = next(
                callback
                for callback in receipt["config"]["skyrl"]["trainer"]["callbacks"]
                if callback["type"] == "evaluation"
            )
            evaluation["eval_steps"] = 17
        reasons = ("eval_steps",)
    report = compare(preset.name, "--candidate", name)
    if reasons:
        assert report["verdict"] == "invalid" and all(reason in report["reason"] for reason in reasons), report
    else:
        assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
        if case.startswith("initializer"):
            assert "inputs.model.uri" in report["run"]["config_changes"]
    if case == "cached-default":
        promoted = replace(
            preset,
            recipe=replace(
                preset.recipe,
                config=preset.recipe.config.with_settings(("trainer.policy.optimizer_config.weight_decay=0.1",)),
            ),
        )
        preset_table(promoted)
        updated = compare(preset.name, "--candidate", name)
        assert updated["verdict"] == "invalid" and "weight_decay" in updated["reason"]
        assert updated["report"] != report["report"]


@pytest.mark.parametrize("preset_name", ["learning-medium", "data-medium"])
def test_python_run_saves_typed_inputs_for_comparison(
    preset_table, recorded_runs, selection, tmp_path, monkeypatch, preset_name
):
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    gain_key = "eval_lite/family_mean/gain_per_gpu_hour"
    model = tmp_path / "model"
    model.mkdir()
    model_uri = f"s3://marin-test/models/{tmp_path.name}"

    fake_s3_checkpoint(monkeypatch, model_uri, model)
    local = replace(
        preset.model,
        step=ArtifactStep.adopt(
            preset.model.step.name,
            preset.model.step.version,
            model_uri,
            kind=preset.model.step.artifact_type,
        ),
    )
    preset = replace(preset, model=local)
    preset_table(preset)
    controls(recorded_runs, preset_name)
    tasks = selection if preset_name == "data-medium" else None
    settings = (
        (
            "trainer.policy.optimizer_config.lr=2.0e-6",
            "trainer.train_batch_size=64",
            "trainer.policy_mini_batch_size=64",
            "trainer.micro_train_batch_size_per_gpu=2",
            "generator.n_samples_per_prompt=8",
            'extra_env.VLLM_PROBE="1"',
            "generator.max_num_seqs=1024",
        )
        if tasks is None
        else ()
    )
    export = model / "selected-hf"
    export.mkdir()
    for filename in ("config.json", "tokenizer_config.json"):
        (export / filename).write_text("{}")
    initializer = (
        ArtifactHfModel(
            ArtifactStep.adopt(
                "tests/noncanonical-initializer",
                "2026.11.17",
                model_uri,
                kind=local.step.artifact_type,
                config={"tokenizer_uri": "tests/own-tokenizer"},
            ),
            "tests/own-tokenizer",
            "own-tokenizer-revision",
        )
        if preset.accepts_init
        else None
    )
    built = snowball.build_experiment(
        preset,
        settings=settings,
        tasks=tasks,
        init=initializer,
        requirements=("policy/raw_grad_norm<=10",),
        table=snowball_model.SNOWBALL,
    )
    task_name = SELECTION_IDENTITY if tasks else None
    source = recorded_runs(
        preset.name,
        built.runs[0].name,
        fingerprint=built.runs[0].fingerprint(),
        inputs=built.inputs,
    )
    staged = source.with_name("native-staging")
    source.rename(staged)
    (staged / RUN_INPUTS_FILENAME).unlink()

    def native(config, *, forward_env=()):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(snowball, "run_skyrl", native)
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preset", "train-long", "--require", "unrelated"])
    run(*built.runs)
    recorded = read_run_inputs(str(source))
    assert recorded.preset_name == preset_name and recorded.requirements == ("policy/raw_grad_norm<=10",)
    decoded = dict(item.split("=", 1) for item in recorded.settings)
    assert {key: json.loads(value) for key, value in decoded.items()} == (
        {}
        if tasks
        else {
            "trainer": {
                "policy": {"optimizer_config": {"lr": 2e-6}},
                "train_batch_size": 64,
                "policy_mini_batch_size": 64,
                "micro_train_batch_size_per_gpu": 2,
            },
            "extra_env": {"VLLM_PROBE": "1"},
            "generator": {"max_num_seqs": 1024, "n_samples_per_prompt": 8},
        }
    )
    assert recorded.tasks == task_name
    if initializer is not None:
        assert (
            recorded.init.artifact_root,
            recorded.init.relative_path,
            recorded.init.tokenizer_uri,
            recorded.init.tokenizer_revision,
        ) == (model_uri, None, "tests/own-tokenizer", "own-tokenizer-revision")
        assert json.loads(recorded.init.fingerprint_payload)["config"]["tokenizer_uri"] == "tests/own-tokenizer"
        newer = model / "newer-hf"
        newer.mkdir()
        for filename in ("config.json", "tokenizer_config.json"):
            (newer / filename).write_text("{}")
    report = compare(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["requirements"] == list(recorded.requirements)
    assert report["run"]["settings"] == list(recorded.settings)
    assert report["run"]["tasks"] == recorded.tasks
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert comparisons(report)[gain_key]["run"] == pytest.approx(0.005)
    _json(source / "exports/best_checkpoint.json", {"step": preset.updates})
    [status] = snowball._status(built, preset, stored_only=True)["runs"]
    assert status["best_checkpoint_step"] == preset.updates
    if tasks is None:
        assert {
            "inputs.model.uri",
            "skyrl.trainer.policy.optimizer_config.lr",
            "skyrl.generator.max_num_seqs",
            "skyrl.trainer.train_batch_size",
            "skyrl.trainer.micro_forward_batch_size_per_gpu",
            "skyrl.trainer.rollout_buffer.max_in_flight",
        } <= set(report["run"]["config_changes"])
        trainer = read_attempt(str(source / "exports"), "recorded-attempt").native.config["skyrl"]["trainer"]
        assert (trainer["micro_forward_batch_size_per_gpu"], trainer["rollout_buffer"]["max_in_flight"]) == (2, 64)
