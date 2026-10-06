# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from marin.external_dependencies import MARIN_SKYRL

from experiments.post_training.russell_rsi.dose_qualification import QUALIFIED_RUNTIME_COMMIT, qualified_dose_source

QUALIFIED_RUNTIME = replace(MARIN_SKYRL, commit=QUALIFIED_RUNTIME_COMMIT)


@pytest.fixture
def qualification(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.post_training.russell_rsi.dose_qualification.MARIN_SKYRL", QUALIFIED_RUNTIME)
    files = {}
    roots = {role: tmp_path / role for role in ("rl", "optimizer", "reload")}
    for root in roots.values():
        root.mkdir()
    source_data = tmp_path / "data"
    source_data.mkdir()
    export = tmp_path / "exports" / "global_step_4" / "policy"
    export.mkdir(parents=True)
    model = {"identity": "parent-pin", "tokenizer_uri": "tokenizer", "tokenizer_revision": "pin"}
    requested = {
        "run": {"id": "source", "attempt_id": "attempt"},
        "runtime": {"launcher_commit": QUALIFIED_RUNTIME.commit, "profile": "megatron"},
        "artifacts": {
            "terminal_manifest_uri": str(roots["rl"] / "terminal.json"),
            "resolved_config_uri": str(roots["rl"] / "resolved.yaml"),
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "export_root": str(tmp_path / "exports"),
        },
        "inputs": {"model": model, "train_data": [{"uri": str(source_data)}]},
        "skyrl": {
            "trainer": {"policy": {"optimizer_config": {"lr": 5e-7}}},
            "generator": {"sampling_params": {"temperature": 1}},
            "context_budget": {"max_prompt_tokens": 229376},
        },
    }
    compiled = {"terminal": deepcopy(requested), "resolved": {"config": deepcopy(requested)}}

    def cpu_compiler(argv, *, text):
        assert yaml.safe_load(Path(argv[-1]).read_text()) == requested
        return json.dumps(compiled)

    monkeypatch.setattr("experiments.post_training.russell_rsi.dose_qualification.subprocess.check_output", cpu_compiler)
    rl_dependency = "source-rl@2026.10.05"
    optimizer_dependency = "source-optimizer@2026.10.05"
    result = {
        "hf_model_uri": str(export),
        "global_step": 4,
        "iris_job_id": "job",
        "checkpoint_root": requested["artifacts"]["checkpoint_root"],
        "terminal_manifest_uri": requested["artifacts"]["terminal_manifest_uri"],
        "tokenizer_uri": "tokenizer",
        "tokenizer_revision": "pin",
    }
    values = {
        "rl": {
            "name": "source-rl",
            "version": "2026.10.05",
            "fingerprint": "rlpin",
            "output_path": str(roots["rl"]),
            "config": {"launch_config_yaml": yaml.safe_dump(requested)},
            "result": result,
        },
        "optimizer": {
            "name": "source-optimizer",
            "version": "2026.10.05",
            "fingerprint": "optpin",
            "output_path": str(roots["optimizer"]),
            "deps": [rl_dependency],
            "dep_paths": [],
            "config": {"expected_updates": 4, "actual_updates": 4, "export_uri": str(export)},
        },
        "reload": {
            "name": "source-reload",
            "version": "2026.10.05",
            "fingerprint": "reloadpin",
            "output_path": str(roots["reload"]),
            "deps": [rl_dependency, optimizer_dependency],
            "dep_paths": [],
            "config": {
                "model": {"location": str(export), "identity": rl_dependency + ":rlpin"},
                "evals": "mmlu-smoke",
                "limit": 1,
            },
            "result": {"results_paths": ["graded-reload"]},
        },
        "terminal": {
            "config": deepcopy(compiled["terminal"]),
            "result": {
                "state": "succeeded",
                "iris_job_state": "succeeded",
                "failure": None,
                "launcher_commit": QUALIFIED_RUNTIME.commit,
                "runtime_profile": "megatron",
                "run_id": "source",
                "attempt_id": "attempt",
                "iris_job_id": "job",
                "model": {
                    "global_step": 4,
                    "policy_export_uri": str(export),
                    "checkpoint_root": result["checkpoint_root"],
                    "terminal_manifest_uri": result["terminal_manifest_uri"],
                    "tokenizer_uri": "tokenizer",
                    "tokenizer_revision": "pin",
                },
            },
        },
        "resolved": deepcopy(compiled["resolved"]),
        "source_replay": {"schedule_sha256": "replay-pin"},
        "export_index": {"weight_map": {"tensor": "weights.safetensors"}},
    }
    paths = {role: roots[role] / ".artifact.json" for role in roots}
    paths.update(
        terminal=roots["rl"] / "terminal.json",
        resolved=roots["rl"] / "resolved.yaml",
        source_replay=source_data / "replay-plan.json",
        export_index=export / "model.safetensors.index.json",
        export_manifest=export / ".marinskyrl-model-manifest.json",
    )

    def write(role):
        data = json.dumps(values[role]).encode()
        paths[role].write_bytes(data)
        files[role] = {"uri": str(paths[role]), "sha256": hashlib.sha256(data).hexdigest()}

    for role in values:
        write(role)
    values["export_manifest"] = {
        "files": [
            {"path": "model.safetensors.index.json", "sha256": files["export_index"]["sha256"]},
            {"path": "weights.safetensors", "sha256": "a" * 64},
        ]
    }
    write("export_manifest")
    for role, root in roots.items():
        path = root / ".executor_status"
        path.write_bytes(b"SUCCESS")
        files[f"{role}_status"] = {"uri": str(path), "sha256": hashlib.sha256(b"SUCCESS").hexdigest()}
    return {"files": files}, values, paths, write


def test_completed_four_update_source_qualifies_without_submission(qualification):
    evidence, _, _, _ = qualification
    result = qualified_dose_source(evidence)
    assert result.rl_identity == "source-rl@2026.10.05:rlpin"
    assert result.reload_identity == "source-reload@2026.10.05:reloadpin"
    assert result.export_uri.endswith("global_step_4/policy")


@pytest.mark.parametrize("setting", ["optimizer", "sampling", "context"])
def test_agreeing_terminal_and_resolved_reject_effective_recipe_changes(qualification, setting):
    evidence, values, _, write = qualification
    for role in ("terminal", "resolved"):
        recipe = values[role]["config"]["skyrl"]
        if setting == "optimizer":
            recipe["trainer"]["policy"]["optimizer_config"]["lr"] = 1e-6
        elif setting == "sampling":
            recipe["generator"]["sampling_params"]["temperature"] = 0.5
        else:
            recipe["context_budget"]["max_prompt_tokens"] = 8192
        write(role)
    with pytest.raises(ValueError, match="executed settings"):
        qualified_dose_source(evidence)


@pytest.mark.parametrize("failure", ["wrong_export", "zero_updates", "failed", "tamper"])
def test_unqualified_source_rejects_invalid_completion_or_links(qualification, failure):
    evidence, values, paths, write = qualification
    if failure == "wrong_export":
        values["reload"]["config"]["model"]["location"] = "different-export"
        write("reload")
    elif failure == "zero_updates":
        values["rl"]["result"]["global_step"] = 0
        values["terminal"]["result"]["model"]["global_step"] = 0
        write("rl")
        write("terminal")
    elif failure == "failed":
        path = paths["optimizer"].parent / ".executor_status"
        path.write_bytes(b"FAILED")
        evidence["files"]["optimizer_status"]["sha256"] = hashlib.sha256(b"FAILED").hexdigest()
    else:
        paths["rl"].write_bytes(paths["rl"].read_bytes() + b" ")
    with pytest.raises(ValueError):
        qualified_dose_source(evidence)


def test_resume_revalidates_source_hashes(qualification):
    evidence, _, paths, _ = qualification
    qualified_dose_source(evidence)
    paths["resolved"].write_text("{}")
    with pytest.raises(ValueError):
        qualified_dose_source(evidence)


def test_historical_source_rejects_incompatible_current_runtime(qualification, monkeypatch):
    evidence, _, _, _ = qualification
    monkeypatch.setattr("experiments.post_training.russell_rsi.dose_qualification.MARIN_SKYRL", MARIN_SKYRL)
    with pytest.raises(ValueError, match="declared runtime commit"):
        qualified_dose_source(evidence)
