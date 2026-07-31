# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace

import experiments.domain_phase_mix.nextgen_validate_candidate as validate_candidate


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def test_load_candidate_for_model_reads_assignment(tmp_path):
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()

    _write_json(
        fit_dir / "candidate_assignments.json",
        {"DS-RE-CEQ": "cand-abc123"},
    )
    _write_json(
        fit_dir / "candidates.json",
        [
            {
                "candidate_id": "cand-abc123",
                "model_name": "DS-RE-CEQ",
                "kind": "schedule",
                "phase_weights": {
                    "phase_0": {"nemotron_full": 0.9, "starcoder": 0.1},
                    "phase_1": {"nemotron_full": 0.8, "starcoder": 0.2},
                    "phase_2": {"nemotron_full": 0.7, "starcoder": 0.3},
                },
                "policy_ref": None,
                "predicted_objective": 0.85,
            }
        ],
    )

    selected = validate_candidate.load_candidate_for_model(str(fit_dir), "DS-RE-CEQ")
    assert selected.candidate_id == "cand-abc123"
    assert selected.phase_weights["phase_2"]["starcoder"] == 0.3


def test_validation_experiment_name_stays_within_wandb_tag_limit():
    name = validate_candidate._validation_experiment_name(
        "DS-RE-CEQ-WITH-A-VERY-LONG-MODEL-NAME-FOR-VALIDATION",
        "cand-deadbeefcafebabe",
    )
    assert len(name) <= 64
    assert name.startswith("nextgen_validation_")
    assert name.endswith("_cand-dea")


def test_main_builds_validation_training_step(monkeypatch, tmp_path):
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    _write_json(fit_dir / "candidate_assignments.json", {"DS-RE-CEQ": "cand-deadbeefcafebabe"})
    _write_json(
        fit_dir / "candidates.json",
        [
            {
                "candidate_id": "cand-deadbeefcafebabe",
                "model_name": "DS-RE-CEQ",
                "kind": "schedule",
                "phase_weights": {
                    "phase_0": {"nemotron_full": 1.0, "starcoder": 0.0},
                    "phase_1": {"nemotron_full": 0.8, "starcoder": 0.2},
                    "phase_2": {"nemotron_full": 0.7, "starcoder": 0.3},
                },
                "policy_ref": None,
                "predicted_objective": 0.85,
            }
        ],
    )

    captured = {}

    class _FakeExperiment:
        def create_training_step(self, *, weight_config, name_prefix, run_name):
            captured["weight_config"] = weight_config
            captured["name_prefix"] = name_prefix
            captured["run_name"] = run_name
            return "train-step"

    def _fake_create_three_phase_experiment(**kwargs):
        captured["experiment_kwargs"] = kwargs
        return _FakeExperiment()

    def _fake_create_cache_tokenizer_step(**kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return "tokenizer-step"

    def _fake_create_cache_eval_datasets_step(**kwargs):
        captured["eval_cache_kwargs"] = kwargs
        return "eval-cache-step"

    monkeypatch.setattr(
        validate_candidate,
        "_parse_args",
        lambda: (
            Namespace(
                loop_name="loop",
                state_root="domain_phase_mix/nextgen",
                model_name="DS-RE-CEQ",
                fit_dir=str(fit_dir),
                name_prefix="my_validation_prefix",
            ),
            [],
        ),
    )
    monkeypatch.setattr(validate_candidate, "marin_prefix", lambda: "gs://marin-eu-west4")
    monkeypatch.setattr(validate_candidate, "create_three_phase_experiment", _fake_create_three_phase_experiment)
    monkeypatch.setattr(validate_candidate, "create_cache_tokenizer_step", _fake_create_cache_tokenizer_step)
    monkeypatch.setattr(validate_candidate, "create_cache_eval_datasets_step", _fake_create_cache_eval_datasets_step)
    monkeypatch.setattr(
        validate_candidate,
        "executor_main",
        lambda *, steps, description: captured.update({"steps": steps, "description": description}),
    )
    monkeypatch.delenv("CI", raising=False)

    validate_candidate.main()

    assert captured["steps"] == ["tokenizer-step", "eval-cache-step", "train-step"]
    assert captured["name_prefix"] == "my_validation_prefix"
    assert captured["run_name"].startswith("validate_ds-re-ceq_cand-dea")
    assert captured["weight_config"].phase_weights["phase_1"]["starcoder"] == 0.2
    assert captured["experiment_kwargs"]["name"] == "nextgen_validation_ds-re-ceq_cand-dea"
    assert captured["experiment_kwargs"]["eval_datasets_cache_path"].startswith("gs://marin-eu-west4/")
    assert captured["eval_cache_kwargs"]["gcs_path"].startswith("gs://marin-eu-west4/")
    assert captured["tokenizer_kwargs"]["gcs_path"].startswith("gs://marin-eu-west4/")
