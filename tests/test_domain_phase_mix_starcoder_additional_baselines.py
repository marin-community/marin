# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.domain_phase_mix import launch_starcoder_additional_baselines as launcher
from experiments.domain_phase_mix import starcoder_additional_baselines as baselines


def test_compute_additional_baselines_two_phase_satisfies_k4_constraint():
    results = baselines.compute_additional_baselines("two_phase_starcoder")
    assert [result.label for result in results] == ["proportional", "olmix_unconstrained", "olmix_k4"]

    proportional, olmix_unconstrained, olmix_k4 = results
    config = baselines.get_topology_config("two_phase_starcoder")

    np.testing.assert_allclose(
        proportional.phase_starcoder_weights,
        [baselines.NATURAL_STARCODER_PROPORTION] * config.n_phases,
    )
    assert proportional.run_id == 97001
    assert olmix_unconstrained.run_id == 97002
    assert olmix_k4.run_id == 97003

    weighted_k4_share = float(np.dot(config.phase_fractions, olmix_k4.phase_starcoder_weights))
    assert weighted_k4_share <= config.max_starcoder_share_k4 + 1e-8
    assert olmix_unconstrained.predicted_objective is not None
    assert olmix_k4.predicted_objective is not None
    assert olmix_unconstrained.phase_starcoder_weights[1] > proportional.phase_starcoder_weights[1]


def test_compute_additional_baselines_three_phase_has_expected_shapes():
    results = baselines.compute_additional_baselines("three_phase_starcoder")
    assert [result.label for result in results] == ["proportional", "olmix_unconstrained", "olmix_k4"]

    config = baselines.get_topology_config("three_phase_starcoder")
    for result in results:
        assert len(result.phase_starcoder_weights) == config.n_phases
        assert all(0.0 <= value <= 1.0 for value in result.phase_starcoder_weights)


def test_prepare_launch_uses_us_central1_resources(monkeypatch):
    fake_baselines = [
        baselines.StarcoderBaseline(
            topology="two_phase_starcoder",
            label="proportional",
            run_id=97001,
            run_name="proportional",
            phase_starcoder_weights=(0.1, 0.1),
        ),
        baselines.StarcoderBaseline(
            topology="two_phase_starcoder",
            label="olmix_unconstrained",
            run_id=97002,
            run_name="olmix_unconstrained",
            phase_starcoder_weights=(0.2, 0.3),
            predicted_objective=0.9,
        ),
    ]
    captured: dict[str, object] = {}

    class DummyExperiment:
        def __init__(self):
            self.resources = None

        def create_weight_configs_step(self, *, configs, summary, seed, name_prefix):
            captured["configs"] = configs
            captured["summary"] = summary
            captured["seed"] = seed
            captured["name_prefix"] = name_prefix
            return "weights-step"

        def create_training_step(self, weight_config, *, name_prefix, run_name):
            captured.setdefault("training_steps", []).append(
                {
                    "run_id": weight_config.run_id,
                    "name_prefix": name_prefix,
                    "run_name": run_name,
                }
            )
            return f"train:{run_name}"

    dummy_experiment = DummyExperiment()
    monkeypatch.setattr(launcher, "compute_additional_baselines", lambda *args, **kwargs: fake_baselines)
    monkeypatch.setattr(launcher, "_create_experiment", lambda topology: dummy_experiment)
    monkeypatch.setattr(launcher, "create_cache_tokenizer_step", lambda **kwargs: ("tokenizer", kwargs))
    monkeypatch.setattr(launcher, "create_cache_eval_datasets_step", lambda **kwargs: ("evals", kwargs))

    name_prefix, steps, summary = launcher.prepare_launch("two_phase_starcoder", name_prefix="launch_prefix")

    assert name_prefix == "launch_prefix"
    assert dummy_experiment.resources.regions == ["us-central1"]
    assert steps == [
        ("tokenizer", steps[0][1]),
        ("evals", steps[1][1]),
        "weights-step",
        "train:proportional",
        "train:olmix_unconstrained",
    ]
    assert [item["label"] for item in summary] == ["proportional", "olmix_unconstrained"]
    assert [item["run_name"] for item in captured["training_steps"]] == ["proportional", "olmix_unconstrained"]


def test_baseline_payload_round_trip(monkeypatch):
    fake_baselines = [
        baselines.StarcoderBaseline(
            topology="two_phase_starcoder",
            label="proportional",
            run_id=97001,
            run_name="proportional",
            phase_starcoder_weights=(0.1, 0.1),
        ),
        baselines.StarcoderBaseline(
            topology="three_phase_starcoder",
            label="olmix_k4",
            run_id=98003,
            run_name="olmix_k4",
            phase_starcoder_weights=(0.1, 0.2, 0.3),
            predicted_objective=0.87,
        ),
    ]

    def fake_compute(topology, **kwargs):
        return [baseline for baseline in fake_baselines if baseline.topology == topology]

    monkeypatch.setattr(launcher, "compute_additional_baselines", fake_compute)

    payload = launcher.build_baseline_payload(["two_phase_starcoder", "three_phase_starcoder"])
    decoded = launcher.decode_baseline_payload(launcher.encode_baseline_payload(payload))

    assert set(decoded) == {"two_phase_starcoder", "three_phase_starcoder"}
    assert decoded["two_phase_starcoder"][0].phase_starcoder_weights == (0.1, 0.1)
    assert decoded["three_phase_starcoder"][0].predicted_objective == 0.87
