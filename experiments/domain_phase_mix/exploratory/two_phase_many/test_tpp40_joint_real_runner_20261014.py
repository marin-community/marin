# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_joint_real_round4_20261014 as runner


def test_optimizer_adapter_replays_joint_head_and_keeps_historical_output_isolated(tmp_path: Path) -> None:
    original_output = runner.round2.OUTPUT
    fitted = runner.joint.JointFit(
        model="bounded_benefit_odds",
        alpha=0.8,
        shape={"rate": 0.25, "power": 0.7, "threshold": 3.0},
        parameters={"kappa": 4.0},
        ridge=0.0,
        kappa=2.5,
        known_floor=None,
        inner_rmse=0.01,
        head=runner.joint.mariner.Head(0.2, np.array([0.5, 0.4, 0.03, 0.02]), 0.6),
        selection_profile=(),
        search_space={},
    )
    source = tmp_path / "model.json"
    source.write_text(json.dumps(fitted.to_json()))
    name = runner.MODEL_NAMES[1]

    def load_from_disk(fold: int, task: int) -> dict[str, runner.joint.JointFit]:
        assert (fold, task) == (0, 0)
        return {name: runner.joint.JointFit.from_json(json.loads(source.read_text()))}

    frozen = {"test_protocol": "isolated_optimizer_replay"}
    backend = runner.optimizer_backend(tmp_path, lambda: frozen, load_from_disk)
    restored = backend.load_fit_bundle(0, 0)[name]
    early, late = np.array([[1.0, 2.0], [3.0, 0.0]]), np.array([[0.4, 1.0], [0.0, 2.0]])
    np.testing.assert_array_equal(restored.predict(early, late), fitted.predict(early, late))
    context = {"stage": "adapter_replay"}
    backend.finish(tmp_path, context, ("model.json",), expected_identity=frozen)
    assert backend.completed(tmp_path, context)
    assert runner.round2.OUTPUT == original_output
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError):
        backend.completed(tmp_path, context)
