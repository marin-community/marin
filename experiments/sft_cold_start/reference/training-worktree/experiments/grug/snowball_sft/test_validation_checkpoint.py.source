# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.tensorstore_serialization import TensorStoreWriteConfig

from experiments.grug.snowball_sft.validation_checkpoint import SELECTION_KIND, BestValidationCheckpoint


def test_best_weights_survive_resume_and_replacement_from_an_earlier_step(tmp_path):
    root = str(tmp_path / "best")
    selector = BestValidationCheckpoint(root, "eval/loss", TensorStoreWriteConfig())
    first = selector.consider({"weight": jnp.array([1.0, 2.0])}, 20, 3.0)
    assert first is not None
    assert selector.consider({"weight": jnp.array([8.0, 9.0])}, 25, 4.0) is None
    restored = load_checkpoint({"params": {"weight": jnp.zeros(2)}}, first.path)
    np.testing.assert_array_equal(restored["params"]["weight"], [1.0, 2.0])

    # Simulate a process ending after native commit but before best.json updates.
    orphan = root + "/step-40-interrupted"
    save_checkpoint(
        {"params": {"weight": jnp.array([3.0, 4.0])}},
        40,
        orphan,
        metadata={"selection_kind": SELECTION_KIND, "validation_metric": "eval/loss", "validation_score": 2.0},
    )
    resumed = BestValidationCheckpoint(root, "eval/loss", TensorStoreWriteConfig())
    assert resumed.best.path == orphan
    assert json.loads((tmp_path / "best/best.json").read_text())["path"] == orphan
    best = resumed.consider({"weight": jnp.array([5.0, 6.0])}, 30, 1.5)
    assert best is not None
    assert not (tmp_path / "best" / first.path.rsplit("/", 1)[1]).exists()
    assert not (tmp_path / "best/step-40-interrupted").exists()
    restored = load_checkpoint({"params": {"weight": jnp.zeros(2)}}, best.path)
    np.testing.assert_array_equal(restored["params"]["weight"], [5.0, 6.0])
    again = BestValidationCheckpoint(root, "eval/loss", TensorStoreWriteConfig())
    assert again.best.step == 30
    assert again.best.score == 1.5


def test_selector_refuses_a_prefix_with_unowned_checkpoints(tmp_path):
    foreign = str(tmp_path / "step-1")
    save_checkpoint({"params": jnp.array([7.0])}, 1, foreign)
    with pytest.raises(ValueError, match="Unrecognized selection checkpoint"):
        BestValidationCheckpoint(str(tmp_path), "eval/loss", TensorStoreWriteConfig())
    restored = load_checkpoint({"params": jnp.zeros(1)}, foreign)
    np.testing.assert_array_equal(restored["params"], [7.0])
