# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.perf.grug_fixed_replay import (
    build_loss_weight,
    repacked_operational_micro_loss,
)
from scripts.perf.grug_levanter_fixed_replay_benchmark import (
    runtime_git_revision,
    tree_finite_evidence,
    validate_hardware_evidence,
    validate_output,
)


def test_runtime_git_revision_uses_clean_iris_launch_provenance(monkeypatch):
    requested = "74635e90aeb917368ca3c53b89754f02b91c8bf3"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )
    monkeypatch.setenv(
        "MARIN_PROVENANCE",
        json.dumps({"base_commit": requested[:10], "tree_hash": "abc123def", "dirty": False}),
    )

    actual, evidence = runtime_git_revision(requested)

    assert actual == requested
    assert evidence == {
        "method": "iris_launch_provenance",
        "base_commit": requested[:10],
        "tree_hash": "abc123def",
        "dirty": False,
    }


def test_runtime_git_revision_rejects_dirty_or_wrong_iris_provenance(monkeypatch):
    requested = "74635e90aeb917368ca3c53b89754f02b91c8bf3"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )
    monkeypatch.setenv(
        "MARIN_PROVENANCE",
        json.dumps({"base_commit": requested[:10], "tree_hash": "abc123def", "dirty": True}),
    )
    with pytest.raises(RuntimeError, match="dirty source bundle"):
        runtime_git_revision(requested)

    monkeypatch.setenv(
        "MARIN_PROVENANCE",
        json.dumps({"base_commit": "deadbeef00", "tree_hash": "abc123def", "dirty": False}),
    )
    with pytest.raises(RuntimeError, match="does not identify requested"):
        runtime_git_revision(requested)


def test_build_loss_weight_matches_skyrl_action_logprob_slice():
    loss_mask = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=np.float32)

    result = build_loss_weight(loss_mask, sequence_length=6)

    np.testing.assert_array_equal(
        result,
        np.asarray(
            [
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_loss_weight_rejects_actions_longer_than_next_token_positions():
    with np.testing.assert_raises(ValueError):
        build_loss_weight(np.ones((1, 4), dtype=np.float32), sequence_length=4)


def test_repacked_operational_loss_uses_token_sum_and_router_mean():
    ce_sums = np.asarray([10.0, 30.0])
    router_aux_losses = np.asarray([2.0, 6.0])

    result = sum(
        repacked_operational_micro_loss(
            ce_sum,
            router_aux,
            global_loss_tokens=40,
            microbatch_count=2,
        )
        for ce_sum, router_aux in zip(ce_sums, router_aux_losses, strict=True)
    )

    assert result == 5.0


def test_tree_finite_evidence_preserves_paths_and_nonfinite_counts():
    tree = {
        "finite": jnp.asarray([1.0, -3.0], dtype=jnp.float32),
        "nested": {"bad": jnp.asarray([jnp.nan, jnp.inf, 2.0], dtype=jnp.float32)},
        "ignored": None,
    }

    evidence = tree_finite_evidence(tree)

    assert evidence["checked_arrays"] == 2
    assert evidence["checked_elements"] == 5
    assert evidence["nonfinite_arrays"] == 1
    assert evidence["nonfinite_elements"] == 2
    assert evidence["max_finite_abs"] == 3.0
    assert evidence["leaves"] == [
        {
            "path": "['finite']",
            "shape": [2],
            "dtype": "float32",
            "elements": 2,
            "finite": True,
            "nonfinite_elements": 0,
            "max_finite_abs": 3.0,
        },
        {
            "path": "['nested']['bad']",
            "shape": [3],
            "dtype": "float32",
            "elements": 3,
            "finite": False,
            "nonfinite_elements": 2,
            "max_finite_abs": 2.0,
        },
    ]


def test_validate_output_requires_nonzero_matched_gradients():
    with pytest.raises(RuntimeError, match="only zero gradients"):
        validate_output("matched_ce", (jnp.asarray(1.0), {"weight": jnp.zeros((4,))}))

    values, finite = validate_output("matched_ce", (jnp.asarray(1.0), {"weight": jnp.ones((4,))}))

    assert values == {"loss": 1.0}
    assert finite["checked_elements"] == 4


def test_validate_hardware_evidence_requires_unique_h100s_on_complete_nodes():
    hardware = [
        {
            "hostname": f"node-{process}",
            "process_index": process,
            "devices": [
                {"uuid": f"gpu-{process}-{device}", "name": "NVIDIA H100 80GB HBM3", "memory_total_mib": 81_559}
                for device in range(8)
            ],
            "jax_devices": [{"id": device} for device in range(8)],
        }
        for process in range(4)
    ]

    validate_hardware_evidence(hardware, expected_world=32)

    hardware[-1]["devices"][-1]["uuid"] = "gpu-0-0"
    with pytest.raises(RuntimeError, match="distinct physical GPU"):
        validate_hardware_evidence(hardware, expected_world=32)
