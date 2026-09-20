# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.grug_sft.data_ablation import ablated_group_weights, versioned_run_id


def test_ablated_group_weights_preserves_sft_share_and_relative_weights():
    weights = {"math": 0.4, "code": 0.3, "agents": 0.1}

    result = ablated_group_weights(weights, ["code"])

    assert result.keys() == {"math", "agents"}
    assert math.isclose(math.fsum(result.values()), 0.8)
    assert math.isclose(result["math"] / result["agents"], 4.0)


def test_ablated_group_weights_rejects_unknown_or_complete_exclusion():
    weights = {"math": 0.5, "code": 0.3}

    with pytest.raises(ValueError, match="Unknown SFT groups: agents"):
        ablated_group_weights(weights, ["agents"])
    with pytest.raises(ValueError, match="retain at least one"):
        ablated_group_weights(weights, ["math", "code"])


def test_versioned_run_id_is_order_independent_and_separates_baseline():
    base = "grug-sft"

    baseline = versioned_run_id(base, "v1", [])
    first = versioned_run_id(base, "v1", ["math", "code"])
    second = versioned_run_id(base, "v1", ["code", "math"])

    assert baseline == "grug-sft-v1"
    assert first == second
    assert first != baseline


def test_versioned_run_id_rejects_unsafe_version():
    with pytest.raises(ValueError, match="Version must contain"):
        versioned_run_id("grug-sft", "../../overwrite", [])
