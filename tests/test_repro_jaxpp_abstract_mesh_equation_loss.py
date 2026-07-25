# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("jaxpp")

from experiments.grug.moe.repro_jaxpp_abstract_mesh_equation_loss import (
    abstract_mesh_shard_map_jaxpr,
    reproduce_equation_loss,
)


def test_pinned_replace_captured_meshes_drops_abstract_mesh_shard_map_equation():
    original = abstract_mesh_shard_map_jaxpr()
    assert len(original.eqns) == 1
    assert original.eqns[0].primitive.name == "shard_map"

    result = reproduce_equation_loss()

    assert result.original_equations == 1
    assert result.rewritten_equations == 0
    assert result.validation_error is not None
    assert "not defined" in result.validation_error
