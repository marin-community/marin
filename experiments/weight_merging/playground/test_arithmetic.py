# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity with the independently maintained production implementation."""

import pytest
import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, merge_tensor

from experiments.weight_merging.playground.arithmetic import Method, Recipe, merge


@pytest.mark.parametrize("method", list(Method))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (13,), (4, 7), (2, 3, 5)])
def test_reference_matches_marin_across_methods_shapes_and_dtypes(method, dtype, shape):
    rng = torch.Generator().manual_seed(73)
    anchor = torch.randn(shape, generator=rng).to(dtype)
    donors = [torch.randn(shape, generator=rng).to(dtype) for _ in range(3)]
    density = 1 if method in (Method.AVERAGE, Method.TASK_ARITHMETIC) else 0.3
    recipe = Recipe(method, (0.2, 0.3, 0.1), density, 1 if method == Method.AVERAGE else 0.7, 17, 0.5)
    parameters = MergeParameters(
        MergeMethod(method.value), recipe.coefficients, density, recipe.scale, recipe.seed, ties_density=0.5
    )
    expected = merge_tensor(anchor, donors, parameters, tensor_name="model.layer.3.weight")
    actual = merge(anchor, donors, recipe, "model.layer.3.weight")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ties_cancels_zero_sign_votes_and_normalizes_by_agreeing_mass():
    anchor = torch.zeros(3)
    donors = [torch.tensor([2.0, 4.0, -4.0]), torch.tensor([-2.0, -1.0, -2.0])]
    actual = merge(anchor, donors, Recipe(Method.TIES, (0.5, 0.5)), "weight")
    torch.testing.assert_close(actual, torch.tensor([0.0, 4.0, -3.0]), rtol=0, atol=0)


def test_dare_masks_are_independent_of_tensor_processing_order():
    anchor = torch.zeros(127)
    donors = [torch.ones_like(anchor), torch.full_like(anchor, 2)]
    recipe = Recipe(Method.DARE_LINEAR, (0.3, 0.7), density=0.3)
    first = merge(anchor, donors, recipe, "a")
    merge(anchor, donors, recipe, "b")
    torch.testing.assert_close(merge(anchor, donors, recipe, "a"), first, rtol=0, atol=0)
    assert not torch.equal(merge(anchor, donors, recipe, "b"), first)
