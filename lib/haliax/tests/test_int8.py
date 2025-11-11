# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax.random as jrandom
from chex import assert_trees_all_close

import haliax as hax
from haliax.nn import Linear
from haliax.quantization import Int8DotGeneralOp


def test_int8_is_reasonable():
    In = hax.Axis("In", 8)
    Out = hax.Axis("Out", 8)
    linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), init_scale=0.1)

    int8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=Int8DotGeneralOp.init(), init_scale=0.1)

    input = hax.random.normal(jrandom.PRNGKey(3), In)
    output = linear(input)
    int8_output = int8_linear(input)

    assert output.shape == int8_output.shape
    assert output.dtype == int8_output.dtype

    assert_trees_all_close(output.array, int8_output.array, atol=1e-2, rtol=5e-2)
