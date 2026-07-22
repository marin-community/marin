# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest
from haliax.quantization import Fp8RaggedDotOp

from experiments.grug.moe.repro_jaxpp_fp8_expert_compile import (
    Config,
    Fp8ExpertLayer,
    accumulate_gradients,
    average_gradients,
    parse_config,
)


def _host_array(shape, dtype, fill):
    return jnp.full(shape, fill, dtype)


def _gradient_layer(weight_value: float, state_value: float) -> Fp8ExpertLayer:
    op = Fp8RaggedDotOp(
        input_scale=_host_array((1,), jnp.float32, state_value),
        output_grad_scale=_host_array((1,), jnp.float32, state_value),
        kernel_scale=_host_array((1,), jnp.float32, state_value),
        input_amax_history=_host_array((4,), jnp.float32, state_value),
        output_grad_amax_history=_host_array((4,), jnp.float32, state_value),
        kernel_amax_history=_host_array((4,), jnp.float32, state_value),
        compute_dtype=None,
        fwd_dtype=jnp.float8_e4m3fn,
        rev_dtype=jnp.float8_e4m3fn,
    )
    return Fp8ExpertLayer(
        w13=jnp.full((1, 2, 4), weight_value, jnp.float32),
        w2=jnp.full((1, 2, 2), weight_value, jnp.float32),
        w13_op=op,
        w2_op=op,
    )


def test_gradient_reduction_adds_weights_maxes_state_and_averages_only_weights() -> None:
    first = (_gradient_layer(2.0, 3.0),)
    second = (_gradient_layer(4.0, 5.0),)

    accumulated = accumulate_gradients(first, second)
    averaged = average_gradients(accumulated, microbatches=2)

    np.testing.assert_array_equal(averaged[0].w13, jnp.full((1, 2, 4), 3.0))
    np.testing.assert_array_equal(averaged[0].w2, jnp.full((1, 2, 2), 3.0))
    np.testing.assert_array_equal(averaged[0].w13_op.input_scale, jnp.full((1,), 5.0))
    np.testing.assert_array_equal(averaged[0].w2_op.output_grad_amax_history, jnp.full((4,), 5.0))


@pytest.mark.parametrize(
    ("argument", "value"),
    (("--tokens", "127"), ("--hidden", "96"), ("--intermediate", "192")),
)
def test_fp8_config_rejects_shapes_that_cannot_lower(argument: str, value: str) -> None:
    with pytest.raises(ValueError, match="must be divisible by 128"):
        parse_config(["--runtime", "direct", "--kernel", "fp8", argument, value])


def test_bf16_control_accepts_non_fp8_aligned_shapes() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--experts",
            "3",
            "--tokens",
            "12",
            "--hidden",
            "6",
            "--intermediate",
            "5",
        ]
    )

    assert config.kernel == "bf16"
    assert config.tokens_per_expert == 4


def test_config_rejects_unequal_expert_groups() -> None:
    config = Config(
        runtime="direct",
        kernel="bf16",
        layers=1,
        experts=3,
        tokens=10,
        hidden=8,
        intermediate=8,
        microbatches=1,
        amax_history=4,
        timeout=30,
        stack_after=10,
        coordinator_port=5793,
        stop_after="execute",
        dump_dir=None,
    )

    with pytest.raises(ValueError, match="tokens=10 must be divisible by experts=3"):
        config.validate()
