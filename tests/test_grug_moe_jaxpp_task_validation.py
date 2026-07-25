# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from experiments.grug.moe import train as grug_train


def test_jaxpp_task_call_jaxpr_validation_accepts_closed_task():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))

    grug_train._check_jaxpp_task_call_jaxpr("fwd_0", call_jaxpr)


def test_jaxpp_task_call_jaxpr_validation_rejects_missing_operand():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))
    malformed_jaxpr = call_jaxpr.jaxpr.replace(invars=())
    malformed_call_jaxpr = call_jaxpr.replace(jaxpr=malformed_jaxpr)

    with pytest.raises(ValueError, match="JaxPP generated invalid task JAXPR 'fwd_0'"):
        grug_train._check_jaxpp_task_call_jaxpr("fwd_0", malformed_call_jaxpr)
