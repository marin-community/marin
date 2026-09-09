# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.june_tpu_67b_a2b import dispatch


def test_dispatch_forwards_explicit_jax_coordination_port(monkeypatch):
    monkeypatch.setenv("IRIS_PORT_JAX", "19415")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    forwarded = dispatch._forwarded_env_vars()

    assert forwarded["IRIS_PORT_JAX"] == "19415"
    assert "JAX_PLATFORMS" not in forwarded
