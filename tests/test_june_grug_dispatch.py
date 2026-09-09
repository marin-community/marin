# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.june_tpu_67b_a2b import dispatch


def test_dispatch_forwards_training_runtime_controls(monkeypatch):
    monkeypatch.setenv("IRIS_PORT_JAX", "19415")
    monkeypatch.setenv("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("UNRELATED_RUNTIME_CONTROL", "ignored")

    forwarded = dispatch._forwarded_env_vars()

    assert forwarded["IRIS_PORT_JAX"] == "19415"
    assert forwarded["LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS"] == "0"
    assert "JAX_PLATFORMS" not in forwarded
    assert "UNRELATED_RUNTIME_CONTROL" not in forwarded
