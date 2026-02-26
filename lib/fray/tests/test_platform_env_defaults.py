# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.v2.types import CpuConfig, GpuConfig, TpuConfig


def test_tpu_defaults_set_libtpu_init_args_for_v5p_and_v6e():
    v5p_env = TpuConfig(variant="v5p-8").default_env_vars()
    v6e_env = TpuConfig(variant="v6e-8").default_env_vars()

    assert v5p_env["LIBTPU_INIT_ARGS"] == "--xla_tpu_scoped_vmem_limit_kib=50000"
    assert v6e_env["LIBTPU_INIT_ARGS"] == "--xla_tpu_scoped_vmem_limit_kib=50000"


def test_v4_tpu_does_not_get_libtpu_init_args():
    env = TpuConfig(variant="v4-8").default_env_vars()
    assert "LIBTPU_INIT_ARGS" not in env


def test_defaults_set_jax_platforms_by_device_kind():
    cpu_env = CpuConfig().default_env_vars()
    gpu_env = GpuConfig(variant="H100").default_env_vars()
    tpu_env = TpuConfig(variant="v5p-8").default_env_vars()

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert gpu_env["JAX_PLATFORMS"] == ""
    assert tpu_env["JAX_PLATFORMS"] == ""


def test_non_tpu_defaults_do_not_include_libtpu_init_args():
    assert "LIBTPU_INIT_ARGS" not in CpuConfig().default_env_vars()
