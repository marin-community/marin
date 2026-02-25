# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.platform_env_defaults import PlatformEnvDefaultRule, apply_platform_default_env_vars
from fray.v2.types import CpuConfig, GpuConfig, TpuConfig


def test_default_rules_set_libtpu_init_args_for_v5p_and_v6e():
    v5p_env = apply_platform_default_env_vars(TpuConfig(variant="v5p-8"), {})
    v6e_env = apply_platform_default_env_vars(TpuConfig(variant="v6e-8"), {})

    assert v5p_env["LIBTPU_INIT_ARGS"] == "--xla_tpu_scoped_vmem_limit_kib=50000"
    assert v6e_env["LIBTPU_INIT_ARGS"] == "--xla_tpu_scoped_vmem_limit_kib=50000"


def test_default_rules_do_not_override_user_values():
    env = apply_platform_default_env_vars(
        TpuConfig(variant="v5p-8"),
        {"LIBTPU_INIT_ARGS": "--user-specified"},
    )
    assert env["LIBTPU_INIT_ARGS"] == "--user-specified"


def test_default_rules_skip_non_matching_device():
    env = apply_platform_default_env_vars(CpuConfig(), {})
    assert "LIBTPU_INIT_ARGS" not in env


def test_custom_rules_are_supported():
    rules = (
        PlatformEnvDefaultRule(
            env_var="CUDA_DEVICE_MAX_CONNECTIONS",
            value="1",
            device_kind="gpu",
            device_variant_prefixes=("H100",),
        ),
    )
    env = apply_platform_default_env_vars(GpuConfig(variant="H100"), {}, rules=rules)
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
