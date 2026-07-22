# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.train import GrugJaxPPConfig, pack_fp8_pipeline_wire, unpack_fp8_pipeline_wire


@pytest.mark.parametrize(
    ("dtype", "fp8_dtype"),
    (("e4m3", jnp.float8_e4m3fn), ("e5m2", jnp.float8_e5m2)),
)
def test_fp8_pipeline_wire_packs_values_and_current_per_token_scale_in_one_tensor(dtype, fp8_dtype):
    values = jnp.asarray(
        [
            [[-3.0, -1.0, 0.5, 2.0], [0.01, -0.02, 0.03, -0.04]],
            [[16.0, -8.0, 4.0, -2.0], [0.25, -0.5, 0.75, -1.0]],
        ],
        dtype=jnp.bfloat16,
    )

    packed = pack_fp8_pipeline_wire(values, dtype)
    decoded_scales = jax.lax.bitcast_convert_type(packed[..., -4:], jnp.float32)
    expected_scales = jnp.max(jnp.abs(values.astype(jnp.float32)), axis=-1) / jnp.asarray(
        jnp.finfo(fp8_dtype).max, dtype=jnp.float32
    )

    assert packed.shape == (*values.shape[:-1], values.shape[-1] + 4)
    assert packed.dtype == jnp.uint8
    np.testing.assert_array_equal(decoded_scales, expected_scales)


@pytest.mark.parametrize("dtype", ("e4m3", "e5m2"))
def test_fp8_pipeline_wire_zero_tokens_round_trip_without_nan(dtype):
    values = jnp.zeros((2, 3, 16), dtype=jnp.bfloat16)

    restored = unpack_fp8_pipeline_wire(pack_fp8_pipeline_wire(values, dtype), dtype)

    assert restored.dtype == jnp.bfloat16
    np.testing.assert_array_equal(restored, values)
    assert np.isfinite(np.asarray(restored)).all()


@pytest.mark.parametrize(("dtype", "max_relative_l2_error"), (("e4m3", 0.035), ("e5m2", 0.065)))
def test_fp8_pipeline_wire_jitted_round_trip_has_expected_numerical_error(dtype, max_relative_l2_error):
    values = jnp.asarray(np.random.default_rng(0).normal(size=(2, 3, 64)), dtype=jnp.bfloat16)

    def round_trip(value):
        return unpack_fp8_pipeline_wire(pack_fp8_pipeline_wire(value, dtype), dtype)

    restored = jax.jit(round_trip).lower(values).compile()(values)
    values_f32 = np.asarray(values, dtype=np.float32)
    restored_f32 = np.asarray(restored, dtype=np.float32)
    relative_l2_error = np.linalg.norm(restored_f32 - values_f32) / np.linalg.norm(values_f32)

    assert restored.shape == values.shape
    assert restored.dtype == jnp.bfloat16
    assert 0 < relative_l2_error < max_relative_l2_error


def test_fp8_pipeline_wire_config_accepts_only_explicit_mpmd_multimicrobatch_std_1f1b():
    config = GrugJaxPPConfig(
        stages=2,
        microbatches=2,
        schedule="std_1f1b",
        implementation="explicit_mpmd",
        explicit_mpmd_pipeline_wire_format="fp8",
    )

    assert config.explicit_mpmd_pipeline_wire_format == "fp8"

    invalid_configs = (
        {"implementation": "auto", "schedule": "std_1f1b", "microbatches": 2},
        {"implementation": "explicit_mpmd", "schedule": "gpipe", "microbatches": 2},
        {"implementation": "explicit_mpmd", "schedule": "std_1f1b", "microbatches": 1},
    )
    for overrides in invalid_configs:
        with pytest.raises(ValueError, match="FP8 explicit MPMD pipeline wire format"):
            GrugJaxPPConfig(
                stages=2,
                explicit_mpmd_pipeline_wire_format="fp8",
                **overrides,
            )


def test_explicit_mpmd_pipeline_wire_config_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown explicit MPMD pipeline wire format"):
        GrugJaxPPConfig(
            stages=2,
            microbatches=2,
            schedule="std_1f1b",
            implementation="explicit_mpmd",
            explicit_mpmd_pipeline_wire_format="int4",  # type: ignore[arg-type]
        )
