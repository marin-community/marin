# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.coral.batch_config import (
    BYTES_PER_GIB,
    adam_optimizer_bytes,
    batch_config,
    dense_transformer_bytes,
)


def test_common_usage():
    batch_size = 128
    model_shape = {
        "parameter_count": 100_000_000,
        "seq_len": 2048,
        "hidden_dim": 1024,
        "intermediate_dim": 4096,
        "num_layers": 12,
    }
    param_bytes, activation_bytes = dense_transformer_bytes(batch_size=batch_size, **model_shape)
    optimizer_bytes = adam_optimizer_bytes(model_shape["parameter_count"])

    per_device_parallelism, grad_accum = batch_config(
        "v5litepod-4",
        batch_size,
        param_bytes=param_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
    )

    assert isinstance(per_device_parallelism, int)
    assert isinstance(grad_accum, int)


def test_data_axis_size():
    pdp, grad_accum = batch_config(
        "v5litepod-4",
        batch_size=128,
        param_bytes=0,
        optimizer_bytes=0,
        activation_bytes=128 * BYTES_PER_GIB,
        overhead_factor=1.0,
        data_axis_size=2,
    )

    assert (pdp, grad_accum) == (32, 2)


def test_batch_not_divisible():
    with pytest.raises(
        ValueError,
        match=r"batch_size \(130\) must be divisible by data_axis_size \(4\)",
    ):
        batch_config(
            "v5litepod-4",
            batch_size=130,
            param_bytes=0,
            optimizer_bytes=0,
            activation_bytes=1,
            data_axis_size=4,
        )


def test_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch does not fit on v5litepod-4: even per_device_parallelism=1",
    ):
        batch_config(
            "v5litepod-4",
            batch_size=4,
            param_bytes=0,
            optimizer_bytes=0,
            activation_bytes=128 * BYTES_PER_GIB,
            overhead_factor=1.0,
        )


def test_unknown_tpu():
    with pytest.raises(ValueError, match=r"^Unknown TPU type: v5e-8$"):
        batch_config(
            "v5e-8",
            batch_size=8,
            param_bytes=0,
            optimizer_bytes=0,
            activation_bytes=1,
        )
