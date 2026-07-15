# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.coral.batch_calibration import (
    BYTES_PER_GIB,
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)


def test_example_usage():
    batch_size = 128
    parameter_count = 100_000_000
    parameter_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=parameter_count,
        batch_size=batch_size,
        seq_len=2048,
        hidden_dim=1024,
        intermediate_dim=4096,
        num_layers=12,
    )
    optimizer_bytes = adam_optimizer_bytes(parameter_count)
    batch_bytes = batch_memory_bytes(
        parameter_bytes=parameter_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
    )

    batch_config = tpu_batch_config(
        "v5litepod-4",
        batch_size,
        batch_bytes,
    )

    assert parameter_bytes == 400_000_000
    assert optimizer_bytes == 800_000_000
    assert activation_bytes == 43_486_543_872
    assert batch_bytes == 44_686_543_872
    assert batch_config == (-1, 1)


def test_adam_optimizer_bytes():
    assert (
        adam_optimizer_bytes(
            10,
            first_moment_dtype_bytes=2,
            second_moment_dtype_bytes=8,
        )
        == 100
    )


def test_dense_transformer_bytes():
    parameter_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=3,
        batch_size=2,
        seq_len=4,
        hidden_dim=5,
        intermediate_dim=7,
        num_layers=8,
        parameter_dtype_bytes=2,
        activation_dtype_bytes=4,
        activation_layer_fraction=0.5,
        activation_layer_floor=6,
    )

    assert parameter_bytes == 6
    assert activation_bytes == 6144


def test_batch_memory_bytes():
    assert (
        batch_memory_bytes(
            parameter_bytes=1,
            optimizer_bytes=2,
            activation_bytes=2,
            correction_factor=0.5,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("batch_bytes", "expected_config"),
    [
        (64 * BYTES_PER_GIB, (-1, 1)),
        (128 * BYTES_PER_GIB, (16, 2)),
        (160 * BYTES_PER_GIB, (8, 4)),
    ],
)
def test_tpu_batch_config(batch_bytes, expected_config):
    assert (
        tpu_batch_config(
            "v5litepod-4",
            batch_size=128,
            batch_bytes=batch_bytes,
        )
        == expected_config
    )


def test_data_axis_size():
    batch_config = tpu_batch_config(
        "v5litepod-4",
        batch_size=128,
        batch_bytes=128 * BYTES_PER_GIB,
        data_axis_size=2,
    )

    assert batch_config == (32, 2)


def test_nondivisible_batch():
    with pytest.raises(
        ValueError,
        match=r"batch_size \(130\) must be divisible by data_axis_size \(4\)",
    ):
        tpu_batch_config(
            "v5litepod-4",
            batch_size=130,
            batch_bytes=1,
            data_axis_size=4,
        )


def test_minimum_microbatch_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch does not fit on v5litepod-4: even per_device_parallelism=1",
    ):
        tpu_batch_config(
            "v5litepod-4",
            batch_size=4,
            batch_bytes=128 * BYTES_PER_GIB,
        )
