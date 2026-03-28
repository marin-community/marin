# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.inference.vllm_server import _engine_kwargs_to_cli_args


def test_engine_kwargs_to_cli_args_includes_tensor_parallel_size() -> None:
    args = _engine_kwargs_to_cli_args(
        {
            "load_format": "runai_streamer",
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
        }
    )

    assert args == [
        "--load-format",
        "runai_streamer",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "4096",
    ]
