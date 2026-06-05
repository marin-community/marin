# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM server configuration."""

from marin.inference.vllm_server import _engine_kwargs_to_cli_args


def test_engine_kwargs_include_tokenizer_cli_arg() -> None:
    """Tokenizer overrides should reach vLLM as well as lm-eval."""
    args = _engine_kwargs_to_cli_args({"tokenizer": "hf-tokenizer", "max_model_len": 8192})

    assert args == ["--tokenizer", "hf-tokenizer", "--max-model-len", "8192"]
