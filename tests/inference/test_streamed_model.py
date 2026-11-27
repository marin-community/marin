# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exported-model inputs at the native vLLM process boundary."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from jinja2 import Template
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmCompilationCacheMode, VllmEngineConfig
from marin.inference.vllm_backend import VllmBackend
from marin.testing.inference import fake_vllm_server
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from transformers import PreTrainedTokenizerFast

_CONCURRENCY = "RUNAI_STREAMER_CONCURRENCY"
_MEMORY_LIMIT = "RUNAI_STREAMER_MEMORY_LIMIT"


@pytest.fixture
def recorded_vllm_child(tmp_path, monkeypatch):
    """Replace the external executable while keeping the native launch path intact."""
    observed = tmp_path / "launch.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.argv = [sys.argv[0], 'record-launch', {str(observed)!r}, "
        f"{json.dumps([_CONCURRENCY, _MEMORY_LIMIT])!r}, *sys.argv[1:]]\n"
        f"runpy.run_path({fake_vllm_server.__file__!r}, run_name='__main__')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=":")
    monkeypatch.delenv(_CONCURRENCY, raising=False)
    monkeypatch.delenv(_MEMORY_LIMIT, raising=False)
    return observed


@pytest.fixture
def backend():
    return VllmBackend(
        VllmEngineConfig(compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED, startup_timeout_seconds=10)
    )


def _export_spec(weights: str) -> ModelSpec:
    return ModelSpec(
        weights=weights,
        api_model="export",
        num_chips=None,
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=None,
        chat_template_content="{{ messages }}",
    )


@pytest.mark.parametrize(
    "extra_args,parent_environment,child_environment,expected_config,expected_environment",
    [
        pytest.param(
            (), {}, {}, '{"distributed":true}', {_CONCURRENCY: "4", _MEMORY_LIMIT: str(16 << 30)}, id="automatic"
        ),
        pytest.param(
            ('--model-loader-extra-config={"distributed":true}',),
            {_CONCURRENCY: "7"},
            {},
            '{"distributed":true}',
            {_CONCURRENCY: "7", _MEMORY_LIMIT: str(16 << 30)},
            id="ambient-concurrency",
        ),
        pytest.param(
            ("--model-loader-extra-config", '{"distributed":true}'),
            {_CONCURRENCY: "7"},
            {_CONCURRENCY: "9", _MEMORY_LIMIT: str(8 << 30)},
            '{"distributed":true}',
            {_CONCURRENCY: "9", _MEMORY_LIMIT: str(8 << 30)},
            id="caller-environment",
        ),
        pytest.param(
            ("--model-loader-extra-config", '{"distributed":true,"concurrency":2,"memory_limit":8589934592}'),
            {},
            {},
            '{"distributed":true,"concurrency":2,"memory_limit":8589934592}',
            {_CONCURRENCY: None, _MEMORY_LIMIT: None},
            id="explicit-json",
        ),
        pytest.param(
            ("--model-loader-extra-config", '{"distributed":false}'),
            {},
            {},
            '{"distributed":false}',
            {_CONCURRENCY: None, _MEMORY_LIMIT: None},
            id="non-distributed",
        ),
        pytest.param(
            ("--model-loader-extra-config", '{"distributed":true}', '--model-loader-extra-config={"distributed":false}'),
            {},
            {},
            '{"distributed":false}',
            {_CONCURRENCY: None, _MEMORY_LIMIT: None},
            id="last-config-wins",
        ),
    ],
)
def test_streamed_export_child_preserves_explicit_settings(
    recorded_vllm_child,
    backend,
    monkeypatch,
    extra_args,
    parent_environment,
    child_environment,
    expected_config,
    expected_environment,
):
    for key, value in parent_environment.items():
        monkeypatch.setenv(key, value)
    with backend.start(
        _export_spec("s3://fixture/export"), extra_args=extra_args, subprocess_env=child_environment
    ) as env:
        env.wait_until_ready(poll_interval_seconds=0.05)
        observed = json.loads(recorded_vllm_child.read_text())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-loader-extra-config")
    native, _ = parser.parse_known_args(observed["argv"])
    assert native.model_loader_extra_config == expected_config
    assert observed["environment"] == expected_environment


def test_export_uses_separate_tokenizer_tool_template(recorded_vllm_child, backend, tmp_path):
    weights = tmp_path / "export-weights"
    weights.mkdir()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordPiece(vocab={"[UNK]": 0, "a": 1})))
    tokenizer.chat_template = {
        "default": "WRONG TEMPLATE",
        "tool_use": "TOOLS: {% for message in messages %}{{ message.content }}{% endfor %}",
    }
    tokenizer.save_pretrained(tokenizer_dir)
    spec = replace(
        _export_spec(str(weights)),
        chat_template_content=None,
        tokenizer=str(tokenizer_dir),
        tokenizer_revision="tokenizer-pin",
    )
    with backend.start(spec) as env:
        env.wait_until_ready(poll_interval_seconds=0.05)
        observed = json.loads(recorded_vllm_child.read_text())
    assert Template(observed["chat_template"]).render(messages=[{"role": "user", "content": "hello"}]) == "TOOLS: hello"
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    native, _ = parser.parse_known_args(observed["argv"])
    assert Path(native.tokenizer) == tokenizer_dir
    assert native.tokenizer_revision == "tokenizer-pin"
    with pytest.raises(ValueError, match="Couldn't instantiate the backend tokenizer"):
        with backend.start(replace(spec, tokenizer=None)):
            pytest.fail("Weights without tokenizer files must not start a server")
