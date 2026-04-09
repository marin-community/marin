# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import (
    GPT_OSS_REASONING_PARSER,
    VllmEnvironment,
    VllmServerHandle,
    _build_vllm_env_dict,
    _engine_kwargs_to_cli_args,
    _vllm_env,
)


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


def test_engine_kwargs_to_cli_args_includes_extended_server_args() -> None:
    args = _engine_kwargs_to_cli_args(
        {
            "tokenizer": "gs://bucket/tokenizer",
            "hf_overrides": {"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            "additional_config": {"skip_quantization": True},
            "tensor_parallel_size": 4,
        }
    )

    assert args == [
        "--tokenizer",
        "gs://bucket/tokenizer",
        "--hf-overrides",
        '{"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"}',
        "--additional-config",
        '{"skip_quantization": true}',
        "--tensor-parallel-size",
        "4",
    ]


def test_vllm_env_builders_apply_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_IMPL_TYPE", raising=False)
    monkeypatch.delenv("TPU_STDERR_LOG_LEVEL", raising=False)

    docker_env = _build_vllm_env_dict(env_overrides={"MODEL_IMPL_TYPE": "flax_nnx"})
    native_env = _vllm_env(env_overrides={"TPU_STDERR_LOG_LEVEL": "1"})

    assert docker_env["MODEL_IMPL_TYPE"] == "flax_nnx"
    assert native_env["MODEL_IMPL_TYPE"] == "vllm"
    assert native_env["TPU_STDERR_LOG_LEVEL"] == "1"


def test_vllm_environment_adds_gpt_oss_reasoning_parser(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start_native_server(**kwargs):
        captured.update(kwargs)
        return VllmServerHandle(server_url="http://127.0.0.1:8000/v1", port=8000, log_dir="/tmp")

    monkeypatch.setattr("marin.inference.vllm_server._get_first_model_id", lambda _url: "gpt-oss-model-id")
    monkeypatch.setattr("marin.inference.vllm_server._start_vllm_native_server", fake_start_native_server)

    env = VllmEnvironment(
        ModelConfig(
            name="local-model",
            path="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
            engine_kwargs={
                "load_format": "runai_streamer",
                "tensor_parallel_size": 4,
            },
        ),
        mode="native",
    )
    env.__enter__()
    env.close()

    assert captured["extra_cli_args"] == [
        "--load-format",
        "runai_streamer",
        "--tensor-parallel-size",
        "4",
        "--reasoning-parser",
        GPT_OSS_REASONING_PARSER,
    ]


def test_vllm_environment_passes_native_server_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start_native_server(**kwargs):
        captured.update(kwargs)
        return VllmServerHandle(server_url="http://127.0.0.1:8000/v1", port=8000, log_dir="/tmp")

    monkeypatch.setattr("marin.inference.vllm_server._get_first_model_id", lambda _url: "model-id")
    monkeypatch.setattr("marin.inference.vllm_server._start_vllm_native_server", fake_start_native_server)

    env = VllmEnvironment(
        ModelConfig(name="local-model", path="/tmp/model", engine_kwargs={}),
        mode="native",
        env_overrides={"MODEL_IMPL_TYPE": "vllm"},
        native_stderr_mode="tee",
    )
    env.__enter__()
    env.close()

    assert captured["env_overrides"] == {"MODEL_IMPL_TYPE": "vllm"}
    assert captured["native_stderr_mode"] == "tee"


def test_vllm_environment_defaults_native_stderr_mode_to_file(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start_native_server(**kwargs):
        captured.update(kwargs)
        return VllmServerHandle(server_url="http://127.0.0.1:8000/v1", port=8000, log_dir="/tmp")

    monkeypatch.setattr("marin.inference.vllm_server._get_first_model_id", lambda _url: "model-id")
    monkeypatch.setattr("marin.inference.vllm_server._start_vllm_native_server", fake_start_native_server)

    env = VllmEnvironment(
        ModelConfig(name="local-model", path="/tmp/model", engine_kwargs={}),
        mode="native",
    )
    env.__enter__()
    env.close()

    assert captured["native_stderr_mode"] == "file"
