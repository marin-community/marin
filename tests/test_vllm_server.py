# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import (
    GPT_OSS_REASONING_PARSER,
    VllmEnvironment,
    VllmServerHandle,
    _engine_kwargs_to_cli_args,
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


def test_engine_kwargs_to_cli_args_ignores_local_staging_flag() -> None:
    args = _engine_kwargs_to_cli_args(
        {
            "stage_remote_model_locally": True,
            "tensor_parallel_size": 4,
        }
    )

    assert args == ["--tensor-parallel-size", "4"]


def test_vllm_environment_stages_remote_model_locally(monkeypatch, tmp_path: Path) -> None:
    model_stage_root = tmp_path / "model-stage"
    model_stage_dir = model_stage_root / "model"
    model_stage_dir.mkdir(parents=True)
    tokenizer_stage_root = tmp_path / "tokenizer-stage"
    tokenizer_stage_dir = tokenizer_stage_root / "model"
    tokenizer_stage_dir.mkdir(parents=True)

    def fake_stage_remote_directory(remote_dir: str, *, prefix: str) -> str:
        if remote_dir == "gs://bucket/model":
            return str(model_stage_dir)
        if remote_dir == "gs://bucket/tokenizer":
            return str(tokenizer_stage_dir)
        raise AssertionError(f"Unexpected remote dir: {remote_dir}")

    monkeypatch.setattr("marin.inference.vllm_server._stage_remote_directory", fake_stage_remote_directory)
    monkeypatch.setattr("marin.inference.vllm_server._get_first_model_id", lambda _url: "staged-model-id")

    captured: dict[str, object] = {}

    class FakeBackend:
        def start(self, **kwargs):
            captured.update(kwargs)
            return VllmServerHandle(server_url="http://127.0.0.1:8000/v1", port=8000, log_dir="/tmp")

        def logs_tail(self, handle, *, max_lines=200):
            return ""

        def diagnostics(self, handle, *, max_lines=200):
            return {}

        def stop(self, handle):
            captured["stopped"] = True

    env = VllmEnvironment(
        ModelConfig(
            name="gpt-oss",
            path="gs://bucket/model",
            engine_kwargs={
                "load_format": "runai_streamer",
                "stage_remote_model_locally": True,
                "tokenizer": "gs://bucket/tokenizer",
                "tensor_parallel_size": 4,
            },
        ),
        mode="native",
    )
    env._backend = FakeBackend()

    with env:
        assert captured["model_name_or_path"] == str(model_stage_dir)
        assert captured["extra_cli_args"] == [
            "--tokenizer",
            str(tokenizer_stage_dir),
            "--tensor-parallel-size",
            "4",
        ]

    assert captured["stopped"] is True
    assert not model_stage_root.exists()
    assert not tokenizer_stage_root.exists()


def test_vllm_environment_adds_gpt_oss_reasoning_parser(monkeypatch) -> None:
    monkeypatch.setattr("marin.inference.vllm_server._get_first_model_id", lambda _url: "gpt-oss-model-id")

    captured: dict[str, object] = {}

    class FakeBackend:
        def start(self, **kwargs):
            captured.update(kwargs)
            return VllmServerHandle(server_url="http://127.0.0.1:8000/v1", port=8000, log_dir="/tmp")

        def logs_tail(self, handle, *, max_lines=200):
            return ""

        def diagnostics(self, handle, *, max_lines=200):
            return {}

        def stop(self, handle):
            captured["stopped"] = True

    env = VllmEnvironment(
        ModelConfig(
            name="gpt-oss",
            path="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
            engine_kwargs={
                "load_format": "runai_streamer",
                "tensor_parallel_size": 4,
            },
        ),
        mode="native",
    )
    env._backend = FakeBackend()

    with env:
        assert captured["extra_cli_args"] == [
            "--load-format",
            "runai_streamer",
            "--tensor-parallel-size",
            "4",
            "--reasoning-parser",
            GPT_OSS_REASONING_PARSER,
        ]

    assert captured["stopped"] is True
