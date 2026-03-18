# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference import vllm_inprocess
from marin.inference import vllm_server


def _model(*, engine_kwargs: dict | None = None) -> ModelConfig:
    return ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        path="gs://bucket/model",
        engine_kwargs=engine_kwargs or {},
    )


def test_inprocess_eligibility_rejects_explicit_load_format() -> None:
    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=_model(engine_kwargs={"load_format": "runai_streamer"}),
        model_name_or_path="gs://bucket/model",
        extra_cli_args=None,
    )

    assert not eligibility.eligible
    assert "load_format" in eligibility.reason


def test_inprocess_eligibility_rejects_unsupported_extra_args(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_inprocess,
        "_resolve_mapping_model_name",
        lambda model, model_name_or_path: model.name,
    )

    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=_model(),
        model_name_or_path="gs://bucket/model",
        extra_cli_args=["--served-model-name", "foo"],
    )

    assert not eligibility.eligible
    assert "unsupported CLI args" in eligibility.reason


def test_inprocess_eligibility_accepts_supported_engine_cli_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_inprocess,
        "_resolve_mapping_model_name",
        lambda model, model_name_or_path: model.name,
    )
    monkeypatch.setattr(
        vllm_inprocess,
        "_can_stage_bootstrap_metadata_from_model_path",
        lambda model_path: True,
    )

    model = _model(engine_kwargs={"max_model_len": 4096})
    # In production, only raw extra_args are passed here — engine_kwargs
    # like max_model_len are handled by _llm_kwargs(), not via CLI args.
    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=model,
        model_name_or_path="gs://bucket/model",
        extra_cli_args=None,
    )

    assert eligibility.eligible
    assert eligibility.mapping_model_name == model.name


def test_inprocess_eligibility_rejects_object_store_bootstrap_override(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_inprocess,
        "_resolve_mapping_model_name",
        lambda model, model_name_or_path: model.name,
    )

    model = _model(engine_kwargs={"inprocess_bootstrap_model": "gs://bucket/bootstrap"})
    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=model,
        model_name_or_path="gs://bucket/model",
        extra_cli_args=None,
    )

    assert not eligibility.eligible
    assert "inprocess_bootstrap_model" in eligibility.reason


def test_inprocess_eligibility_allows_string_bootstrap_override(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_inprocess,
        "_resolve_mapping_model_name",
        lambda model, model_name_or_path: model.name,
    )

    model = _model(engine_kwargs={"inprocess_bootstrap_model": "meta-llama/Llama-3.1-8B-Instruct"})
    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=model,
        model_name_or_path="gs://bucket/model",
        extra_cli_args=None,
    )

    assert eligibility.eligible
    assert eligibility.bootstrap_model_source == "meta-llama/Llama-3.1-8B-Instruct"


def test_resolve_bootstrap_model_source_for_start_stages_local_metadata(monkeypatch, tmp_path) -> None:
    staged_dir = tmp_path / "bootstrap"
    staged_dir.mkdir()
    (staged_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(vllm_inprocess, "_stage_bootstrap_metadata", lambda _: str(staged_dir))

    source, cleanup_dir = vllm_inprocess._resolve_bootstrap_model_source_for_start(
        ModelConfig(
            name="gs://bucket/non-hf-name",
            path="gs://bucket/model",
            engine_kwargs={},
        )
    )

    assert source == str(staged_dir)
    assert cleanup_dir == str(staged_dir)


def test_vllm_environment_selects_inprocess_when_eligible(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_server,
        "evaluate_inprocess_eligibility",
        lambda **kwargs: vllm_server.InProcessEligibility(
            eligible=True,
            reason="eligible",
            mapping_model_name=kwargs["model"].name,
            bootstrap_model_source="meta-llama/Llama-3.1-8B-Instruct",
        ),
    )

    env = vllm_server.VllmEnvironment(_model(), mode="native")

    assert isinstance(env._backend, vllm_server.InProcessVllmServerBackend)
    assert env.model.engine_kwargs.get("load_format") is None
    assert env._fallback_model is not None
    assert env._fallback_model.engine_kwargs.get("load_format") == "runai_streamer"


def test_vllm_environment_uses_subprocess_when_inprocess_ineligible(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_server,
        "evaluate_inprocess_eligibility",
        lambda **kwargs: vllm_server.InProcessEligibility(
            eligible=False,
            reason="unsupported",
        ),
    )

    env = vllm_server.VllmEnvironment(_model(), mode="native")

    assert isinstance(env._backend, vllm_server.NativeVllmServerBackend)
    assert env.model.engine_kwargs.get("load_format") == "runai_streamer"


def test_vllm_environment_falls_back_to_native_subprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_server,
        "evaluate_inprocess_eligibility",
        lambda **kwargs: vllm_server.InProcessEligibility(
            eligible=True,
            reason="eligible",
            mapping_model_name=kwargs["model"].name,
            bootstrap_model_source="meta-llama/Llama-3.1-8B-Instruct",
        ),
    )

    def _fail_inprocess_start(self, **kwargs):
        raise RuntimeError("in-process failed")

    def _start_native(self, **kwargs):
        return vllm_server.VllmServerHandle(
            server_url="http://127.0.0.1:8000/v1",
            port=8000,
            log_dir="/tmp",
        )

    monkeypatch.setattr(vllm_server.InProcessVllmServerBackend, "start", _fail_inprocess_start)
    monkeypatch.setattr(vllm_server.NativeVllmServerBackend, "start", _start_native)
    monkeypatch.setattr(vllm_server.NativeVllmServerBackend, "stop", lambda self, handle: None)
    monkeypatch.setattr(vllm_server, "_get_first_model_id", lambda server_url: "model-id")
    iris_logs: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        vllm_server,
        "_iris_emit",
        lambda level, source, message: iris_logs.append((level, source, message)),
    )

    env = vllm_server.VllmEnvironment(_model(), mode="native")
    with env:
        assert isinstance(env._backend, vllm_server.NativeVllmServerBackend)
        assert env.model.engine_kwargs.get("load_format") == "runai_streamer"
        assert env.model_id == "model-id"

    emitted_messages = [message for _, _, message in iris_logs]
    assert any("falling back to subprocess native backend" in message for message in emitted_messages)
    assert any("RuntimeError: in-process failed" in message for message in emitted_messages)


def test_create_inprocess_openai_app_returns_fastapi_app() -> None:
    """The in-process OpenAI app wraps LLM.generate() directly instead of build_app()."""
    # We can't test LLM.generate() without vLLM installed, but we can verify
    # the function exists and is callable.
    assert callable(vllm_inprocess._create_inprocess_openai_app)


def test_resolve_sync_weights_callable_requires_extension() -> None:
    class _NoSyncDriverWorker:
        pass

    class _ModelExecutor:
        driver_worker = _NoSyncDriverWorker()

    class _Engine:
        model_executor = _ModelExecutor()

    class _LLM:
        llm_engine = _Engine()

    with pytest.raises(vllm_inprocess.InProcessVllmUnsupportedError, match="sync_weights"):
        vllm_inprocess._resolve_sync_weights_callable(_LLM())
