# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import jax
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference import vllm_async
from marin.inference import vllm_inprocess
from marin.inference import vllm_server
from marin.inference import vllm_tpu_bootstrap_patch
from marin.rl.environments.inference_ctx import startup_debug
from marin.rl.environments.inference_ctx.inflight import worker as inflight_worker


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
        extra_cli_args=["--some-unsupported-flag", "value"],
    )

    assert not eligibility.eligible
    assert "unsupported CLI args" in eligibility.reason


def test_inprocess_eligibility_accepts_served_model_name(monkeypatch) -> None:
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

    eligibility = vllm_inprocess.evaluate_inprocess_eligibility(
        model=_model(),
        model_name_or_path="gs://bucket/model",
        extra_cli_args=["--served-model-name", "my-model"],
    )

    assert eligibility.eligible


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

    assert isinstance(env._backend, vllm_server.ManagedAsyncVllmServerBackend)
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

    def _fail_async_start(self, **kwargs):
        raise RuntimeError("async native failed")

    def _start_native(self, **kwargs):
        return vllm_server.VllmServerHandle(
            server_url="http://127.0.0.1:8000/v1",
            port=8000,
            log_dir="/tmp",
        )

    monkeypatch.setattr(vllm_server.ManagedAsyncVllmServerBackend, "start", _fail_async_start)
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
    assert any("RuntimeError: async native failed" in message for message in emitted_messages)


def test_vllm_environment_raises_without_fallback_when_configured(monkeypatch) -> None:
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

    monkeypatch.setattr(
        vllm_server.ManagedAsyncVllmServerBackend,
        "start",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("async native failed")),
    )
    iris_logs: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        vllm_server,
        "_iris_emit",
        lambda level, source, message: iris_logs.append((level, source, message)),
    )

    env = vllm_server.VllmEnvironment(
        _model(),
        mode="native",
        native_startup_failure_mode="raise",
    )

    with pytest.raises(RuntimeError, match="async native failed"):
        with env:
            pass

    emitted_messages = [message for _, _, message in iris_logs]
    assert any("not falling back" in message for message in emitted_messages)
    assert not any("falling back to subprocess native backend" in message for message in emitted_messages)


def test_build_openai_server_cli_args_sets_async_native_defaults() -> None:
    cli_args = vllm_async._build_openai_server_cli_args(
        model=_model(engine_kwargs={"max_model_len": 4096, "tensor_parallel_size": 4, "enforce_eager": True}),
        bootstrap_model_source="/tmp/bootstrap",
        host="127.0.0.1",
        port=8000,
        served_name="served-name",
    )

    assert "--model" in cli_args
    assert "/tmp/bootstrap" in cli_args
    assert "--served-model-name" in cli_args
    assert "served-name" in cli_args
    assert "--disable-frontend-multiprocessing" in cli_args
    assert "--load-format" in cli_args
    assert "dummy" in cli_args
    assert "--worker-extension-cls" in cli_args
    assert "marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension" in cli_args
    assert "--tensor-parallel-size" in cli_args
    assert "4" in cli_args
    assert "--enforce-eager" in cli_args


def test_configure_async_vllm_environment_sets_defaults(monkeypatch) -> None:
    for key in (
        "MODEL_IMPL_TYPE",
        "MARIN_VLLM_FAST_BOOTSTRAP",
        "MARIN_VLLM_DIRECT_SAMPLING_KEY",
        "TPU_MIN_LOG_LEVEL",
        "TPU_STDERR_LOG_LEVEL",
        "MARIN_VLLM_STARTUP_TIMING",
        "MARIN_VLLM_STARTUP_FAULTHANDLER",
        "MARIN_VLLM_STARTUP_FAULTHANDLER_SECS",
        "SKIP_JAX_PRECOMPILE",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "VLLM_XLA_CACHE_PATH",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
    ):
        monkeypatch.delenv(key, raising=False)

    vllm_async._configure_async_vllm_environment()

    assert vllm_async.os.environ["MODEL_IMPL_TYPE"] == "auto"
    assert vllm_async.os.environ["MARIN_VLLM_FAST_BOOTSTRAP"] == "1"
    assert vllm_async.os.environ["MARIN_VLLM_DIRECT_SAMPLING_KEY"] == "1"
    assert vllm_async.os.environ["TPU_MIN_LOG_LEVEL"] == "3"
    assert vllm_async.os.environ["TPU_STDERR_LOG_LEVEL"] == "3"
    assert vllm_async.os.environ["MARIN_VLLM_STARTUP_TIMING"] == "1"
    assert vllm_async.os.environ["MARIN_VLLM_STARTUP_FAULTHANDLER"] == "1"
    assert vllm_async.os.environ["MARIN_VLLM_STARTUP_FAULTHANDLER_SECS"] == "300"
    assert vllm_async.os.environ["SKIP_JAX_PRECOMPILE"] == "1"
    assert vllm_async.os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
    assert vllm_async.os.environ["JAX_ENABLE_COMPILATION_CACHE"] == "1"
    assert vllm_async.os.environ["JAX_COMPILATION_CACHE_DIR"]
    assert vllm_async.os.environ["VLLM_XLA_CACHE_PATH"] == vllm_async.os.environ["JAX_COMPILATION_CACHE_DIR"]
    assert vllm_async.os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "-1"
    assert vllm_async.os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "2"
    assert vllm_async.os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"


def test_log_async_engine_startup_emits_effective_engine_settings(monkeypatch) -> None:
    events: list[str] = []
    iris_messages: list[tuple[str, str, str]] = []

    class _EngineArgs:
        model = "/tmp/bootstrap"
        tensor_parallel_size = 4
        enforce_eager = True

    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "1")
    monkeypatch.setenv("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("MARIN_VLLM_FAST_BOOTSTRAP", "1")
    monkeypatch.setenv("MARIN_VLLM_DIRECT_SAMPLING_KEY", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setattr(
        vllm_async,
        "_iris_emit",
        lambda level, source, message: iris_messages.append((level, source, message)),
    )

    vllm_async._log_async_engine_startup(
        engine_args=_EngineArgs(),
        requested_model_name_or_path="gs://bucket/model",
        events=events,
    )

    assert len(events) == 1
    assert "bootstrap_model='/tmp/bootstrap'" in events[0]
    assert "requested_model='gs://bucket/model'" in events[0]
    assert "MODEL_IMPL_TYPE='auto'" in events[0]
    assert "MARIN_VLLM_FAST_BOOTSTRAP='1'" in events[0]
    assert "MARIN_VLLM_DIRECT_SAMPLING_KEY='1'" in events[0]
    assert "tensor_parallel_size=4" in events[0]
    assert "enforce_eager=True" in events[0]
    assert "SKIP_JAX_PRECOMPILE='1'" in events[0]
    assert "MARIN_VLLM_STARTUP_TIMING='1'" in events[0]
    assert "VLLM_WORKER_MULTIPROC_METHOD='fork'" in events[0]
    assert "VLLM_ENABLE_V1_MULTIPROCESSING='0'" in events[0]
    assert iris_messages == [
        (
            "I",
            "vllm.async",
            "Creating AsyncLLM engine with bootstrap_model='/tmp/bootstrap' "
            "requested_model='gs://bucket/model' MODEL_IMPL_TYPE='auto' "
            "MARIN_VLLM_FAST_BOOTSTRAP='1' "
            "MARIN_VLLM_DIRECT_SAMPLING_KEY='1' tensor_parallel_size=4 "
            "enforce_eager=True SKIP_JAX_PRECOMPILE='1' "
            "MARIN_VLLM_STARTUP_TIMING='1' "
            "VLLM_WORKER_MULTIPROC_METHOD='fork' "
            "VLLM_ENABLE_V1_MULTIPROCESSING='0'",
        )
    ]


def test_install_marin_fast_tpu_bootstrap_patch_intercepts_llama_dummy_bootstrap(monkeypatch) -> None:
    emitted: list[str] = []
    original_calls: list[str] = []

    fake_jax_module = ModuleType("jax")
    fake_jnp_module = ModuleType("jax.numpy")
    fake_jax_sharding_module = ModuleType("jax.sharding")

    class _NamedSharding:
        pass

    class _PartitionSpec:
        def __init__(self, *parts) -> None:
            self.parts = parts

    class _SingleDeviceSharding:
        def __init__(self) -> None:
            self.device_set = {"fake-device"}

    fake_jax_module.numpy = fake_jnp_module
    fake_jax_sharding_module.NamedSharding = _NamedSharding
    fake_jax_sharding_module.PartitionSpec = _PartitionSpec
    fake_jax_sharding_module.SingleDeviceSharding = _SingleDeviceSharding

    fake_flax_module = ModuleType("flax")
    fake_flax_module.nnx = object()

    fake_model_loader_module = ModuleType("tpu_inference.models.common.model_loader")

    def _original(*args, **kwargs):
        original_calls.append("original")
        return "original"

    fake_model_loader_module._get_nnx_model = _original
    fake_model_loader_module.apply_qwix_on_abstract_model = lambda cfg: False
    fake_model_loader_module.apply_qwix_quantization = lambda *args, **kwargs: args[1]

    fake_common_module = ModuleType("tpu_inference.models.common")
    fake_common_module.model_loader = fake_model_loader_module

    monkeypatch.setitem(sys.modules, "jax", fake_jax_module)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp_module)
    monkeypatch.setitem(sys.modules, "jax.sharding", fake_jax_sharding_module)
    monkeypatch.setitem(sys.modules, "flax", fake_flax_module)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.common", fake_common_module)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.common.model_loader", fake_model_loader_module)
    monkeypatch.setenv("MARIN_VLLM_FAST_BOOTSTRAP", "1")
    monkeypatch.setattr(
        vllm_tpu_bootstrap_patch,
        "_build_abstract_bootstrap_model",
        lambda **kwargs: "fast-bootstrap",
    )

    vllm_tpu_bootstrap_patch.install_marin_fast_tpu_bootstrap_patch(emit=emitted.append)

    class LlamaForCausalLM:
        pass

    class UnsupportedModel:
        pass

    cfg = SimpleNamespace(
        load_config=SimpleNamespace(load_format="dummy"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(quantization_config=None)),
    )

    assert fake_model_loader_module._get_nnx_model(LlamaForCausalLM, cfg, None, None) == "fast-bootstrap"
    assert fake_model_loader_module._get_nnx_model(UnsupportedModel, cfg, None, None) == "original"
    assert original_calls == ["original"]
    assert any("Applying Marin fast TPU bootstrap patch" in message for message in emitted)
    assert any("Using Marin abstract-state flax_nnx path" in message for message in emitted)


def test_build_abstract_bootstrap_model_returns_abstract_state() -> None:
    emitted: list[str] = []

    class _ToyModel(nnx.Module):
        def __init__(self, vllm_config, rng, mesh) -> None:
            del vllm_config, mesh
            self.rng = nnx.Rngs(rng)
            self.linear = nnx.Linear(4, 3, rngs=self.rng)

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    model = vllm_tpu_bootstrap_patch._build_abstract_bootstrap_model(
        model_class=_ToyModel,
        vllm_config=SimpleNamespace(),
        rng=jax.random.PRNGKey(0),
        mesh=mesh,
        jax_module=jax,
        nnx_module=nnx,
        emit=emitted.append,
    )

    _, state = nnx.split(model)
    first_param = next(variable for _, variable in nnx.to_flat_state(state) if isinstance(variable, nnx.Param))

    assert type(first_param.value).__name__ == "ShapeDtypeStruct"
    assert any("START Marin abstract-state bootstrap" in message for message in emitted)
    assert any("END Marin fast-bootstrap model prep" in message for message in emitted)


def test_wrap_timed_instance_method_logs_start_and_end(monkeypatch) -> None:
    logged: list[tuple[str, str, object | None, float | None]] = []

    class _Dummy:
        rank = 3

        def step(self, value: int) -> int:
            return value + 1

    monkeypatch.setattr(
        inflight_worker,
        "_startup_log",
        lambda event, label, subject, elapsed=None: logged.append((event, label, subject, elapsed)),
    )

    inflight_worker._wrap_timed_instance_method(_Dummy, "step", label="Dummy.step")
    result = _Dummy().step(4)

    assert result == 5
    assert [entry[0] for entry in logged] == ["START", "END"]
    assert all(entry[1] == "Dummy.step" for entry in logged)


def test_startup_log_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    class _Dummy:
        rank = 7

    inflight_worker._startup_log("START", "Dummy.step", _Dummy())

    captured = capsys.readouterr()
    assert "[marin-vllm-startup] START Dummy.step" in captured.err
    assert "rank=7" in captured.err


def test_install_timed_delegate_method_logs_start_and_end(monkeypatch) -> None:
    logged: list[tuple[str, str, object | None, float | None]] = []

    class _Worker:
        def load_model(self) -> str:
            return "loaded"

    class _Wrapper:
        rpc_rank = 2

        def __init__(self) -> None:
            self.worker = _Worker()

    monkeypatch.setattr(
        inflight_worker,
        "_startup_log",
        lambda event, label, subject, elapsed=None: logged.append((event, label, subject, elapsed)),
    )

    inflight_worker._install_timed_delegate_method(_Wrapper, "load_model", label="Wrapper.load_model")
    result = _Wrapper().load_model()

    assert result == "loaded"
    assert [entry[0] for entry in logged] == ["START", "END"]
    assert all(entry[1] == "Wrapper.load_model" for entry in logged)


def test_install_dummy_init_progress_instrumentation_logs_aggregated_progress(monkeypatch) -> None:
    class _Param:
        def __init__(self, shape: tuple[int, ...], *, dtype: str = "bf16", device: str = "cpu") -> None:
            self.shape = shape
            self.dtype = dtype
            self.device = device

        def numel(self) -> int:
            total = 1
            for dim in self.shape:
                total *= dim
            return total

    class _Model:
        def __init__(self) -> None:
            self._state = {
                "layers.0.large": _Param((8,)),
                "layers.1.small": _Param((2,)),
            }

        def modules(self) -> list[object]:
            return []

        def state_dict(self) -> dict[str, _Param]:
            return self._state

    class _WeightUtils:
        @staticmethod
        def initialize_dummy_weights(model, model_config, low: float = -1e-3, high: float = 1e-3, seed: int = 1234):
            del model, model_config, low, high, seed

    class _DummyLoaderModule:
        initialize_dummy_weights = _WeightUtils.initialize_dummy_weights

    calls: list[int] = []

    def _initialize_single_dummy_weight(param: _Param, low: float, high: float, seed: int) -> None:
        del low, high, seed
        calls.append(param.numel())

    _WeightUtils.initialize_single_dummy_weight = staticmethod(_initialize_single_dummy_weight)
    logs: list[str] = []

    monkeypatch.setattr(startup_debug, "_DUMMY_INIT_LARGE_PARAM_NUMEL", 4)
    startup_debug.install_dummy_init_progress_instrumentation(
        vllm_dummy_loader_module=_DummyLoaderModule,
        vllm_weight_utils_module=_WeightUtils,
        emit=logs.append,
    )

    _DummyLoaderModule.initialize_dummy_weights(_Model(), None)

    assert calls == [8, 2]
    assert any("START dummy-init.loop" in log for log in logs)
    assert any("dummy-init.param-start index=1/2 name=layers.0.large" in log for log in logs)
    assert any("dummy-init.progress params=2/2" in log or "dummy-init.param-end index=1/2" in log for log in logs)
    assert any("END dummy-init.loop" in log for log in logs)


def test_install_detailed_upstream_tpu_model_runner_load_model_logs_tail(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(vllm_async, "_emit_startup_trace", logs.append)
    monkeypatch.delenv("MARIN_VLLM_DIRECT_SAMPLING_KEY", raising=False)

    class _TPUModelRunner:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()
            self.rng_key = "rng"
            self.mesh = "mesh"
            self.drafter = None
            self.model_config = SimpleNamespace(
                seed=123,
                is_multimodal_model=False,
                hf_config=SimpleNamespace(),
            )
            self.devices = ["fake-device"]

        def load_model(self) -> None:
            raise AssertionError("original load_model should be replaced")

    class _Rngs:
        def __init__(self, key: int) -> None:
            self.key = key

        def params(self) -> tuple[str, int]:
            return ("params", self.key)

    fake_tpu_runner_module = SimpleNamespace(
        TPUModelRunner=_TPUModelRunner,
        get_model=lambda vllm_config, rng_key, mesh: (
            "model_fn",
            "compute_logits_fn",
            "pooler_fn",
            "combine_hidden_states_fn",
            {},
            "state",
            None,
            "model",
        ),
        nnx=SimpleNamespace(Rngs=_Rngs),
        jax=SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        common_utils=SimpleNamespace(hbm_usage_gb=lambda devices: 1.0),
    )

    vllm_async._install_detailed_upstream_tpu_model_runner_load_model(fake_tpu_runner_module)
    runner = _TPUModelRunner()

    runner.load_model()

    assert any("installed detailed early TPUModelRunner.load_model instrumentation" in log for log in logs)
    assert any("START TPUModelRunner.load_model.get_model_tuple" in log for log in logs)
    assert any("END TPUModelRunner.load_model.get_model_tuple" in log for log in logs)
    assert any("END TPUModelRunner.load_model.rng_params_for_sampling" in log for log in logs)
    assert any("END TPUModelRunner.load_model.log_init_model" in log for log in logs)
    assert any("END TPUModelRunner.load_model in" in log for log in logs)
    assert runner.model == "model"
    assert runner.state == "state"
    assert runner.rng_params_for_sampling == ("params", 123)


def test_install_detailed_upstream_tpu_model_runner_load_model_accepts_legacy_7_tuple(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(vllm_async, "_emit_startup_trace", logs.append)
    monkeypatch.delenv("MARIN_VLLM_DIRECT_SAMPLING_KEY", raising=False)

    class _TPUModelRunner:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()
            self.rng_key = "rng"
            self.mesh = "mesh"
            self.drafter = None
            self.model_config = SimpleNamespace(
                seed=123,
                is_multimodal_model=False,
                hf_config=SimpleNamespace(),
            )
            self.devices = ["fake-device"]

        def load_model(self) -> None:
            raise AssertionError("original load_model should be replaced")

    class _Rngs:
        def __init__(self, key: int) -> None:
            self.key = key

        def params(self) -> tuple[str, int]:
            return ("params", self.key)

    fake_tpu_runner_module = SimpleNamespace(
        TPUModelRunner=_TPUModelRunner,
        get_model=lambda vllm_config, rng_key, mesh: (
            "model_fn",
            "compute_logits_fn",
            "pooler_fn",
            {},
            "state",
            None,
            "model",
        ),
        nnx=SimpleNamespace(Rngs=_Rngs),
        jax=SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        common_utils=SimpleNamespace(hbm_usage_gb=lambda devices: 1.0),
    )

    vllm_async._install_detailed_upstream_tpu_model_runner_load_model(fake_tpu_runner_module)
    runner = _TPUModelRunner()

    runner.load_model()

    assert any("get_model_tuple length=7" in log for log in logs)
    assert runner.combine_hidden_states_fn is None
    assert runner.model == "model"
    assert runner.state == "state"


def test_install_detailed_upstream_tpu_model_runner_load_model_can_use_direct_sampling_key(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(vllm_async, "_emit_startup_trace", logs.append)
    monkeypatch.setenv("MARIN_VLLM_DIRECT_SAMPLING_KEY", "1")

    class _TPUModelRunner:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()
            self.rng_key = "rng"
            self.mesh = "mesh"
            self.drafter = None
            self.model_config = SimpleNamespace(
                seed=123,
                is_multimodal_model=False,
                hf_config=SimpleNamespace(),
            )
            self.devices = ["fake-device"]

        def load_model(self) -> None:
            raise AssertionError("original load_model should be replaced")

    fake_tpu_runner_module = SimpleNamespace(
        TPUModelRunner=_TPUModelRunner,
        get_model=lambda vllm_config, rng_key, mesh: (
            "model_fn",
            "compute_logits_fn",
            "pooler_fn",
            {},
            "state",
            None,
            "model",
        ),
        nnx=SimpleNamespace(Rngs=lambda key: SimpleNamespace(params=lambda: ("params", key))),
        jax=SimpleNamespace(random=SimpleNamespace(key=lambda seed: ("direct-key", seed))),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        common_utils=SimpleNamespace(hbm_usage_gb=lambda devices: 1.0),
    )

    vllm_async._install_detailed_upstream_tpu_model_runner_load_model(fake_tpu_runner_module)
    runner = _TPUModelRunner()

    runner.load_model()

    assert any("rng_params_for_sampling mode=direct_key" in log for log in logs)
    assert runner.rng_params_for_sampling == ("direct-key", 123)


def test_install_detailed_upstream_create_kv_caches_logs_substeps(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(vllm_async, "_emit_startup_trace", logs.append)

    class _Jnp:
        @staticmethod
        def empty(*, shape, dtype):
            return {"shape": shape, "dtype": dtype}

    class _Jax:
        @staticmethod
        def jit(fn, out_shardings):
            def _wrapped():
                result = fn()
                result["sharding"] = out_shardings
                return result

            return _wrapped

    class _PartitionSpec:
        def __init__(self, *parts) -> None:
            self.parts = parts

        def __repr__(self) -> str:
            return f"PartitionSpec{self.parts!r}"

    class _NamedSharding:
        def __init__(self, mesh, spec) -> None:
            self.mesh = mesh
            self.spec = spec

        def __repr__(self) -> str:
            return f"NamedSharding(mesh={self.mesh!r}, spec={self.spec!r})"

    fake_kv_cache_module = SimpleNamespace(
        DEFAULT_KV_CACHE_DTYPE="bf16",
        ShardingAxisName=SimpleNamespace(ATTN_DATA="data", ATTN_HEAD="head", MLP_TENSOR="mlp"),
        PartitionSpec=_PartitionSpec,
        NamedSharding=_NamedSharding,
        jnp=_Jnp,
        jax=_Jax,
        get_kv_cache_shape_with_mesh=(
            lambda mesh, total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype, use_mla: (
                total_num_pages,
                page_size,
                actual_num_kv_heads,
                actual_head_dim,
            )
        ),
        create_kv_caches=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("original should be replaced")),
    )

    vllm_async._install_detailed_upstream_create_kv_caches(fake_kv_cache_module)
    caches = fake_kv_cache_module.create_kv_caches(
        num_blocks=8,
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        mesh="mesh",
        layer_names=["layer.0", "layer.1"],
        cache_dtype="bf16",
        use_mla=False,
    )

    assert len(caches) == 2
    assert caches[0]["shape"] == (8, 16, 4, 128)
    assert any("installed detailed early kv_cache.create_kv_caches instrumentation" in log for log in logs)
    assert any("START kv_cache.create_kv_caches.shape" in log for log in logs)
    assert any("END kv_cache.create_kv_caches.make_jit" in log for log in logs)
    assert any("START kv_cache.create_kv_caches.allocate index=1/2" in log for log in logs)
    assert any("END kv_cache.create_kv_caches.allocate index=2/2" in log for log in logs)


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
