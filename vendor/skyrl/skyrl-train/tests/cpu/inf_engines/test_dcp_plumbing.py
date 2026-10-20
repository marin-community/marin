"""Plumbing tests for vLLM engine configuration.

Asserts the engine-launch wiring, with NO GPU and NO real Ray actor / vLLM init:

  (seam / G5) `create_ray_wrapped_inference_engines_from_config` reads
      `cfg.generator.inference_engine_decode_context_parallel_size` and forwards it as
      `decode_context_parallel_size`. This single config-assembly seam is shared by both
      `BasePPOExp` (standard) and `TerminalBenchExp` entrypoints (both inherit
      `_setup_trainer`), so wiring here covers both (G5).

  `create_ray_wrapped_inference_engines` forwards `decode_context_parallel_size` and the
      attention backend to the vLLM actor `.remote(...)` when they are set.

See notes/RL/skyrl/vllm_dcp_rollout_stages/stage1_vllm_support_and_plumbing_scope.md.

Run:
    uv run --isolated --group dev --extra cpu pytest tests/cpu/inf_engines/test_dcp_plumbing.py -v
"""

import sys
import types

import pytest
from omegaconf import OmegaConf
from ray.exceptions import ActorDiedError

import skyrl_train.inference_engines.ray_wrapped_inference_engine as wrapped_engines
import skyrl_train.tokenizer as tokenizer_module
from skyrl_train.utils import validate_generator_cfg
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base

DCP_KEY = "inference_engine_decode_context_parallel_size"


# ===================================================================== seam / G5
def test_from_config_forwards_vllm_engine_options(monkeypatch):
    """The config-assembly seam forwards typed vLLM engine options.

    Covers both entrypoints (BasePPOExp + TerminalBenchExp) since they share this seam.
    """
    captured = {}

    def fake_create(**engine_kwargs):
        captured.update(engine_kwargs)
        return []

    # The function imports create_ray_wrapped_inference_engines lazily from this module,
    # so patch it at the source module.

    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", fake_create)

    # dcp=1 (default): forwarded as 1 (the signature default => not passed to vLLM downstream).
    cfg = get_default_config()
    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)
    assert captured["decode_context_parallel_size"] == 1
    assert captured["inference_engine_enable_sleep"] is True
    assert captured["vllm_attention_backend"] is None
    assert "speculative_config" not in captured["engine_init_kwargs"]
    assert "weight_transfer_config" not in captured["engine_init_kwargs"]

    # dcp=2 with admissible TP: forwarded as 2.
    captured.clear()
    cfg2 = get_default_config()
    cfg2.generator.inference_engine_tensor_parallel_size = 8
    cfg2.generator[DCP_KEY] = 2
    cfg2.generator.vllm_attention_backend = "FLASH_ATTN"
    main_base.create_ray_wrapped_inference_engines_from_config(cfg2, colocate_pg=None, tokenizer=None)
    assert captured["decode_context_parallel_size"] == 2
    assert captured["vllm_attention_backend"] == "FLASH_ATTN"

    captured.clear()
    main_base.create_ray_wrapped_inference_engines_from_config(
        cfg2,
        colocate_pg=None,
        tokenizer=None,
        operation=main_base.EntrypointOperation.GENERATE,
    )
    assert captured["inference_engine_enable_sleep"] is False


def test_standard_entrypoint_identity_survives_python_module_execution(monkeypatch):
    """Online EAGLE remains supported when ``python -m`` names the module ``__main__``."""
    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", lambda **_kwargs: [])
    monkeypatch.setattr(main_base.BasePPOExp, "__module__", "__main__")

    cfg = get_default_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.speculative_decoding = {
        "method": "eagle3",
        "model": {
            "source_uri": "hf://laion/snowball-64k-eagle3-draft-r2egym",
            "source_identity": "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        },
        "num_speculative_tokens": 3,
        "training": {"interval_steps": 1},
    }

    experiment = object.__new__(main_base.BasePPOExp)
    experiment.cfg = cfg
    experiment.colocate_pg = None
    experiment.tokenizer = None

    client = experiment.create_inference_engine_client()

    assert client.backend == "vllm"
    assert client.engines == []


def test_online_eagle_training_uses_vllm_synchronous_scheduling(monkeypatch):
    """SkyRL's async actor API must not turn on vLLM's async scheduler for capture."""
    captured = {}

    def fake_create(**engine_kwargs):
        captured.update(engine_kwargs)
        return []

    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", fake_create)

    cfg = get_default_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.speculative_decoding = {
        "method": "eagle3",
        "model": {
            "source_uri": "hf://laion/snowball-64k-eagle3-draft-r2egym",
            "source_identity": "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        },
        "num_speculative_tokens": 3,
        "training": {"interval_steps": 1},
    }

    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)

    assert captured["engine_init_kwargs"]["async_scheduling"] is False
    assert captured["engine_init_kwargs"]["weight_transfer_config"] == {"backend": "runai_streamer"}




@pytest.mark.parametrize(
    ("load_format", "transport"),
    [(None, "broadcast"), ("dummy", "expert_block"), ("Dummy", "expert_block"), ("dummy", "broadcast")],
)
def test_from_config_selects_weight_load_mode_for_object_store_policy(monkeypatch, load_format, transport):
    captured = {}
    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", lambda **kwargs: captured.update(kwargs) or [])
    cfg = get_default_config()
    cfg.trainer.logger = ["console"]
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.model.path = "/tmp/model-metadata"
    cfg.trainer.policy.model.source_uri = "s3://models/policy"
    cfg.trainer.policy.model.source_identity = "sha256:" + "a" * 64
    cfg.trainer.policy.model.revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
    cfg.trainer.policy.model.tokenizer_path = "/tmp/tokenizer-metadata"

    cfg.generator.weight_sync_transport = transport
    if load_format is not None:
        OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", load_format, force_add=True)

    if load_format is not None and transport == "broadcast":
        with pytest.raises(ValueError, match="weight_sync_transport=expert_block"):
            validate_generator_cfg(cfg)
        return
    validate_generator_cfg(cfg)
    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)

    if load_format is not None:
        assert captured["pretrain"] == "/tmp/model-metadata"
        assert captured["engine_init_kwargs"]["load_format"] == "dummy"
        assert "model_loader_extra_config" not in captured["engine_init_kwargs"]
        assert wrapped_engines.MODEL_METADATA_PATH_KEY not in captured["engine_init_kwargs"]
    else:
        assert captured["pretrain"] == "s3://models/policy"
        assert captured["engine_init_kwargs"]["load_format"] == "runai_streamer"
        assert captured["engine_init_kwargs"]["model_loader_extra_config"] == {"distributed": True}
        assert captured["engine_init_kwargs"][wrapped_engines.MODEL_METADATA_PATH_KEY] == "/tmp/model-metadata"
    assert captured["engine_init_kwargs"]["tokenizer"] == "/tmp/tokenizer-metadata"
    assert "revision" not in captured["engine_init_kwargs"]


def test_from_config_loads_staged_policy_weights_locally(monkeypatch):
    captured = {}
    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", lambda **kwargs: captured.update(kwargs) or [])
    cfg = get_default_config()
    cfg.generator.model_loading = "stage_local"
    cfg.trainer.policy.model.path = "/tmp/staged-policy"
    cfg.trainer.policy.model.source_uri = "s3://models/policy"

    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)

    assert captured["pretrain"] == "/tmp/staged-policy"
    assert captured["engine_init_kwargs"].get("load_format") != "runai_streamer"
    assert "model_loader_extra_config" not in captured["engine_init_kwargs"]


def test_from_config_retries_s3_engine_gang_after_actor_startup_failure(monkeypatch):
    attempts = []

    def fake_create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ActorDiedError()
        return ["ready-engine"]

    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", fake_create)
    cfg = get_default_config()
    cfg.trainer.policy.model.path = "/tmp/model-metadata"
    cfg.trainer.policy.model.source_uri = "s3://models/policy"
    cfg.trainer.policy.model.source_identity = "sha256:" + "a" * 64
    cfg.trainer.model_load_retry.max_retries = 1
    cfg.trainer.model_load_retry.backoff_base_seconds = 0.001
    cfg.trainer.model_load_retry.backoff_cap_seconds = 0.001

    engines = main_base.create_ray_wrapped_inference_engines_from_config(
        cfg,
        colocate_pg=None,
        tokenizer=None,
    )

    assert engines == ["ready-engine"]
    assert len(attempts) == 2
    assert attempts[1]["engine_init_timeout_seconds"] < attempts[0]["engine_init_timeout_seconds"]


@pytest.mark.parametrize(
    ("training", "evaluation", "profile", "expected"),
    [(16, 24, None, 25), (0, None, 5, 6), (None, None, None, 1)],
)
def test_from_config_reserves_enough_rollout_logprobs(monkeypatch, training, evaluation, profile, expected):
    captured = {}
    monkeypatch.setattr(wrapped_engines, "create_ray_wrapped_inference_engines", lambda **kwargs: captured.update(kwargs) or [])
    cfg = get_default_config()
    cfg.generator.sampling_params.logprobs = training
    cfg.generator.eval_sampling_params.logprobs = evaluation
    profiles = {"sampled": {"sampling_params": {"logprobs": profile}}} if profile is not None else None
    OmegaConf.update(
        cfg,
        "trainer.callbacks",
        [{"type": "evaluation", "additional_evaluations": profiles}],
        force_add=True,
    )

    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)

    assert captured["max_logprobs"] == expected






# ===================================================== remote forwarding G1 + G4
class _RemoteCapture:
    """Captures the kwargs passed to the (mocked) vLLM actor .options(...).remote(...)."""

    def __init__(self):
        self.remote_calls = []

    def make_actor_class(self):
        capture = self

        class _Actor:
            @staticmethod
            def options(**_opts):
                class _Bound:
                    @staticmethod
                    def remote(**kwargs):
                        capture.remote_calls.append(kwargs)
                        # Stand-in vLLM actor handle. The disaggregated readiness gate
                        # and context-limit probe call both actor methods.
                        return types.SimpleNamespace(
                            report_engine_hosts=types.SimpleNamespace(remote=lambda *a, **k: None),
                            get_model_max_len=types.SimpleNamespace(remote=lambda *a, **k: "max-model-len-ref"),
                        )

                return _Bound()

        return _Actor


def _run_create(
    monkeypatch,
    dcp: int,
    attention_backend: str | None = None,
    *,
    startup_failure: BaseException | None = None,
    removed_placement_groups: list | None = None,
):
    """Drive the real create_ray_wrapped_inference_engines with Ray/PG/actor mocked.

    Uses tp=1, pp=1 (uni backend) so no real GPU/PG bundle reservation is needed; the
    placement_group + readiness + rendezvous + actor are all stubbed. Returns the
    _RemoteCapture so the caller can inspect the kwargs forwarded to .remote(...).
    """

    capture = _RemoteCapture()
    monkeypatch.setattr(wrapped_engines, "_validate_installed_vllm_for_model", lambda _pretrain: None)

    # Stub the vllm import + actor classes (the module imports them lazily in the vllm branch).
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.20.2rc0.dev0+test"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_engine_mod = types.ModuleType("skyrl_train.inference_engines.vllm.vllm_engine")
    fake_engine_mod.AsyncVLLMRayActor = capture.make_actor_class()
    fake_engine_mod.WorkerWrap = object
    monkeypatch.setitem(sys.modules, "skyrl_train.inference_engines.vllm.vllm_engine", fake_engine_mod)

    # Stub Ray plumbing referenced inside create_ray_wrapped_inference_engines.
    class _FakeRemote:
        def remote(self):
            return None

    monkeypatch.setattr(
        wrapped_engines, "placement_group", lambda bundles, strategy=None: ("PG", tuple(len(bundles) for _ in [0]))
    )
    monkeypatch.setattr(wrapped_engines, "ray_noset_visible_devices", lambda *a, **k: False)
    monkeypatch.setattr(wrapped_engines.ray, "kill", lambda _actor: None)
    removed_placement_groups = [] if removed_placement_groups is None else removed_placement_groups
    monkeypatch.setattr(wrapped_engines, "remove_placement_group", removed_placement_groups.append)

    fake_get_all = types.SimpleNamespace(remote=lambda: None)
    monkeypatch.setattr(wrapped_engines, "get_all_env_variables", fake_get_all)
    monkeypatch.setattr(wrapped_engines, "get_ray_pg_ready_with_timeout", lambda *a, **k: None)

    def fake_ray_get(refs):
        if refs == ["max-model-len-ref"]:
            return [32768]
        return {}

    monkeypatch.setattr(wrapped_engines.ray, "get", fake_ray_get)

    def wait_for_startup(*_args, **_kwargs):
        if startup_failure is not None:
            raise startup_failure

    monkeypatch.setattr(wrapped_engines, "wait_for_inference_engine_startup", wait_for_startup)
    # The real RayWrappedInferenceEngine wrapper is trivial (it only stores the actor
    # handle as .inference_engine_actor), so use it unmocked — the readiness gate reads
    # engine.inference_engine_actor off it.

    wrapped_engines.create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=1,
        model_dtype="bfloat16",
        pretrain="dummy/model",
        seed=0,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=dcp,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=False,
        backend="vllm",
        vllm_attention_backend=attention_backend,
        engine_init_timeout_seconds=60,
    )
    return capture


def test_failed_engine_startup_releases_owned_placement_group(monkeypatch):
    removed_placement_groups = []
    with pytest.raises(ActorDiedError):
        _run_create(
            monkeypatch,
            dcp=1,
            startup_failure=ActorDiedError(),
            removed_placement_groups=removed_placement_groups,
        )

    assert removed_placement_groups == [("PG", (1,))]


def test_dcp_enabled_kwarg_present_in_remote(monkeypatch):
    capture = _run_create(monkeypatch, dcp=2)
    assert len(capture.remote_calls) == 1
    assert capture.remote_calls[0].get("decode_context_parallel_size") == 2


def test_attention_backend_forwarded_to_vllm_actor(monkeypatch):
    capture = _run_create(monkeypatch, dcp=1, attention_backend="FLASH_ATTN")
    assert capture.remote_calls[0]["attention_backend"] == "FLASH_ATTN"
