"""Regression for chunked-prefill configuration at the Ray/vLLM boundary."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.inference_engines.configuration import InferenceEngineRoleConfig, inference_engine_kwargs_from_config
from skyrl_train.inference_engines import ray_wrapped_inference_engine as engine_factory


@pytest.mark.parametrize("chunked_prefill,token_budget", [(False, 24576), (True, 8192)])
def test_configured_chunked_prefill_reaches_vllm_actor_boot(chunked_prefill, token_budget, tmp_path, monkeypatch):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[3] / "skyrl-train/skyrl_train/config/ppo_base_config.yaml")
    cfg.generator.enable_chunked_prefill = chunked_prefill
    cfg.generator.max_num_batched_tokens = token_budget
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "max_position_embeddings": 24576})
    )
    boot_requests = []

    class BootCaptured(Exception):
        pass

    class LocalActor:
        @classmethod
        def options(cls, **_options):
            return cls

        @staticmethod
        def remote(**kwargs):
            boot_requests.append(kwargs)
            # Stop at the external Ray/vLLM actor boundary, before any GPU process.
            raise BootCaptured

    class LocalPlacement:
        def ready(self):
            return self

    vllm = ModuleType("vllm")
    monkeypatch.setattr(vllm, "__version__", "dev", raising=False)
    registry = ModuleType("vllm.model_executor.models")
    monkeypatch.setattr(
        registry, "ModelRegistry", SimpleNamespace(get_supported_archs=lambda: ["LlamaForCausalLM"]), raising=False
    )
    actor_module = ModuleType("skyrl_train.inference_engines.vllm.vllm_engine")
    monkeypatch.setattr(actor_module, "AsyncVLLMRayActor", LocalActor, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", registry)
    monkeypatch.setitem(sys.modules, "skyrl_train.inference_engines.vllm.vllm_engine", actor_module)
    monkeypatch.setattr(engine_factory.ray, "get", lambda result, **_options: result)
    monkeypatch.setattr(engine_factory, "get_all_env_variables", SimpleNamespace(remote=lambda: {}))
    monkeypatch.setattr(engine_factory, "placement_group", lambda *_args, **_options: LocalPlacement())
    role = InferenceEngineRoleConfig(
        pretrain=str(model),
        backend="vllm",
        num_inference_engines=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        decode_context_parallel_size=1,
        shared_pg=None,
        inference_engine_enable_sleep=False,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]"
    )
    kwargs = inference_engine_kwargs_from_config(
        cfg, tokenizer=tokenizer, role=role, engine_init_kwargs={"max_model_len": 24576}
    )
    with pytest.raises(BootCaptured):
        engine_factory.create_ray_wrapped_inference_engines(**kwargs)
    assert [(request["enable_chunked_prefill"], request["max_num_batched_tokens"]) for request in boot_requests] == [
        (chunked_prefill, token_budget)
    ]
    if not chunked_prefill:
        assert boot_requests[0]["max_num_batched_tokens"] >= boot_requests[0]["max_model_len"]
