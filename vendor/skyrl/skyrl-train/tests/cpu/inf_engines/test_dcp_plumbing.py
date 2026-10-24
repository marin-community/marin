import skyrl_train.inference_engines.ray_wrapped_inference_engine as rwie
from skyrl_train.config.numerics import validate_numerics_config
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base
from skyrl_train.inference_engines.vllm.decode_invariant.engine import (
    DECODE_INVARIANT_ATTENTION_BACKEND,
    DECODE_INVARIANT_ENGINE_ENVIRONMENT,
    decode_invariant_engine_init_kwargs,
)


def test_exact_numerics_builds_decode_invariant_policy_engines_from_the_written_config(monkeypatch):
    captured = {}

    def fake_create(**engine_kwargs):
        captured.update(engine_kwargs)
        return []

    # main_base imports create_ray_wrapped_inference_engines inside the function, so the patch goes on its module.
    monkeypatch.setattr(rwie, "create_ray_wrapped_inference_engines", fake_create)

    main_base.create_ray_wrapped_inference_engines_from_config(get_default_config(), colocate_pg=None, tokenizer=None)
    assert captured["vllm_attention_backend"] is None
    assert captured["engine_environment"] == {}

    # trainer.numerics.mode=exact with the Snowball engine settings validates, builds decode-invariant rollout engines
    # and leaves the config as written.
    captured.clear()
    cfg = get_default_config()
    cfg.trainer.numerics.mode = "exact"
    cfg.generator.enforce_eager = False
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.vllm_attention_backend = "FLASH_ATTN"
    cfg.generator.engine_init_kwargs = {"moe_backend": "triton"}
    validate_numerics_config(cfg)
    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)
    assert captured["vllm_attention_backend"] == DECODE_INVARIANT_ATTENTION_BACKEND
    assert captured["engine_environment"] == DECODE_INVARIANT_ENGINE_ENVIRONMENT
    assert (
        captured["engine_init_kwargs"].items()
        >= {"moe_backend": "triton", **decode_invariant_engine_init_kwargs()}.items()
    )
    assert cfg.generator.vllm_attention_backend == "FLASH_ATTN"
    assert "additional_config" not in cfg.generator.engine_init_kwargs
